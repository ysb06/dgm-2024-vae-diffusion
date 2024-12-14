import copy

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from baseline.models.diffusion.ddpm import DDPM
from baseline.models.vae import VAE
from baseline.util import configure_device, get_dataset
from baseline.models.diffusion import SuperResModel, UNetModel
from hybrid_vd.models.diffusion.wrapper import DDPMWrapper


class DiffuseVAE(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)

        config.ddpm.decoder.attention_resolutions = self._parse_str(
            config.ddpm.decoder.attention_resolutions
        )
        config.ddpm.decoder.channel_mult = self._parse_str(config.ddpm.decoder.channel_mult)

        self.vae_layer = VAE(**config.vae.model)
        self.ddpm_decoder = SuperResModel(**config.ddpm.decoder)
        self.ddpm_ema_decoder = copy.deepcopy(self.ddpm_decoder)
        for param in self.ddpm_ema_decoder.parameters():
            param.requires_grad = False
        self.online_ddpm_layer = DDPM(self.ddpm_decoder, **config.ddpm.model)
        self.target_ddpm_layer = DDPM(self.ddpm_ema_decoder, **config.ddpm.model)
        self.ddpm_wrapper_layer = DDPMWrapper(
            self.online_ddpm_layer,
            self.target_ddpm_layer,
            self.vae_layer,
            **config.ddpm.wrapper,
        )


    def _parse_str(self, s: str):
        split = s.split(",")
        return [int(s) for s in split if s != "" and s is not None]


class JointModule(pl.LightningModule):
    """ChatGPT가 제안한 모델"""
    def __init__(
        self,
        vae_config,
        ddpm_config,
        alpha=1.0,  # VAE Loss 가중치 (KL 항에 이미 alpha가 있으나 필요하다면 명칭 변경)
        beta=0.1,  # DDPM Loss 가중치
        lr=1e-4,
        loss_type="l2",  # ddpm loss type: 'l1' or 'l2'
    ):
        super().__init__()
        self.save_hyperparameters()

        # VAE 초기화
        self.vae = VAE(
            input_res=vae_config["data"]["image_size"],
            enc_block_str=vae_config["model"]["enc_block_config"],
            dec_block_str=vae_config["model"]["dec_block_config"],
            enc_channel_str=vae_config["model"]["enc_channel_config"],
            dec_channel_str=vae_config["model"]["dec_channel_config"],
            alpha=vae_config["training"]["alpha"],
            lr=vae_config["training"][
                "lr"
            ],  # 여기서는 사용하지 않음, joint training이므로 공통 lr 사용
        )

        # DDPM 초기화
        self.ddpm = DDPM(
            decoder=None,  # 아래에서 유닛모델 정의 필요
            beta_1=ddpm_config["model"]["beta1"],
            beta_2=ddpm_config["model"]["beta2"],
            T=ddpm_config["model"]["n_timesteps"],
            var_type="fixedlarge",
        )

        # DDPM의 decoder(UNetModel) 생성
        attn_resolutions = self._parse_str(ddpm_config["model"]["attn_resolutions"])
        dim_mults = self._parse_str(ddpm_config["model"]["dim_mults"])
        self.ddpm.decoder = UNetModel(
            in_channels=ddpm_config["data"]["n_channels"],
            model_channels=ddpm_config["model"]["dim"],
            out_channels=3,
            num_res_blocks=ddpm_config["model"]["n_residual"],
            attention_resolutions=attn_resolutions,
            channel_mult=dim_mults,
            use_checkpoint=False,
            dropout=ddpm_config["model"]["dropout"],
            num_heads=ddpm_config["model"]["n_heads"],
        )

        # 손실 함수 설정
        if loss_type == "l2":
            self.ddpm_criterion = nn.MSELoss(reduction="mean")
        else:
            self.ddpm_criterion = nn.L1Loss(reduction="mean")

        self.alpha = alpha
        self.beta = beta
        self.lr = lr

        # manual_backward 사용
        self.automatic_optimization = False

    def _parse_str(self, s):
        if s.endswith(","):
            s = s[:-1]
        return [int(a) for a in s.split(",") if a != ""]

    def vae_loss_fn(self, x):
        # VAE loss 계산용 함수화
        # VAE: recons_loss + alpha * kl_loss
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        decoder_out = self.vae.decode(z)
        mse_loss = F.mse_loss(decoder_out, x, reduction="sum")
        kl_loss = self.vae.compute_kl(mu, logvar)

        recons_loss = mse_loss
        total_vae_loss = recons_loss + self.vae.alpha * kl_loss

        return total_vae_loss, recons_loss, kl_loss

    def ddpm_loss_fn(self, x):
        # DDPM loss는 epsilon 예측 정확도 기준
        # Forward: t 랜덤 샘플, eps = gaussian noise
        B = x.size(0)
        t = torch.randint(0, self.ddpm.T, size=(B,), device=x.device)
        eps = torch.randn_like(x)

        # eps_pred = ddpm.forward(x_0, eps, t) 형태로 노이즈 예측
        # ddpm.forward의 인자: forward(self, x, eps, t, low_res=None, z=None)
        # 여기서는 unconditional이므로 low_res=None, z=None 사용
        eps_pred = self.ddpm(x, eps, t, low_res=None, z=None)
        ddpm_loss = self.ddpm_criterion(eps_pred, eps)
        return ddpm_loss

    def training_step(self, batch, batch_idx):
        # batch: (N,3,32,32) CIFAR-10 이미지 [-1,1] 범위인지 check 필요
        # 만약 get_dataset이 [-1,1]로 정규화했다면 그대로 사용
        # VAE는 [0,1] 범위로 가정했었음 -> 여기서 diff를 맞추기 위해
        # VAE 학습 시 입력 x는 원래 [0,1] 범위였음 (vae 코드 참조)
        # ddpm 코드에서는 [-1,1]을 사용함
        # Joint training 시 하나로 통일 필요.
        # 여기서는 CIFAR-10을 [-1,1]로 읽는다고 가정하면,
        # VAE에 들어갈 때 [0,1]로 바꿔야 함: x_vae = (x+1)/2
        x = batch
        x_vae = (x + 1.0) * 0.5  # [-1,1] -> [0,1]

        # VAE loss
        vae_total_loss, recons_loss, kl_loss = self.vae_loss_fn(x_vae)

        # DDPM loss
        ddpm_loss = self.ddpm_loss_fn(x)

        # Joint loss
        total_loss = vae_total_loss * self.alpha + ddpm_loss * self.beta

        # Optimize
        optim = self.optimizers()
        optim.zero_grad()
        self.manual_backward(total_loss)
        optim.step()

        # 로깅
        self.log("VAE_total_loss", vae_total_loss, prog_bar=True)
        self.log("DDPM_loss", ddpm_loss, prog_bar=True)
        self.log("Joint_loss", total_loss, prog_bar=True)
        self.log("recons_loss", recons_loss)
        self.log("kl_loss", kl_loss)

        return total_loss

    def configure_optimizers(self):
        # VAE와 DDPM 파라미터를 모두 학습
        params = list(self.vae.parameters()) + list(self.ddpm.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        return optimizer


if __name__ == "__main__":
    # 예: Hydra나 argparse를 통해 config를 로딩하는 대신 간단히 dict로 예시
    vae_config = {
        "data": {
            "root": "./datasets",
            "name": "cifar10",
            "image_size": 32,
        },
        "model": {
            "enc_block_config": "32x7,32d2,32t16,16x4,16d2,16t8,8x4,8d2,8t4,4x3,4d4,4t1,1x3",
            "enc_channel_config": "32:64,16:128,8:256,4:256,1:512",
            "dec_block_config": "1x1,1u4,1t4,4x2,4u2,4t8,8x3,8u2,8t16,16x7,16u2,16t32,32x15",
            "dec_channel_config": "32:64,16:128,8:256,4:256,1:512",
        },
        "training": {
            "alpha": 1.0,
            "lr": 1e-4,
        },
    }

    ddpm_config = {
        "data": {
            "root": "./datasets",
            "name": "cifar10",
            "image_size": 32,
            "hflip": True,
            "n_channels": 3,
            "norm": True,
        },
        "model": {
            "dim": 128,
            "attn_resolutions": "16,",
            "n_residual": 2,
            "dim_mults": "1,2,2,2",
            "dropout": 0.3,
            "n_heads": 8,
            "beta1": 0.0001,
            "beta2": 0.02,
            "n_timesteps": 1000,
        },
    }

    device_str = "gpu:0"
    dev, _ = configure_device(device_str)
    dataset = get_dataset("cifar10", "./datasets", 32, norm=True, flip=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    model = JointModule(
        vae_config, ddpm_config, alpha=1.0, beta=0.1, lr=1e-4, loss_type="l2"
    )
    model = model.to(dev)

    trainer = pl.Trainer(
        max_epochs=10, log_every_n_steps=10, devices=1, accelerator="gpu"
    )
    trainer.fit(model, loader)
