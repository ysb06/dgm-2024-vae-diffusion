import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.models.diffusion.unet_openai import SuperResModel
from baseline.util import space_timesteps
from baseline.models.diffusion.ddpm_form2 import DDPMv2
from baseline.models.diffusion.spaced_diff import SpacedDiffusion
from baseline.models.diffusion.spaced_diff_form2 import SpacedDiffusionForm2
from baseline.models.vae import VAE


class DDPMWrapper(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae,
        lr=2e-5,
        cfd_rate=0.0,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddpm",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
        # loss_chi=1.0,     # total_loss에서 vae_loss 가중치
        # loss_delta=1.0,   # total_loss에서 ddpm_loss 가중치
        # 위 가중치를 적용할지는 고민
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]

        self.z_cond = z_cond
        self.online_network: SuperResModel = online_network
        self.target_network = target_network
        self.vae: VAE = vae
        self.cfd_rate = cfd_rate

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents

        # Disable automatic optimization
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)
        self.spaced_diffusion = None

    def forward(
        self,
        x: torch.Tensor,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        if self.conditional:
            # VAE로부터 z와 cond 얻기
            # training_step에서와 동일하게 x를 [0,1]로 변환 후 VAE 인코딩/디코딩
            x_01 = (x + 1.0) * 0.5  # [-1,1] -> [0,1]
            mu, logvar = self.vae.encode(x_01)        # VAE는 [0,1] 입력 기대
            z = self.vae.reparameterize(mu, logvar)
            cond_01 = self.vae.decode(z)              # cond_01: [0,1]
            cond = 2 * cond_01 - 1.0                  # cond: [-1,1]

            # clf-free guidance 조건
            if self.training and torch.rand(1)[0] < self.cfd_rate:
                cond = torch.zeros_like(x)
                z = torch.zeros_like(z)

        else:
            # unconditional일 경우 cond, z 없음
            cond = None
            z = None

        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        spaced_nw = (
            SpacedDiffusionForm2
            if isinstance(self.online_network, DDPMv2)
            else SpacedDiffusion
        )
        # For spaced resampling
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)

            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            return self.spaced_diffusion(
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

        # For truncated resampling
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
        return sample_nw.sample(
            x,
            cond=cond,
            z_vae=z,
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )

    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        cond = None
        z = None
        if self.conditional:
            x = (batch + 1.0) * 0.5
            # with torch.no_grad():
            #     mu, logvar = self.vae.encode(x * 0.5 + 0.5)
            #     z = self.vae.reparameterize(mu, logvar)
            #     cond = self.vae.decode(z)
            #     cond = 2 * cond - 1
            # Todo: 이것만으로 충분할 지 확인 필요

            mu, logvar = self.vae.encode(x * 0.5 + 0.5)
            z = self.vae.reparameterize(mu, logvar)
            cond = self.vae.decode(z)
            cond = 2 * cond - 1
            # Normalized 되어서 변화

            # Compute VAE loss
            mse_loss = nn.MSELoss(reduction="sum")
            recons_loss = mse_loss(cond, x)
            kl_loss = self.vae.compute_kl(mu, logvar)
            vae_loss = recons_loss + self.vae.alpha * kl_loss

            # Set the conditioning signal based on clf-free guidance rate
            if torch.rand(1)[0] < self.cfd_rate:
                cond = torch.zeros_like(x)
                z = torch.zeros_like(z)
        else:
            x = batch

        # Sample timepoints
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
        eps_pred = self.online_network(
            x, eps, t, low_res=cond, z=z.squeeze() if self.z_cond else None
        )

        # Compute loss
        ddpm_loss = self.criterion(eps, eps_pred)
        total_loss = ddpm_loss + vae_loss

        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()

        # Scheduler step
        lr_sched.step()

        self.log("vae_loss", vae_loss, prog_bar=True)
        self.log("ddpm_loss", ddpm_loss, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)
        
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = (batch + 1.0) * 0.5
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        if self.eval_mode == "sample":
            x_t = (batch + 1.0) * 0.5

            mu, logvar = self.vae.encode(x_t * 0.5 + 0.5)
            z = self.vae.reparameterize(mu, logvar)

            recons = self.vae(z)
            recons = 2 * recons - 1

            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            if isinstance(self.online_network, DDPMv2):
                x_t = recons + self.temp * torch.randn_like(recons)
        else:
            img = (batch + 1.0) * 0.5
            recons = self.vae.forward_recons(img * 0.5 + 0.5)
            recons = 2 * recons - 1

            # DDPM encoder
            x_t = self.online_network.compute_noisy_input(
                img,
                torch.randn_like(img),
                torch.tensor(
                    [self.online_network.T - 1] * img.size(0), device=img.device
                ),
            )

            if isinstance(self.online_network, DDPMv2):
                x_t += recons

        return (
            self(
                x_t,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            recons,
        )

    def configure_optimizers(self):
        params = list(self.online_network.decoder.parameters()) + list(
            self.vae.parameters()
        )
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Define the LR scheduler (As in Ho et al.)
        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }
