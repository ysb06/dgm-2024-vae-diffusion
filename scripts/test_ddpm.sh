python -m baseline.eval.ddpm.sample_cond +dataset=cifar10/test dataset.ddpm.data.norm=True dataset.ddpm.model.dim=160 dataset.ddpm.model.dropout=0.3 dataset.ddpm.model.attn_resolutions="16," dataset.ddpm.model.n_residual=3 dataset.ddpm.model.dim_mults="1,2,2,2" dataset.ddpm.model.n_heads=8 dataset.ddpm.evaluation.guidance_weight=0.0 dataset.ddpm.evaluation.seed=0 dataset.ddpm.evaluation.sample_prefix='gpu_0' dataset.ddpm.evaluation.device="gpu:0" dataset.ddpm.evaluation.save_mode='image' dataset.ddpm.evaluation.chkpt_path="outputs\checkpoints\ddpmv2--epoch=4999-loss=0.0196.ckpt" dataset.ddpm.evaluation.type='form1' dataset.ddpm.evaluation.resample_strategy='truncated' dataset.ddpm.evaluation.skip_strategy='quad' dataset.ddpm.evaluation.sample_method='ddpm' dataset.ddpm.evaluation.sample_from='target' dataset.ddpm.evaluation.temp=1.0 dataset.ddpm.evaluation.batch_size=64 dataset.ddpm.evaluation.save_path="outputs/ddpm/samples" dataset.ddpm.evaluation.z_cond=False dataset.ddpm.evaluation.n_samples=2500 dataset.ddpm.evaluation.n_steps=1000 dataset.ddpm.evaluation.save_vae=True dataset.ddpm.evaluation.workers=1 dataset.vae.evaluation.chkpt_path="outputs\checkpoints\vae--epoch=999-train_loss=0.0000.ckpt" dataset.vae.evaluation.expde_model_path="outputs\checkpoints\expde.ckpt"