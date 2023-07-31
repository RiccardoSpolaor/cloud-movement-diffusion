from types import SimpleNamespace

# TODO: Some of these parameters are not used, remove them.
config = SimpleNamespace(
    epochs=50, # Number of epochs.
    model_name='unet_small', # Model name to save [unet_small, unet_big].
    strategy='ddpm', # Strategy to use ddpm.
    noise_steps=1_000, # Number of noise steps on the diffusion process.
    sampler_steps=333, # Number of sampler steps on the diffusion process.
    seed=42, # Random seed.
    batch_size=128, # Batch size.
    img_size=64, # Image size.
    device='cuda', # Device to use.
    num_workers=0, # Number of workers for dataloader.
    num_frames=4, # Number of frames to use as input.
    lr=5e-4, # Learning rate.
    validation_days=3, # Number of days to use for validation.
    log_every_epoch=5, # Log every n epochs to wandb.
    n_preds=8, # Number of predictions to make.
    validate_epochs=False, # Whether to validate every epoch.
    )