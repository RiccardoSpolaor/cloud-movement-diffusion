from functools import partial
from typing import Callable

import torch
from fastprogress import progress_bar
from diffusers.schedulers import DDIMScheduler

from .model import UNet2D

@torch.no_grad()
def _diffusers_sampler(
    model: UNet2D,
    past_frames: torch.FloatTensor,
    sched: DDIMScheduler,
    **kwargs
    ) -> torch.FloatTensor:
    """
    Get the predicted frame from a UNet2D model using a DDIM scheduler.

    Parameters
    ----------
    model : UNet2D
        The model to sample from.
    past_frames : FloatTensor
        The past frames to condition on.
    sched : DDIMScheduler
        The scheduler to use.

    Returns
    -------
    FloatTensor
        The predicted frame.
    """
    # Set model to eval mode.
    model.eval()
    # Get the device used by the model.
    device = next(model.parameters()).device
    # Create a new frame to condition on.
    new_frame = torch.randn_like(past_frames[:,-1:], dtype=past_frames.dtype, device=device)
    # Store the predicted frames to an empty list.
    preds = []
    # Create a progress bar of the given timestep.
    pbar = progress_bar(sched.timesteps, leave=False)
    # Loop over the timesteps.
    for t in pbar:
        # Update the progress bar.
        pbar.comment = f"DDIM Sampler: frame {t}"
        # Concatenate the past frames with the new frame.
        input = torch.cat([past_frames, new_frame], dim=1)
        # Get the noise.
        noise = model(input, t)
        # Step the scheduler and get the new frame.
        new_frame = sched.step(noise, t, new_frame, **kwargs).prev_sample
        # Append the new frame to the list of predicted frames.
        preds.append(new_frame.float().cpu())
    # Return the last predicted frame.
    return preds[-1]

def ddim_sampler(
    steps: int = 350,
    eta: float = 1.
    ) -> Callable[[UNet2D, torch.FloatTensor], torch.FloatTensor]:
    """Get the DDIM sampler. Faster and a bit better than the built-in sampler.

    Parameters
    ----------
    steps : int, optional
        The number of steps to run the sampler for, by default 350.
    eta : float, optional
        The eta parameter for the DDIM scheduler, by default 1.

    Returns
    -------
    (model: UNet2D, past_frames: FloatTensor) -> FloatTensor
        The DDIM sampler function.
    """
    # Create a new DDIM scheduler.
    ddim_sched = DDIMScheduler()
    # Set the number of timesteps.
    ddim_sched.set_timesteps(steps)
    # Get the partial function for the diffusers sampler.
    return partial(_diffusers_sampler, sched=ddim_sched, eta=eta)
