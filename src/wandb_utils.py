from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import wandb

from .model import WandbModel


def _to_wandb_image(image: torch.FloatTensor) -> wandb.Image:
    """
    Convert a PyTorch tensor representing a set of images to a set
    of wandb images.

    Parameters
    ----------
    image : FloatTensor
        The image to convert.

    Returns
    -------
    Image
        The converted image.
    """
    # TODO: check correctness of comment
    # Split the image into its channels and concatenate them.
    image = torch.cat(image.split(1), dim=-1)
    # Turn the image into a numpy array and remove from GPU.
    image = image.cpu().numpy()
    # Turn the image into a wandb image.
    return wandb.Image(image)

def log_images(
    sample_frames: torch.FloatTensor,
    predicted_frames: torch.FloatTensor,
    scaling_values: Optional[Tuple[float, float]] = None
    ) -> None:
    """
    Log the sampled and predicted images to wandb.

    Parameters
    ----------
    sample_frames : FloatTensor
        The sampled images.
    predicted_frames : FloatTensor
        The images predicted by the model from the sampled images.
    """
    # Concatenate the sampled and predicted images.
    frames = torch.cat([sample_frames, predicted_frames], dim=1)
    # If the scaler is passed, unscale the images and
    if scaling_values is not None:
        min_value, max_value = scaling_values
        frames = (frames - min_value) * 255 / (max_value - min_value)
        frames = frames.long()
    # Convert the images to wandb images.
    wandb_frames = [_to_wandb_image(img) for img in frames]
    # Log the images to wandb.
    wandb.log({'sampled_images': wandb_frames})

def save_model(
    model: Union[WandbModel, torch.nn.Module],
    model_name: str
    ) -> None:
    """Save the model in the local models folder and log it to wandb.

    Parameters
    ----------
    model : WandbModel & Module
        The model to save.
    model_name : str
        The name of the model.
    """
    # Update the model name with the wandb run id.
    model_name = f'{wandb.run.id}_{model_name}'
    # Get the models folder.
    models_folder = Path('models')
    if not models_folder.exists():
        models_folder.mkdir()
    # Save the model in the local models folder.
    torch.save(model.state_dict(), models_folder/f'{model_name}.pth')
    # Create an artifact of the model and log it to wandb.
    artifact = wandb.Artifact(model_name, type='model')
    artifact.add_file(f'models/{model_name}.pth')
    wandb.log_artifact(artifact)
