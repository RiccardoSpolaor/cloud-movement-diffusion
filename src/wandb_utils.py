from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
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
    # Split the image into a vector.
    #image = torch.cat(image.split(1), dim=-1)
    # Turn the image into a numpy array and remove from GPU.
    image = image.cpu().numpy()
    # Turn the image into a wandb image.
    return wandb.Image(image)

def _scale_images_to_gray(
    frames: torch.FloatTensor,
    scaling_values: Tuple[float, float]) -> torch.LongTensor:
    min_value, max_value = scaling_values
    frames = (frames - min_value) * 255 / (max_value - min_value)
    frames = frames.long()
    frames = frames.clamp(min=0, max=255)
    return frames

def log_images(
    target_frames: torch.FloatTensor,
    predicted_frames: torch.FloatTensor,
    scaling_values: Optional[Tuple[float, float]] = None
    ) -> None:
    """
    Log the sampled and predicted images to wandb.

    Parameters
    ----------
    target_frames : FloatTensor
        The sampled target images of shape (num_images, height, width).
    predicted_frames : FloatTensor
        The images predicted by the model from the sampled target images.
        The shape is (num_images, height, width).
    """
    
    # Concatenate the sampled and predicted images.
    #frames = torch.cat([sample_frames, predicted_frames], dim=1)
    # If the scaler is passed, unscale the images and
    #if scaling_values is not None:
    #    sample_frames = _scale_images_to_gray(sample_frames, scaling_values)
    #    predicted_frames = _scale_images_to_gray(predicted_frames, scaling_values)
    # Plot the sampled and predicted images.
    target_frames = torch.cat(target_frames.split(1), dim=-1).squeeze(0)
    target_frames = target_frames.cpu().numpy() 
    predicted_frames = torch.cat(predicted_frames.split(1), dim=-1).squeeze(0)
    predicted_frames = predicted_frames.cpu().numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    print(target_frames.shape)
    plt.imshow(target_frames, cmap='gray')
    plt.title('Sampled Images')
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(predicted_frames, cmap='gray')
    plt.title('Predicted Images')
    plt.axis('off')
    plt.show()

    # Convert the images to wandb images.
    target_frames = wandb.Image(target_frames, caption='Target Images')
    predicted_frames = wandb.Image(predicted_frames, caption='Predicted Images')
    # Log the images to wandb.
    wandb.log({
        'sampled_images': [target_frames, predicted_frames]})

def save_model(
    model_name: str,
    models_folder: str = './model'
    ) -> None:
    """Save the model in the local models folder and log it to wandb.

    Parameters
    ----------
    model : WandbModel & Module
        The model to save.
    model_name : str
        The name of the model.
    """
    # Get the models folder.
    models_folder = Path(models_folder)
    #if not models_folder.exists():
    #    models_folder.mkdir()
    # Save the model in the local models folder.
    # torch.save(model.state_dict(), models_folder/f'{model_name}.pth')
    # Create an artifact of the model and log it to wandb.
    artifact = wandb.Artifact(model_name, type='model')
    artifact.add_file(models_folder/f'{model_name}.pth')
    wandb.log_artifact(artifact)
