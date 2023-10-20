from typing import Union
import torch
from torch import nn
from pathlib import Path

from .model import WandbModel


class Checkpoint():
    """Class to handle the checkpoints of a model
    
    Arguments
    ---------
    best_accuracy : float
        The best accuracy of the model predictions obtained so far.
    path : str
        The path where the checkpoints will be saved/loaded.
    """
    def __init__(
        self,
        models_folder: str = './model',
        initial_accuracy: float = 0.
        ) -> None:
        """Initialize the checkpoint instance.

        Parameters
        ----------
        dir_path : str, optional
            The checkpoint directory path.
        initial_accuray : float, optional
            The initial accuracy value, by default 0.
        """
        self.best_accuracy = initial_accuracy
        self.models_folder = models_folder

    def save_best(
        self,
        model: Union[WandbModel, nn.Module],
        model_name: str,
        val_mse: float,
        val_m_csi: float,
        val_psnr: float,
        val_ssim: float,
        ) -> None:
        """
        Possibly save the best model weights according to the new value of
        the metric.
        
        Parameters
        ----------
        model : Module & WandbModel
            The model which weights are saved.
        model_name : str
            The name of the model.
        new_error : float
            The new error value which is compared to the best so
            far. The checkpoints are updated solely if the new
            error is less than the lowest one saved so far.
        """
        if val_m_csi > self.best_accuracy:
            # Get the models folder.
            models_folder = Path(self.models_folder)
            if not models_folder.exists():
                models_folder.mkdir()
            # Save the model in the local models folder.
            torch.save(
                {
                    'val_mse': val_mse,
                    'val_m_csi': val_m_csi,
                    'val_psnr': val_psnr,
                    'val_ssim': val_ssim,
                    'model_state_dict': model.state_dict(),
                    'model_parameters': sum(p.numel() for p in model.parameters())
                },
                models_folder/f'{model_name}.pth'
                )

            # Update the best accuracy.
            self.best_accuracy = val_m_csi
