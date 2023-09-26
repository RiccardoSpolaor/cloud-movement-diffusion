import os
from typing import Union
from torch import nn

from .model import WandbModel
from .wandb_utils import save_model


class Checkpoint():
    """Class to handle the checkpoints of a model
    
    Arguments
    ---------
    lowest_error : float
        The lowest error of the model predictions obtained by the
        model so far.
    path : str
        The path where the checkpoints will be saved/loaded.
    """
    def __init__(
        self,
        models_folder: str = './model',
        initial_error: float = float('inf')
        ) -> None:
        """Initialize the checkpoint instance.

        Parameters
        ----------
        dir_path : str, optional
            The checkpoint directory path.
        initial_error : float, optional
            The initial error value, by default inf.
        """
        self.lowest_error = initial_error
        self.models_folder = models_folder

    def save_best(
        self,
        model: Union[WandbModel, nn.Module],
        model_name: str,
        new_error: float,
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
        if new_error < self.lowest_error:
            # Create the checkpoints
            save_model(model, model_name, self.models_folder)

            # Update the lowest error.
            self.lowest_error = new_error
