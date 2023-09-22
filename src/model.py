from pathlib import Path

import wandb
import fastcore.all as fc
import torch
from torch import nn
from diffusers import UNet2DModel


def init_unet(model):
    # TODO: find ref to this
    "From Jeremy's bag of tricks on fastai V2 2023"
    for o in model.down_blocks:
        for p in o.resnets:
            p.conv2.weight.data.zero_()
            for p in fc.L(o.downsamplers): 
                nn.init.orthogonal_(p.conv.weight)

    for o in model.up_blocks:
        for p in o.resnets: p.conv2.weight.data.zero_()

    model.conv_out.weight.data.zero_()

class WandbModel:
    """Class of a model that can be saved to wandb"""
    @classmethod
    def from_checkpoint(
        cls,
        model_params: dict,
        checkpoint_file: str
        ) -> 'WandbModel':
        """
        Load a UNet2D model from a checkpoint file.
        
        Parameters
        ----------
        model_params : dict
            The parameters for the model.
        checkpoint_file : str
            The path to the checkpoint file.
            
        Returns
        -------
        WandbModel
            The loaded model.
        """
        model = cls(**model_params)
        print(f"Loading model from: {checkpoint_file}")
        model.load_state_dict(torch.load(checkpoint_file))
        return model

    @classmethod
    def from_artifact(
        cls,
        model_params: dict,
        artifact_name: str
        ) -> 'WandbModel':
        """
        Load a UNet2D model from a wandb.Artifact, need to be run in
        a wandb run.

        Parameters
        ----------
        model_params : dict
            The parameters for the model.
        artifact_name : str
            The name of the artifact to load the model from.

        Returns
        -------
        WandbModel
            The loaded model.
        """
        "Load a UNet2D model from a wandb.Artifact, need to be run in a wandb run"
        artifact = wandb.use_artifact(artifact_name, type='model')
        artifact_dir = Path(artifact.download())
        chpt_file = list(artifact_dir.glob("*.pth"))[0]
        return cls.from_checkpoint(model_params, chpt_file)

def get_unet_params(
    model_name: str,
    num_frames: int,
    num_channels: int = 1,
    ) -> dict: # TODO parametrize the num frames and num channels
    """
    Get the parameters for a diffuser UNet2D model.
    
    Parameters
    ----------
    model_name : str, optional
        The name of the model to get the parameters for, by default
        'unet_small'.
    num_frames : int, optional
        The number of frames in the input, by default 4*3.
    
    Returns
    -------
    dict
        The parameters for the UNet2D diffuser model.
    """
    if model_name == "unet_small":
        return dict(
            block_out_channels=(16, 32, 64, 128), # number of channels for each block
            norm_num_groups=8, # number of groups for the normalization layer
            in_channels=num_frames*num_channels, # number of input channels
            out_channels=num_channels, # number of output channels
            )
    elif model_name == "unet_big":
        return dict(
            block_out_channels=(32, 64, 128, 256), # number of channels for each block
            norm_num_groups=8, # number of groups for the normalization layer
            in_channels=num_frames*num_channels, # number of input channels
            out_channels=num_channels, # number of output channels
            )
    else:
        raise(f"Model name not found: {model_name}, choose between 'unet_small' or 'unet_big'")

class UNet2D(UNet2DModel, WandbModel):
    """Class of a UNet2D model that can be saved to wandb"""
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a UNet2D model that can be saved to wandb.
        
        Parameters
        ----------
        kwargs : dict
            The parameters for the UNet2D model.
        """
        super().__init__(*args, **kwargs)
        # Initialize the weights of the model.
        init_unet(self)

    def forward(self, *args, **kwargs) -> torch.FloatTensor:
        """Apply the UNet2D model to the input.

        Returns
        -------
        torch.FloatTensor
            The output of the UNet2D model.
        """
        return super().forward(*args, **kwargs).sample ## Diffusers's UNet2DOutput class