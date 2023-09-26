from pathlib import Path
from typing import Tuple

import wandb
import fastcore.all as fc
import torch
from torch import nn
from diffusers import UNet2DModel

from .conv_gru import ConvGRU


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
    is_gru: bool = False
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
        params = dict(
            block_out_channels=(16, 32, 64, 128), # number of channels for each block
            norm_num_groups=8, # number of groups for the normalization layer
            in_channels=num_frames*num_channels, # number of input channels
            out_channels=num_channels, # number of output channels
            )
    elif model_name == "unet_big":
        params = dict(
            block_out_channels=(32, 64, 128, 256), # number of channels for each block
            norm_num_groups=8, # number of groups for the normalization layer
            in_channels=num_frames*num_channels, # number of input channels
            out_channels=num_channels, # number of output channels
            )
    else:
        raise(f"Model name not found: {model_name}, choose between 'unet_small' or 'unet_big'")

    if is_gru:
        params.update(
            in_channels=num_channels + num_channels,
            input_dim=1,
            input_size=(64, 64),
            hidden_size=num_channels)

    return params

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

class UNet2DTemporalCondition(UNet2DModel, WandbModel):
    def __init__(self, 
                 *x,
                 input_dim: int,
                 input_size: Tuple[int, int], 
                 hidden_size: int,
                 device: str = "cuda",
                 **kwargs):
        super().__init__(*x, **kwargs)
        init_unet(self)
        self.temporal_encoder = ConvGRU(input_size=input_size,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_size,
                                        kernel_size=(3, 3),
                                        num_layers=2,
                                        dtype=torch.cuda.FloatTensor,
                                        batch_first=True,
                                        bias = True,
                                        return_all_layers = False).to(device)

    def forward(self, *x, **kwargs):
        # TODO: consider case where x has more than one channel?
        # Get first frames and add the channel dimension to them.
        temporal_input = x[0][:, :-1].unsqueeze(2) # first three images
        _, encoder_hidden_states = self.temporal_encoder(temporal_input.to(self.device))
        conv_lstm_features = encoder_hidden_states[0][0].to(self.device)
        noisy_frame = x[0][:, -1:]
        #print(noisy_frame.shape)
        noise_hidden_state = torch.cat([conv_lstm_features, noisy_frame], dim=1)

        return super().forward(noise_hidden_state, timestep=x[1], **kwargs).sample ## Diffusers's UNet2DConditionModel class
