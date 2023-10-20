from pathlib import Path
import random
from typing import Any, Dict, Tuple, Literal

import wandb
import fastcore.all as fc
import torch
from torch import nn
from diffusers import UNet2DModel

from .conv_gru import ConvGRU
from .unet3d import SpaceTimeUnet


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
    num_input_frames: int,
    num_output_frames: int,
    num_channels: int
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
            in_channels=num_input_frames*num_channels, # number of input channels
            out_channels=num_output_frames*num_channels, # number of output channels
            )
    elif model_name == "unet_big":
        params = dict(
            block_out_channels=(32, 64, 128, 256), # number of channels for each block
            norm_num_groups=8, # number of groups for the normalization layer
            in_channels=num_input_frames*num_channels, # number of input channels
            out_channels=num_output_frames*num_channels, # number of output channels
            )
    else:
        raise(f"Model name not found: {model_name}, choose between 'unet_small' or 'unet_big'")

    return params

def add_gru_params(
    params: dict,
    num_input_past_frames: int,
    num_input_prediction_frames: int,
    num_channels: int,
    mode: Literal[
        'last_output',
        'all_outputs',
        'last_output_and_last_frame',
        'interleave_frames_and_outputs'],
    ) -> dict:
    if mode == 'last_output':
        in_channels = 1 * num_channels + num_input_prediction_frames * num_channels
    elif mode == 'all_outputs':
        in_channels = num_input_past_frames * num_channels + num_input_prediction_frames * num_channels
    elif mode == 'last_output_and_last_frame':
        in_channels = 2 * num_channels + num_input_prediction_frames * num_channels
    elif mode == 'interleave_frames_and_outputs':
        num_input_past_frames * 2 * num_channels + num_input_prediction_frames * num_channels
    else:
        raise(f"Mode not found: {mode}, choose between 'last_output', 'all_outputs', 'last_output_and_last_frame', 'interleave_frames_and_outputs'")

    params.update(
        in_channels=in_channels,
        input_dim=num_channels,
        input_size=(64, 64),
        hidden_size=num_channels,
        mode=mode)

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


class UNet3D(nn.Module, WandbModel):
    def __init__(
        self,
        *x,
        input_size: Tuple[int, int, int],
        n_past_frames: int,
        n_prediction_frames: int,
        **kwargs):
        super().__init__(*x, **kwargs)
        self.unet = SpaceTimeUnet(
            dim = 64,
            num_input_frames=n_past_frames + n_prediction_frames,
            num_prediction_frames=n_prediction_frames,
            channels = input_size[0],
            dim_mult = (1, 2, 4, 8),
            temporal_compression = (False, False, False, True),
            self_attns = (False, False, False, True),
            condition_on_timestep=True)

    def forward(self, *x, **kwargs):
        #print(x[0].shape, x[1].shape)
        #temporal_input = x[0] # first 4 images
        #noisy_frames = x[1].unsqueeze(2)
        #input = torch.cat([temporal_input, noisy_frames], dim=1)        
        input = x[0].permute((0,2,1,3,4))
        
        timesteps = x[1]
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=input.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(input.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(input.shape[0], dtype=torch.float32, device=timesteps.device)
        out = self.unet(input, timesteps)#, x[1].type(torch.float32))
        out = out.permute(0,2,1,3,4)
        return out
'''
class UNet3D(UNet3DModel, WandbModel):
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
'''
class UNet2DTemporalCondition(UNet2DModel, WandbModel):
    def __init__(
        self, 
        *x,
        input_dim: int,
        input_size: Tuple[int, int], 
        hidden_size: int,
        device: str = "cuda",
        mode: Literal[
            'last_output',
            'all_outputs',
            'last_output_and_last_frame',
            'interleave_frames_and_outputs'],
        **kwargs):
        super().__init__(*x, **kwargs)
        init_unet(self)
        self.temporal_encoder = ConvGRU(
            input_size=input_size,
            input_dim=input_dim,
            hidden_dim=hidden_size,
            kernel_size=(3, 3),
            num_layers=2,
            dtype=torch.cuda.FloatTensor,
            batch_first=True,
            bias = True,
            return_all_layers = False).to(device)
        self.mode=mode
        self.out_channels=kwargs['out_channels']
        self.input_dim=input_dim

    def forward(self, *x, **kwargs):
        # TODO: consider case where x has more than one channel?
        # Get first frames and add the channel dimension to them.
        b, _, h, w  = x[0].shape
        temporal_input = x[0][:, :-self.out_channels] # past frames
        # Reshape as (b, t, c, h, w)
        temporal_input = temporal_input.reshape(b, -1, self.input_dim, h, w)
        output, _ = self.temporal_encoder(temporal_input.to(self.device))
        if self.mode == 'last_output':
            conv_lstm_features = output[0][:, -1]
        elif self.mode == 'all_outputs':
            conv_lstm_features = output[0].reshape(b, -1, h, w)
        elif self.mode == 'last_output_and_last_frame':
            conv_lstm_features = torch.cat([output[0][:, -1], temporal_input[:, -1]], dim=1) 
        elif self.mode == 'interleave_frames_and_outputs':
            features = []
            for i in range(temporal_input.shape[1]):
                features += [output[0][:, i], temporal_input[:, i]]
                conv_lstm_features = torch.cat(features, dim=1) 
        else:
            raise(f"Mode not found: {self.mode}, choose between 'last_output', 'all_outputs', 'last_output_and_last_frame', 'interleave_frames_and_outputs'")

        #conv_lstm_features = encoder_hidden_states[0][0].to(self.device)
        noisy_frame = x[0][:, -self.out_channels:]
        noise_hidden_state = torch.cat([conv_lstm_features, noisy_frame], dim=1)

        return super().forward(noise_hidden_state, timestep=x[1], **kwargs).sample ## Diffusers's UNet2DConditionModel class

def get_model_dictionary(artifact_name: str, project_name: str) -> Dict[str, Any]:
    """Download a model from wandb and get its values.

    Parameters
    ----------
    at_name : str
        The name of the artifact to download.
    project_name : str
        The name of the project to download the artifact from.
    
    Returns
    -------
    list of str
        The list of files in the downloaded dataset.
    """
    def _get_model(run: Any):
        artifact_path = f'ai-industry/{project_name}/{artifact_name}'
        artifact = run.use_artifact(artifact_path, type='model')
        return artifact.download()

    if wandb.run is not None:
        run = wandb.run
        artifact_dir = _get_model(run)
    else:
        run = wandb.init(project=project_name, job_type='download_model')
        artifact_dir = _get_model(run)
        run.finish()
    # Get the file name
    [file_name] = sorted(list(Path(artifact_dir).iterdir()))
    model_dictionary = torch.load(file_name)
    if 'model_parameters' not in model_dictionary:
        # TODO: remove: Set random number of model parameters
        model_dictionary['model_parameters'] = random.randint(100, 1000)
    return model_dictionary
