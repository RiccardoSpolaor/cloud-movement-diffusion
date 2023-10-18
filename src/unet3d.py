'''# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_3d_blocks import UNetMidBlock3DCrossAttn, get_down_block, get_up_block


@dataclass
class UNet3DOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


class UNet3DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample. Dimensions must be a multiple of `2 ** (len(block_out_channels) -
            1)`.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`):
            Tuple of downsample block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            Block type for middle of UNet, it can be either `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(224, 448, 672, 896)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        downsample_type (`str`, *optional*, defaults to `conv`):
            The downsample type for downsampling layers. Choose between "conv" and "resnet"
        upsample_type (`str`, *optional*, defaults to `conv`):
            The upsample type for upsampling layers. Choose between "conv" and "resnet"
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for normalization.
        attn_norm_num_groups (`int`, *optional*, defaults to `None`):
            If set to an integer, a group norm layer will be created in the mid block's [`Attention`] layer with the
            given number of groups. If left as `None`, the group norm layer will only be created if
            `resnet_time_scale_shift` is set to `default`, and if created will have `norm_num_groups` groups.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock2D`]). Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, or `"identity"`.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim` when performing class
            conditioning with `class_embed_type` equal to `None`.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str] = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1024,
        attention_head_dim: Union[int, Tuple[int]] = 64,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = 2,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        attn_norm_num_groups: Optional[int] = None,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=False,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))

        output_channel = reversed_block_out_channels[0]

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False


            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=False,
                resolution_idx=i,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet3DOutput(sample=sample)''';


import math
import functools
from operator import mul

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def mul_reduce(tup):
    return functools.reduce(mul, tup)

def divisible_by(numer, denom):
    return (numer % denom) == 0

mlist = nn.ModuleList

# for time conditioning

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.theta = theta
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert dtype == torch.float, 'input to sinusoidal pos emb must be a float type'

        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device, dtype = dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1).type(dtype)

# layernorm 3d

class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = eps).rsqrt() * self.g

# feedforward

def shift_token(t):
    t, t_shift = t.chunk(2, dim = 1)
    t_shift = F.pad(t_shift, (0, 0, 0, 0, 1, -1), value = 0.)
    return torch.cat((t, t_shift), dim = 1)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = 1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()

        inner_dim = int(dim * mult * 2 / 3)
        self.proj_in = nn.Sequential(
            nn.Conv3d(dim, inner_dim * 2, 1, bias = False),
            GEGLU()
        )

        self.proj_out = nn.Sequential(
            ChanLayerNorm(inner_dim),
            nn.Conv3d(inner_dim, dim, 1, bias = False)
        )

    def forward(self, x, enable_time = True):
        x = self.proj_in(x)

        if enable_time:
            x = shift_token(x)

        return self.proj_out(x)

# best relative positional encoding

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 1,
        layers = 2
    ):
        super().__init__()
        self.num_dims = num_dims

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, *dimensions):
        device = self.device

        shape = torch.tensor(dimensions, device = device)
        rel_pos_shape = 2 * shape - 1

        # calculate strides

        strides = torch.flip(rel_pos_shape, (0,)).cumprod(dim = -1)
        strides = torch.flip(F.pad(strides, (1, -1), value = 1), (0,))

        # get all positions and calculate all the relative distances

        positions = [torch.arange(d, device = device) for d in dimensions]
        grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'), dim = -1)
        grid = rearrange(grid, '... c -> (...) c')
        rel_dist = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

        # get all relative positions across all dimensions

        rel_positions = [torch.arange(-d + 1, d, device = device) for d in dimensions]
        rel_pos_grid = torch.stack(torch.meshgrid(*rel_positions, indexing = 'ij'), dim = -1)
        rel_pos_grid = rearrange(rel_pos_grid, '... c -> (...) c')

        # mlp input

        bias = rel_pos_grid.float()

        for layer in self.net:
            bias = layer(bias)

        # convert relative distances to indices of the bias

        rel_dist += (shape - 1)  # make sure all positive
        rel_dist *= strides
        rel_dist_indices = rel_dist.sum(dim = -1)

        # now select the bias for each unique relative position combination

        bias = bias[rel_dist_indices]
        return rearrange(bias, 'i j h -> h i j')

# helper classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        nn.init.zeros_(self.to_out.weight.data) # identity with skip connection

    def forward(
        self,
        x,
        rel_pos_bias = None
    ):

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main contribution - pseudo 3d conv

class PseudoConv3d(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        *,
        temporal_kernel_size = None,
        **kwargs
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)

        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size = kernel_size, padding = kernel_size // 2)
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size = temporal_kernel_size, padding = temporal_kernel_size // 2) if kernel_size > 1 else None

        if exists(self.temporal_conv):
            nn.init.dirac_(self.temporal_conv.weight.data) # initialized to be identity
            nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(
        self,
        x,
        enable_time = True
    ):
        b, c, *_, h, w = x.shape

        is_video = x.ndim == 5
        enable_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) c h w')

        x = self.spatial_conv(x)

        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', b = b)

        if not enable_time or not exists(self.temporal_conv):
            return x

        x = rearrange(x, 'b c f h w -> (b h w) c f')

        x = self.temporal_conv(x)

        x = rearrange(x, '(b h w) c f -> b c f h w', h = h, w = w)

        return x

# factorized spatial temporal attention from Ho et al.

class SpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        add_feed_forward = True,
        ff_mult = 4
    ):
        super().__init__()
        self.spatial_attn = Attention(dim = dim, dim_head = dim_head, heads = heads)
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim // 2, heads = heads, num_dims = 2)

        self.temporal_attn = Attention(dim = dim, dim_head = dim_head, heads = heads)
        self.temporal_rel_pos_bias = ContinuousPositionBias(dim = dim // 2, heads = heads, num_dims = 1)

        self.has_feed_forward = add_feed_forward
        if not add_feed_forward:
            return

        self.ff = FeedForward(dim = dim, mult = ff_mult)

    def forward(
        self,
        x,
        enable_time = True
    ):
        b, c, *_, h, w = x.shape
        is_video = x.ndim == 5
        enable_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) (h w) c')
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')

        space_rel_pos_bias = self.spatial_rel_pos_bias(h, w)

        x = self.spatial_attn(x, rel_pos_bias = space_rel_pos_bias) + x

        if is_video:
            x = rearrange(x, '(b f) (h w) c -> b c f h w', b = b, h = h, w = w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)

        if enable_time:

            x = rearrange(x, 'b c f h w -> (b h w) f c')

            time_rel_pos_bias = self.temporal_rel_pos_bias(x.shape[1])

            x = self.temporal_attn(x, rel_pos_bias = time_rel_pos_bias) + x

            x = rearrange(x, '(b h w) f c -> b c f h w', w = w, h = h)

        if self.has_feed_forward:
            x = self.ff(x, enable_time = enable_time) + x

        return x

# resnet block

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size = 3,
        temporal_kernel_size = None,
        groups = 8
    ):
        super().__init__()
        self.project = PseudoConv3d(dim, dim_out, 3)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x,
        scale_shift = None,
        enable_time = False
    ):
        x = self.project(x, enable_time = enable_time)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        timestep_cond_dim = None,
        groups = 8
    ):
        super().__init__()

        self.timestep_mlp = None

        if exists(timestep_cond_dim):
            self.timestep_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_cond_dim, dim_out * 2)
            )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = PseudoConv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x,
        timestep_emb = None,
        enable_time = True
    ):
        assert not (exists(timestep_emb) ^ exists(self.timestep_mlp))

        scale_shift = None

        if exists(self.timestep_mlp) and exists(timestep_emb):
            time_emb = self.timestep_mlp(timestep_emb)
            to_einsum_eq = 'b c 1 1 1' if x.ndim == 5 else 'b c 1 1'
            time_emb = rearrange(time_emb, f'b c -> {to_einsum_eq}')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift, enable_time = enable_time)

        h = self.block2(h, enable_time = enable_time)

        return h + self.res_conv(x)

# pixelshuffle upsamples and downsamples
# where time dimension can be configured

class Downsample(nn.Module):
    def __init__(
        self,
        dim,
        downsample_space = True,
        downsample_time = False,
        nonlin = False
    ):
        super().__init__()
        assert downsample_space or downsample_time

        self.down_space = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(dim * 4, dim, 1, bias = False),
            nn.SiLU() if nonlin else nn.Identity()
        ) if downsample_space else None

        self.down_time = nn.Sequential(
            Rearrange('b c (f p) h w -> b (c p) f h w', p = 2),
            nn.Conv3d(dim * 2, dim, 1, bias = False),
            nn.SiLU() if nonlin else nn.Identity()
        ) if downsample_time else None

    def forward(
        self,
        x,
        enable_time = True
    ):
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, 'b c f h w -> b f c h w')
            x, ps = pack([x], '* c h w')

        if exists(self.down_space):
            x = self.down_space(x)

        if is_video:
            x, = unpack(x, ps, '* c h w')
            x = rearrange(x, 'b f c h w -> b c f h w')

        if not is_video or not exists(self.down_time) or not enable_time:
            return x

        x = self.down_time(x)

        return x

class Upsample(nn.Module):
    def __init__(
        self,
        dim,
        upsample_space = True,
        upsample_time = False,
        nonlin = False
    ):
        super().__init__()
        assert upsample_space or upsample_time

        self.up_space = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.SiLU() if nonlin else nn.Identity(),
            Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1 = 2, p2 = 2)
        ) if upsample_space else None

        self.up_time = nn.Sequential(
            nn.Conv3d(dim, dim * 2, 1),
            nn.SiLU() if nonlin else nn.Identity(),
            Rearrange('b (c p) f h w -> b c (f p) h w', p = 2)
        ) if upsample_time else None

        self.init_()

    def init_(self):
        if exists(self.up_space):
            self.init_conv_(self.up_space[0], 4)

        if exists(self.up_time):
            self.init_conv_(self.up_time[0], 2)

    def init_conv_(self, conv, factor):
        o, *remain_dims = conv.weight.shape
        conv_weight = torch.empty(o // factor, *remain_dims)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = factor)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(
        self,
        x,
        enable_time = True
    ):
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, 'b c f h w -> b f c h w')
            x, ps = pack([x], '* c h w')

        if exists(self.up_space):
            x = self.up_space(x)

        if is_video:
            x, = unpack(x, ps, '* c h w')
            x = rearrange(x, 'b f c h w -> b c f h w')

        if not is_video or not exists(self.up_time) or not enable_time:
            return x

        x = self.up_time(x)

        return x

# space time factorized 3d unet

class SpaceTimeUnet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        channels = 3,
        dim_mult = (1, 2, 4, 8),
        self_attns = (False, False, False, True),
        temporal_compression = (False, True, True, True),
        resnet_block_depths = (2, 2, 2, 2),
        attn_dim_head = 64,
        attn_heads = 8,
        condition_on_timestep = True
    ):
        super().__init__()
        assert len(dim_mult) == len(self_attns) == len(temporal_compression) == len(resnet_block_depths)
        num_layers = len(dim_mult)

        dims = [dim, *map(lambda mult: mult * dim, dim_mult)]
        dim_in_out = zip(dims[:-1], dims[1:])

        # determine the valid multiples of the image size and frames of the video

        self.frame_multiple = 2 ** sum(tuple(map(int, temporal_compression)))
        self.image_size_multiple = 2 ** num_layers

        # timestep conditioning for DDPM, not to be confused with the time dimension of the video

        self.to_timestep_cond = None
        timestep_cond_dim = (dim * 4) if condition_on_timestep else None

        if condition_on_timestep:
            self.to_timestep_cond = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, timestep_cond_dim),
                nn.SiLU()
            )

        # layers

        self.downs = mlist([])
        self.ups = mlist([])

        attn_kwargs = dict(
            dim_head = attn_dim_head,
            heads = attn_heads
        )

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, timestep_cond_dim = timestep_cond_dim)
        self.mid_attn = SpatioTemporalAttention(dim = mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, timestep_cond_dim = timestep_cond_dim)

        for _, self_attend, (dim_in, dim_out), compress_time, resnet_block_depth in zip(range(num_layers), self_attns, dim_in_out, temporal_compression, resnet_block_depths):
            assert resnet_block_depth >= 1

            self.downs.append(mlist([
                ResnetBlock(dim_in, dim_out, timestep_cond_dim = timestep_cond_dim),
                mlist([ResnetBlock(dim_out, dim_out) for _ in range(resnet_block_depth)]),
                SpatioTemporalAttention(dim = dim_out, **attn_kwargs) if self_attend else None,
                Downsample(dim_out, downsample_time = compress_time)
            ]))

            self.ups.append(mlist([
                ResnetBlock(dim_out * 2, dim_in, timestep_cond_dim = timestep_cond_dim),
                mlist([ResnetBlock(dim_in + (dim_out if ind == 0 else 0), dim_in) for ind in range(resnet_block_depth)]),
                SpatioTemporalAttention(dim = dim_in, **attn_kwargs) if self_attend else None,
                Upsample(dim_out, upsample_time = compress_time) #  = compress_time) fix it better
            ]))

        self.skip_scale = 2 ** -0.5 # paper shows faster convergence

        self.conv_in = PseudoConv3d(dim = channels, dim_out = dim, kernel_size = 7, temporal_kernel_size = 3)
        self.conv_out = PseudoConv3d(dim = dim, dim_out = channels, kernel_size = 3, temporal_kernel_size = 3)
        self.time_conv = nn.Conv3d(4, 1, kernel_size=1, bias=False)

    def forward(
        self,
        x,
        timestep = None,
        enable_time = True
    ):

        # some asserts

        assert not (exists(self.to_timestep_cond) ^ exists(timestep))
        is_video = x.ndim == 5

        if enable_time and is_video:
            frames = x.shape[2]
            assert divisible_by(frames, self.frame_multiple), f'number of frames on the video ({frames}) must be divisible by the frame multiple ({self.frame_multiple})'

        height, width = x.shape[-2:]
        assert divisible_by(height, self.image_size_multiple) and divisible_by(width, self.image_size_multiple), f'height and width of the image or video must be a multiple of {self.image_size_multiple}'

        # main logic

        t = self.to_timestep_cond(rearrange(timestep, '... -> (...)')) if exists(timestep) else None

        x = self.conv_in(x, enable_time = enable_time)

        hiddens = []

        for init_block, blocks, maybe_attention, downsample in self.downs:
            x = init_block(x, t, enable_time = enable_time)
            hiddens.append(x.clone())
            
            for block in blocks:
                x = block(x, enable_time = enable_time)
            if exists(maybe_attention):
                x = maybe_attention(x, enable_time = enable_time)
            hiddens.append(x.clone())

            x = downsample(x, enable_time = enable_time)
            
        x = self.mid_block1(x, t, enable_time = enable_time)
        x = self.mid_attn(x, enable_time = enable_time)
        x = self.mid_block2(x, t, enable_time = enable_time)

        for init_block, blocks, maybe_attention, upsample in reversed(self.ups):
            x = upsample(x, enable_time = enable_time)

            x = torch.cat((hiddens.pop() * self.skip_scale, x), dim = 1)

            x = init_block(x, t, enable_time = enable_time)

            x = torch.cat((hiddens.pop() * self.skip_scale, x), dim = 1)

            for block in blocks:
                x = block(x, enable_time = enable_time)

            if exists(maybe_attention):
                x = maybe_attention(x, enable_time = enable_time)
        x = self.conv_out(x, enable_time = enable_time)
        x = x.permute(0,2,1,3,4)
        x = self.time_conv(x)
        x = x.permute(0,2,1,3,4)
        return x