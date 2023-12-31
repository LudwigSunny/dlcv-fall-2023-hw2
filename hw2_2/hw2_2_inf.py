'''
ref: https://github.com/xiaohu2015/nngen/blob/main/models/diffusion_models/ddim_mnist.ipynb
'''
import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


swish = F.silu

@torch.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        return tensor.normal_(0, std)

    else:
        bound = math.sqrt(3 * scale)

        return tensor.uniform_(-bound, bound)


def conv2d(
    in_channel,
    out_channel,
    kernel_size,
    stride=1,
    padding=0,
    bias=True,
    scale=1,
    mode="fan_avg",
):
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias
    )

    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.init.zeros_(conv.bias)

    return conv


def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)

    return lin


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)


class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Sequential):
    def __init__(self, channel):
        layers = [conv2d(channel, channel, 3, stride=2, padding=1)]

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, time_dim, use_affine_time=False, dropout=0
    ):
        super().__init__()

        self.use_affine_time = use_affine_time
        time_out_dim = out_channel
        time_scale = 1
        norm_affine = True

        if self.use_affine_time:
            time_out_dim *= 2
            time_scale = 1e-10
            norm_affine = False

        self.norm1 = nn.GroupNorm(32, in_channel)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = nn.Sequential(
            Swish(), linear(time_dim, time_out_dim, scale=time_scale)
        )

        self.norm2 = nn.GroupNorm(32, out_channel, affine=norm_affine)
        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        if in_channel != out_channel:
            self.skip = conv2d(in_channel, out_channel, 1)

        else:
            self.skip = None

    def forward(self, input, time):
        batch = input.shape[0]

        out = self.conv1(self.activation1(self.norm1(input)))

        if self.use_affine_time:
            gamma, beta = self.time(time).view(batch, -1, 1, 1).chunk(2, dim=1)
            out = (1 + gamma) * self.norm2(out) + beta

        else:
            out = out + self.time(time).view(batch, -1, 1, 1)
            out = self.norm2(out)

        out = self.conv2(self.dropout(self.activation2(out)))

        if self.skip is not None:
            input = self.skip(input)

        return out + input

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 3, 1)
        self.out = conv2d(in_channel, in_channel, 1, scale=1e-10)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class ResBlockWithAttention(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        time_dim,
        dropout,
        use_attention=False,
        attention_head=1,
        use_affine_time=False,
    ):
        super().__init__()

        self.resblocks = ResBlock(
            in_channel, out_channel, time_dim, use_affine_time, dropout
        )

        if use_attention:
            self.attention = SelfAttention(out_channel, n_head=attention_head)

        else:
            self.attention = None

    def forward(self, input, time):
        out = self.resblocks(input, time)

        if self.attention is not None:
            out = self.attention(out)

        return out


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, -1, h_unfold, w_unfold)
    )

class UNet(nn.Module):
    def __init__(
        self,
        in_channel = 3,
        channel = 128,
        attn_heads = 1,
        use_affine_time = False,
        dropout = 0,
    ):
        super().__init__()
        

        time_dim = channel * 4

        self.time = nn.Sequential(
            TimeEmbedding(channel),
            linear(channel, time_dim),
            Swish(),
            linear(time_dim, time_dim),
        )

        self.down1 = conv2d(in_channel, channel, 3, padding=1)
        self.down2 = ResBlockWithAttention(128, 128,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down3 = ResBlockWithAttention(128, 128,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down4 = Downsample(128)
        self.down5 = ResBlockWithAttention(128, 128,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down6 = ResBlockWithAttention(128, 128,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down7 = Downsample(128)
        self.down8 = ResBlockWithAttention(128, 256,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down9 = ResBlockWithAttention(256, 256,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down10 = Downsample(256)
        self.down11 = ResBlockWithAttention(256, 256,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down12 = ResBlockWithAttention(256, 256,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down13 = Downsample(256)
        self.down14 = ResBlockWithAttention(256, 512,
                time_dim,
                dropout,
                use_attention=True,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down15 = ResBlockWithAttention(512, 512,
                time_dim,
                dropout,
                use_attention=True,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down16 = Downsample(512)
        self.down17 = ResBlockWithAttention(512, 512,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.down18 = ResBlockWithAttention(512, 512,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        
        self.mid1 = ResBlockWithAttention(
                    512,
                    512,
                    time_dim,
                    dropout=dropout,
                    use_attention=True,
                    attention_head=attn_heads,
                    use_affine_time=use_affine_time,
                )
        self.mid2 = ResBlockWithAttention(
                    512,
                    512,
                    time_dim,
                    dropout=dropout,
                    use_affine_time=use_affine_time,
                )
        
        self.up1 = ResBlockWithAttention(1024, 512,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up2 = ResBlockWithAttention(1024, 512,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up3 = ResBlockWithAttention(1024, 512,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up4 = Upsample(512)
        self.up5 = ResBlockWithAttention(1024, 512,
                time_dim,
                dropout,
                use_attention=True,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up6 = ResBlockWithAttention(1024, 512,
                time_dim,
                dropout,
                use_attention=True,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up7 = ResBlockWithAttention(768, 512,
                time_dim,
                dropout,
                use_attention=True,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up8 = Upsample(512)
        self.up9 = ResBlockWithAttention(768, 256,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up10 = ResBlockWithAttention(512, 256,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up11 = ResBlockWithAttention(512, 256,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up12 = Upsample(256)
        self.up13 = ResBlockWithAttention(512, 256,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up14 = ResBlockWithAttention(512, 256,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up15 = ResBlockWithAttention(384, 256,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up16 = Upsample(256)
        self.up17 = ResBlockWithAttention(384, 128,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up18 = ResBlockWithAttention(256, 128,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up19 = ResBlockWithAttention(256, 128,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up20 = Upsample(128)
        self.up21 = ResBlockWithAttention(256, 128,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up22 = ResBlockWithAttention(256, 128,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        self.up23 = ResBlockWithAttention(256, 128,
                time_dim,
                dropout,
                use_attention=False,
                attention_head=attn_heads,
                use_affine_time=use_affine_time,
            )
        
        self.out = nn.Sequential(
            nn.GroupNorm(32, 128),
            Swish(),
            conv2d(128, 3 , 3, padding=1, scale=1e-10),
        )

    def forward(self, x, time):
        time_embed = self.time(time)

        feats = []
        
        x = self.down1(x)
        feats.append(x)
        x = self.down2(x, time_embed)
        feats.append(x)
        x = self.down3(x, time_embed)
        feats.append(x)
        x = self.down4(x)
        feats.append(x)
        x = self.down5(x, time_embed)
        feats.append(x)
        x = self.down6(x, time_embed)
        feats.append(x)
        x = self.down7(x)
        feats.append(x)
        x = self.down8(x, time_embed)
        feats.append(x)
        x = self.down9(x, time_embed)
        feats.append(x)
        x = self.down10(x)
        feats.append(x)
        x = self.down11(x, time_embed)
        feats.append(x)
        x = self.down12(x, time_embed)
        feats.append(x)
        x = self.down13(x)
        feats.append(x)
        x = self.down14(x, time_embed)
        feats.append(x)
        x = self.down15(x, time_embed)
        feats.append(x)
        x = self.down16(x)
        feats.append(x)
        x = self.down17(x, time_embed)
        feats.append(x)
        x = self.down18(x, time_embed)
        feats.append(x)
               

        x = self.mid1(x, time_embed)
        x = self.mid2(x, time_embed)
        
        x = self.up1(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up2(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up3(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up4(x)
        x = self.up5(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up6(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up7(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up8(x)
        x = self.up9(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up10(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up11(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up12(x)
        x = self.up13(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up14(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up15(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up16(x)
        x = self.up17(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up18(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up19(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up20(x)
        x = self.up21(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up22(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up23(torch.cat((x, feats.pop()), 1), time_embed)
        
        out = self.out(x)
        out = spatial_unfold(out, 1)

        return out

import torch
def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas

import os
import math
from abc import abstractmethod
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import argparse
parser = argparse.ArgumentParser(description='Description of my script.')

parser.add_argument('--noise_dir', type=str, help='path to the directory of predefined noises')
parser.add_argument('--out_dir', type=str, help='path to the directory for your 10 generated images ')
parser.add_argument('--unet_path', type=str, help='path to the pretrained model weight')

# 解析命令行参数
args = parser.parse_args()

# storing the arguments
noise_dir = args.noise_dir
out_dir = args.out_dir
unet_path = args.unet_path

class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
    ):
        self.timesteps = timesteps
        
        betas = beta_scheduler(self.timesteps)
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        #self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
        
    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs
    
    # sample new images
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
    
    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        image_size,
        batch_size=8,
        channels=3,
        ddim_timesteps=50,
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True,
        noise_img=None):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        sample_img = noise_img.to(device)
        # torch.randn((batch_size, channels, image_size, image_size), device=device)
        
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
    
            # 2. predict noise using model
            pred_noise = model(sample_img, t)
            
            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

            sample_img = x_prev
            
        return sample_img
    
    # compute train losses
    def train_losses(self, model, x_start, t):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
timesteps = 1000

# define model and diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet()
model.to(device)
model.load_state_dict(torch.load(unet_path))
model.eval()
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)

file_names = sorted(os.listdir(noise_dir))
print(file_names)
noise_list = [torch.load(os.path.join(noise_dir,x)) for x in file_names]

noise = torch.cat(noise_list, 0)
print(noise.shape)


ddim_generated_images = gaussian_diffusion.ddim_sample(model, noise.shape[2], batch_size=len(file_names), channels=3, ddim_timesteps=50, noise_img=noise)

# do min-max normalization by each channel of each image
for i in range(ddim_generated_images.shape[0]):
    for j in range(3):
      ddim_generated_images[i][j] = (ddim_generated_images[i][j]-ddim_generated_images[i][j].min()) / (ddim_generated_images[i][j].max()-ddim_generated_images[i][j].min())

for index, file_name in enumerate(file_names):
    save_image(ddim_generated_images[index], os.path.join(out_dir,f"{file_name.split('.')[0]}.png"))


'''# generate new images
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

imgs = ddim_generated_images.reshape(5, 2, 28, 28)
for n_row in range(8):
    for n_col in range(8):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")'''

'''
gt_path = '../hw2_data/face/GT'
import cv2
file_names = sorted(os.listdir(gt_path))
gt_list = list()
for x in file_names:
    img = cv2.imread(os.path.join(gt_path, x))
    img = torch.from_numpy(img)
    img = img.permute(2,0,1).to(torch.float32)
    img.unsqueeze(dim=0)
    # convert numpy.ndarray (H x W x C) in the range
    # to a torch.FloatTensor of shape (C x H x W)
    gt_list.append(img)
gt_tensor = torch.cat(gt_list, 0)

pred_path = '2_exp0'
import cv2
file_names = sorted(os.listdir(pred_path))
pred_list = list()
for x in file_names:
    img = cv2.imread(os.path.join(pred_path, x))
    img = torch.from_numpy(img)
    img = img.permute(2,0,1).to(torch.float32)
    img.unsqueeze(dim=0)
    
    # convert numpy.ndarray (H x W x C) in the range
    # to a torch.FloatTensor of shape (C x H x W)
    
    pred_list.append(img)
pred_tensor = torch.cat(pred_list, 0)

loss_red_is_mean = nn.MSELoss()
print(loss_red_is_mean(pred_tensor, gt_tensor))
'''