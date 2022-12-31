""" Relative Position Vision Transformer (ViT) in PyTorch

Hacked together by / Copyright 2022, Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply
from .layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, to_2tuple
from .registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_relpos_base_patch32_plus_rpn_256': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_replos_base_patch32_plus_rpn_256-sw-dd486f51.pth',
        input_size=(3, 256, 256)),
    'vit_relpos_base_patch16_plus_240': _cfg(url='', input_size=(3, 240, 240)),
    'vit_relpos_base_patch16_rpn_224': _cfg(url=''),
    'vit_relpos_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_relpos_base_patch16_224-sw-49049aed.pth'),
}


def gen_relative_position_index(win_size: Tuple[int, int], class_token: int = 0) -> torch.Tensor:
    # cut and paste w/ modifications from swin / beit codebase
    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = win_size[0] * win_size[1]
    coords = torch.stack(torch.meshgrid([torch.arange(win_size[0]), torch.arange(win_size[1])])).flatten(1)  # 2, Wh, Ww
    relative_coords = coords[:, :, None] - coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_size[1] - 1
    relative_coords[:, :, 0] *= 2 * win_size[1] - 1
    if class_token:
        num_relative_distance = (2 * win_size[0] - 1) * (2 * win_size[1] - 1) + 3
        relative_position_index = torch.zeros(size=(window_area + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = num_relative_distance - 3
        relative_position_index[0:, 0] = num_relative_distance - 2
        relative_position_index[0, 0] = num_relative_distance - 1
    else:
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index


def gen_relative_position_log(win_size: Tuple[int, int]) -> torch.Tensor:
    """Method initializes the pair-wise relative positions to compute the positional biases."""
    coordinates = torch.stack(torch.meshgrid([torch.arange(win_size[0]), torch.arange(win_size[1])])).flatten(1)
    relative_coords = coordinates[:, :, None] - coordinates[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).float()
    relative_coordinates_log = torch.sign(relative_coords) * torch.log(1.0 + relative_coords.abs())
    return relative_coordinates_log


class RelPosMlp(nn.Module):
    # based on timm swin-v2 impl
    def __init__(self, window_size, num_heads=8, hidden_dim=32, class_token=False):
        super().__init__()
        self.window_size = window_size
        self.window_area = self.window_size[0] * self.window_size[1]
        self.class_token = 1 if class_token else 0
        self.num_heads = num_heads

        self.mlp = Mlp(
            2,  # x, y
            hidden_features=min(128, hidden_dim * num_heads),
            out_features=num_heads,
            act_layer=nn.ReLU,
            drop=(0.125, 0.)
        )

        self.register_buffer(
            'rel_coords_log',
            gen_relative_position_log(window_size),
            persistent=False
        )

    def get_bias(self) -> torch.Tensor:
        relative_position_bias = self.mlp(self.rel_coords_log).permute(2, 0, 1).unsqueeze(0)
        if self.class_token:
            relative_position_bias = F.pad(relative_position_bias, [self.class_token, 0, self.class_token, 0])
        return relative_position_bias

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor] = None):
        return attn + self.get_bias()


class RelPosBias(nn.Module):

    def __init__(self, window_size, num_heads, class_token=False):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.class_token = 1 if class_token else 0
        self.bias_shape = (self.window_area + self.class_token,) * 2 + (num_heads,)

        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3 * self.class_token
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))
        self.register_buffer(
            "relative_position_index",
            gen_relative_position_index(self.window_size, class_token=self.class_token),
            persistent=False,
        )

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def get_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.bias_shape)  # win_h * win_w, win_h * win_w, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor] = None):
        return attn + self.get_bias()


class RelPosAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, rel_pos_cls=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rel_pos = rel_pos_cls(num_heads=num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rel_pos is not None:
            attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
        elif shared_rel_pos is not None:
            attn = attn + shared_rel_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RelPosBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, rel_pos_cls=None, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RelPosAttention(
            dim, num_heads, qkv_bias=qkv_bias, rel_pos_cls=rel_pos_cls, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), shared_rel_pos=shared_rel_pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostRelPosBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, rel_pos_cls=None, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values

        self.attn = RelPosAttention(
            dim, num_heads, qkv_bias=qkv_bias, rel_pos_cls=rel_pos_cls, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.norm1(self.attn(x, shared_rel_pos=shared_rel_pos)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class VisionTransformerRelPos(nn.Module):
    """ Vision Transformer w/ Relative Position Bias

    Differing from classic vit, this impl
      * uses relative position index (swin v1 / beit) or relative log coord + mlp (swin v2) pos embed
      * defaults to no class token (can be enabled)
      * defaults to global avg pool for head (can be changed)
      * layer-scale (residual branch gain) enabled
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='avg',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=1e-5,
            class_token=False, rel_pos_type='mlp', shared_rel_pos=False, fc_norm=False,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='skip',
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=RelPosBlock):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'avg')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token (default: False)
            rel_pos_ty pe (str): type of relative position
            shared_rel_pos (bool): share relative pos across all blocks
            fc_norm (bool): use pre classifier norm instead of pre-pool
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        feat_size = self.patch_embed.grid_size

        rel_pos_cls = RelPosMlp if rel_pos_type == 'mlp' else RelPosBias
        rel_pos_cls = partial(rel_pos_cls, window_size=feat_size, class_token=class_token)
        self.shared_rel_pos = None
        if shared_rel_pos:
            self.shared_rel_pos = rel_pos_cls(num_heads=num_heads)
            # NOTE shared rel pos currently mutually exclusive w/ per-block, but could support both...
            rel_pos_cls = None

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim)) if self.num_tokens else None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, rel_pos_cls=rel_pos_cls,
                init_values=init_values, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'moco', '')
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        # FIXME weight init scheme using PyTorch defaults curently
        #named_apply(get_init_weights_vit(mode, head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        shared_rel_pos = self.shared_rel_pos.get_bias() if self.shared_rel_pos is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, shared_rel_pos=shared_rel_pos)
            else:
                x = blk(x, shared_rel_pos=shared_rel_pos)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_vision_transformer_relpos(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(VisionTransformerRelPos, variant, pretrained, **kwargs)
    return model


@register_model
def vit_relpos_base_patch32_plus_rpn_256(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32+) w/ relative log-coord position and residual post-norm, no class token
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=896, depth=12, num_heads=14, block_fn=ResPostRelPosBlock, **kwargs)
    model = _create_vision_transformer_relpos(
        'vit_relpos_base_patch32_plus_rpn_256', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_relpos_base_patch16_plus_240(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16+) w/ relative log-coord position, no class token
    """
    model_kwargs = dict(patch_size=16, embed_dim=896, depth=12, num_heads=14, **kwargs)
    model = _create_vision_transformer_relpos('vit_relpos_base_patch16_plus_240', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_relpos_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ relative log-coord position, no class token
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, fc_norm=True, **kwargs)
    model = _create_vision_transformer_relpos('vit_relpos_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_relpos_base_patch16_rpn_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ relative log-coord position and residual post-norm, no class token
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, block_fn=ResPostRelPosBlock, **kwargs)
    model = _create_vision_transformer_relpos('vit_relpos_base_patch16_rpn_224', pretrained=pretrained, **model_kwargs)
    return model
