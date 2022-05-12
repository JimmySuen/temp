import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pos_embed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import _cfg


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Class_Attention,
                 Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, x_cls):

        u = torch.cat((x_cls, x), dim=1)

        if self.gamma_1 is None:
            x_cls = x_cls + self.drop_path(self.attn(self.norm1(u)))
            x_cls = x_cls + self.drop_path(self.mlp(self.norm2(x_cls)))
        else:
            x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
            x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))

        return x_cls


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, masks=None):
        if masks == None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # nH, Wh*Ww, Wh*Ww
        else:
            # a batch-ed relative pos indexing
            masks = torch.cat(masks, dim=0)  # (M * B) * L == N * L
            masks = masks[:, 1:] - 1  # remove cls token
            masks = masks.contiguous()
            N, L = masks.size(0), masks.size(1)

            coords_h = masks // self.window_size[1]  # N , L
            coords_w = masks % self.window_size[1]  # N , L

            # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, N, L
            coords = torch.stack([coords_h, coords_w])  # 2, N, L

            relative_coords = coords[:, :, :, None] - coords[:, :, None, :]  # 2, N, L, L
            relative_coords = relative_coords.permute(1, 2, 3, 0).contiguous()  # N, L, L, 2
            relative_coords[:, :, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(N, L + 1, L + 1), dtype=relative_coords.dtype)
            relative_position_index[:, 1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[:, 0, 0:] = self.num_relative_distance - 3
            relative_position_index[:, 0:, 0] = self.num_relative_distance - 2
            relative_position_index[:, 0, 0] = self.num_relative_distance - 1

            relative_position_bias = \
                self.relative_position_bias_table[relative_position_index.view(-1)].view(
                    N, L + 1, L + 1, -1)  # N,L+1,L+1,nH
            return relative_position_bias.permute(0, 3, 1, 2).contiguous()  # N, nH, L+1, L+1


# class Attention(nn.Module):
#    def __init__(
#            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#            proj_drop=0., window_size=None, attn_head_dim=None):
#        super().__init__()
#        self.num_heads = num_heads
#        head_dim = dim // num_heads
#        if attn_head_dim is not None:
#            head_dim = attn_head_dim
#        all_head_dim = head_dim * self.num_heads
#        self.scale = qk_scale or head_dim ** -0.5
#
#        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
#        if qkv_bias:
#            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
#            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
#        else:
#            self.q_bias = None
#            self.v_bias = None
#
#        self.attn_drop = nn.Dropout(attn_drop)
#        self.proj = nn.Linear(all_head_dim, dim)
#        self.proj_drop = nn.Dropout(proj_drop)
#
#    def forward(self, x, rel_pos_bias=None):
#        B, N, C = x.shape
#        qkv_bias = None
#        if self.q_bias is not None:
#            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
#        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
#        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
#
#        q = q * self.scale
#        attn = (q @ k.transpose(-2, -1))
#
#        if rel_pos_bias is not None:
#            attn = attn + rel_pos_bias
#
#        attn = attn.softmax(dim=-1)
#        attn = self.attn_drop(attn)
#
#        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#        x = self.proj(x)
#        x = self.proj_drop(x)
#        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# class Block(nn.Module):
#
#    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#        super().__init__()
#        self.norm1 = norm_layer(dim)
#        self.attn = Attention(
#            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#        self.norm2 = norm_layer(dim)
#        mlp_hidden_dim = int(dim * mlp_ratio)
#        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#    def forward(self, x):
#        x = x + self.drop_path(self.attn(self.norm1(x)))
#        x = x + self.drop_path(self.mlp(self.norm2(x)))
#        return x
#

class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 init_values=None,
                 act_layer=None, weight_init='', use_mean=False, use_shared_rel_pos_bias=False, depth_token_only=2,
                 out_dim=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.use_mean = use_mean
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.out_dim = out_dim or embed_dim * 3

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        if use_shared_rel_pos_bias:
            patch_shape = (img_size // patch_size, img_size // patch_size)
            self.rel_pos_bias = RelativePositionBias(window_size=patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, self.embed_dim), requires_grad=False)
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                                cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], init_values=None, norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(depth)])

        self.blocks_token = nn.ModuleList([
            Block_CA(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, init_values=init_values)
            for i in range(depth_token_only)])

        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # weight init
        nn.init.normal_(self.cls_token, std=0.02)
        # nn.init.normal_(self.pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_bias.relative_position_bias_table'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, masks=None, mask_ratio=0.):
        x = self.patch_embed(x)
        # x = torch.cat((cls_token, x), dim=1)
        if self.use_shared_rel_pos_bias:
            x = self.pos_drop(x)
        else:
            x = self.pos_drop(x + self.pos_embed)
        # for student
        if masks is not None and self.student:
            cls_token = self.cls_token.expand(x.shape[0] * len(masks), -1,
                                              -1)  # stole cls_tokens impl from Phil Wang, thanks
            x_list = []
            for mask in masks:
                mask = mask.view(x.shape[0], -1, 1).repeat(1, 1, x.shape[2])
                x_list.append(torch.gather(x, 1, mask))
            x_multi = torch.cat(x_list, dim=0)
        else:
            x_multi = x
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x_multi = self.blocks(x_multi)

        for i, blk in enumerate(self.blocks_token):
            cls_token = blk(x_multi, cls_token)

        x_multi = torch.cat((cls_token, x_multi), dim=1)

        if self.use_mean:
            vs = int(196 * (1 - mask_ratio))
            x = x_multi[:, 1:vs + 1].mean(1)
            return self.norm(x)
        else:
            x = self.norm(x_multi)
            return self.pre_logits(x[:, 0])

    def forward(self, x, masks=None, mask_ratio=0.):
        x = self.forward_features(x, masks, mask_ratio)
        out = self.head(x)
        return [x, out]


def vit_small(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_base(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_large(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
