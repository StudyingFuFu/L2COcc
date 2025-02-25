# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding


@HEADS.register_module()
class VoxFormerHead_Tiny(nn.Module):

    def __init__(
        self,
        *args,
        volume_h,
        volume_w,
        volume_z,
        data_config,
        embed_dims,
        cross_transformer,
        **kwargs,
    ):
        super().__init__()
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.embed_dims = embed_dims

        self.data_config = data_config
        self.cross_transformer = build_transformer(cross_transformer)

        image_grid = self.create_grid()
        self.register_buffer('image_grid', image_grid)
        vox_coords, ref_3d = self.get_voxel_indices()
        self.register_buffer('vox_coords', vox_coords)
        self.register_buffer('ref_3d', ref_3d)

        self.mlp_prior = nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims // 2), nn.LayerNorm(self.embed_dims // 2),
                                       nn.LeakyReLU(), nn.Linear(self.embed_dims // 2, self.embed_dims))

    def get_voxel_indices(self):
        xv, yv, zv = torch.meshgrid(torch.arange(self.volume_h),
                                    torch.arange(self.volume_w),
                                    torch.arange(self.volume_z),
                                    indexing='ij')

        idx = torch.arange(self.volume_h * self.volume_w * self.volume_z)
        vox_coords = torch.cat([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)

        ref_3d = torch.cat([(xv.reshape(-1, 1) + 0.5) / self.volume_h, (yv.reshape(-1, 1) + 0.5) / self.volume_w,
                            (zv.reshape(-1, 1) + 0.5) / self.volume_z],
                           dim=-1)

        return vox_coords, ref_3d

    def create_grid(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        xs = torch.linspace(0, ogfW - 1, ogfW, dtype=torch.float).view(1, 1, ogfW).expand(1, ogfH, ogfW)
        ys = torch.linspace(0, ogfH - 1, ogfH, dtype=torch.float).view(1, ogfH, 1).expand(1, ogfH, ogfW)

        grid = torch.stack((xs, ys), 1)
        return nn.Parameter(grid, requires_grad=False)

    def forward(self, mlvl_feats, proposal, cam_params, lss_volume=None, img_metas=None, **kwargs):
        """ Forward funtion.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information
            depth: Pre-estimated depth map, (B, 1, H_d, W_d)
            cam_params: Transformation matrix, (rots, trans, intrins, post_rots, post_trans, bda)
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype, device = mlvl_feats[0].dtype, mlvl_feats[0].device

        lss_volume_flatten = lss_volume.flatten(2).squeeze(0).permute(1, 0)
        volume_queries = lss_volume_flatten

        if proposal.sum() < 2:
            proposal = torch.ones_like(proposal)

        vox_coords, ref_3d = self.vox_coords.clone(), self.ref_3d.clone()
        unmasked_idx = torch.nonzero(proposal.reshape(-1) > 0).view(-1)
        masked_idx = torch.nonzero(proposal.reshape(-1) == 0).view(-1)

        # Compute seed features of query proposals by deformable cross attention
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats,
            volume_queries,
            self.volume_h,
            self.volume_w,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=None,
            bev_pos=None,
            #  bev_pos=bev_pos_cross_attn,
            img_metas=img_metas,
            prev_bev=None,
            cam_params=cam_params,
            **kwargs)

        vox_feats = torch.empty((self.volume_h, self.volume_w, self.volume_z, self.embed_dims), device=volume_queries.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx, 3], :] = seed_feats[0]

        vox_feats_flatten[vox_coords[masked_idx, 3], :] = self.mlp_prior(lss_volume_flatten[masked_idx, :])

        vox_feats = vox_feats_flatten.reshape(self.volume_h, self.volume_w, self.volume_z, self.embed_dims)
        vox_feats = vox_feats.permute(3, 0, 1, 2).unsqueeze(0)

        return vox_feats
