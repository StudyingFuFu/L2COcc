import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import DETECTORS, HEADS
from mmdet3d.models import builder
import os
import pdb


@DETECTORS.register_module()
class LidarSegmentor(BaseModule):

    def __init__(
        self,
        lidar_tokenizer=None,
        lidar_backbone=None,
        lidar_neck=None,
        tpv_generator=None,
        tpv_aggregator=None,
        pts_bbox_head=None,
        **kwargs,
    ):

        super().__init__()

        self.lidar_tokenizer = builder.build_backbone(lidar_tokenizer)
        self.voxel_backbone = builder.build_backbone(lidar_backbone)
        self.voxel_neck = builder.build_neck(lidar_neck)
        self.tpv_generator = builder.build_backbone(tpv_generator)
        self.tpv_aggregator = builder.build_backbone(tpv_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.fp16_enabled = False

    def extract_lidar_feat(self, points, grid_ind):
        """Extract features of points."""
        x_3d = self.lidar_tokenizer(points, grid_ind)

        x_list = self.voxel_backbone(x_3d)
        output = self.voxel_neck(x_list)
        output = output[0]

        return output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

    def forward_train(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        # lidar encoder
        lidar_voxel_feats = self.extract_lidar_feat(points=points, grid_ind=grid_ind)

        # tpv generator
        tpv_lists, _ = self.tpv_generator(lidar_voxel_feats)

        # tpv aggregator
        x_3d, aggregator_weights = self.tpv_aggregator(tpv_lists, lidar_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # loss
        losses = dict()
        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {'losses': losses, 'pred': pred, 'gt_occ': gt_occ}
        return train_output

    def forward_test(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None

        # lidar encoder
        lidar_voxel_feats = self.extract_lidar_feat(points=points, grid_ind=grid_ind)

        # tpv transformer
        tpv_list, _ = self.tpv_generator(lidar_voxel_feats)

        # tpv aggregator
        x_3d, aggregator_weights = self.tpv_aggregator(tpv_list, lidar_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ, 'tpv_list': tpv_list, 'tpv_weights': aggregator_weights}
        return test_output
