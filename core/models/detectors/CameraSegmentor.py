import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS, HEADS
from mmdet3d.models import builder
from mmcv.runner import force_fp32
import torch.nn.functional as F
import os
import pdb


@DETECTORS.register_module()
class CameraSegmentor(BaseModule):

    def __init__(
        self,
        img_backbone,
        img_neck,
        img_view_transformer,
        depth_net=None,
        proposal_layer=None,
        VoxFormer_head=None,
        voxel_backbone=None,
        voxel_neck=None,
        tpv_generator=None,
        tpv_aggregator=None,
        pts_bbox_head=None,
        init_cfg=None,
        teacher=None,
        distill_cfg=None,
        grid_size=[128, 128, 16],
        **kwargs,
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.depth_net = builder.build_neck(depth_net)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)

        if voxel_backbone is not None:
            assert voxel_neck is not None
            self.voxel_backbone = builder.build_backbone(voxel_backbone)
            self.voxel_neck = builder.build_neck(voxel_neck)

        self.tpv_generator = builder.build_backbone(tpv_generator)
        self.tpv_aggregator = builder.build_backbone(tpv_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.grid_size = grid_size
        # init before teacher to avoid overwriting teacher weights
        self.init_cfg = init_cfg
        self.init_weights()

        if teacher:
            assert distill_cfg is not None
            self.teacher = builder.build_detector(teacher).eval()

            if os.path.exists(distill_cfg['teacher_ckpt']):
                ckpt = torch.load(distill_cfg['teacher_ckpt'], map_location='cpu')['state_dict']
                adjusted_ckpt = {key.replace('model.', ''): value for key, value in ckpt.items()}
                self.teacher.load_state_dict(adjusted_ckpt)
                print(f"Load teacher model from {distill_cfg['teacher_ckpt']}")
            self.freeze_model(self.teacher)
            self.distill_3d_feature = distill_cfg['distill_3d_feature']
            self.distill_aggregator = distill_cfg['distill_aggregator']
            self.distill_view_transformer = distill_cfg['distill_view_transformer']
            self.distill_2d_feature = distill_cfg['distill_2d_feature']
            self.distill_2d_backbone = distill_cfg['distill_2d_backbone']
            self.distill_2d_neck = distill_cfg['distill_2d_neck']
            self.distill_kl_empty = distill_cfg['distill_kl_empty']
            self.ratio_logit_kl = distill_cfg['ratio_logit_kl']
            self.ratio_feats_numeric = distill_cfg['ratio_feats_numeric']
            self.ratio_feats_relation = distill_cfg['ratio_feats_relation']
            self.ratio_aggregator_weights = distill_cfg['ratio_aggregator_weights']

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def image_encoder(self, img):
        imgs = img
        # self.save_img(img)

        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)

        # pdb.set_trace()
        # img = self.grid_mask(imgs)
        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        return x

    def extract_img_feat(self, img_inputs, img_metas):
        img = img_inputs[0]
        img_enc_feats = self.image_encoder(img)

        rots, trans, intrins, post_rots, post_trans, bda = img_inputs[1:7]
        mlp_input = self.depth_net.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)

        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]

        context, depth = self.depth_net([img_enc_feats] + geo_inputs, img_metas)
        view_trans_inputs = [
            rots[:, 0:1, ...], trans[:, 0:1, ...], intrins[:, 0:1, ...], post_rots[:, 0:1, ...], post_trans[:, 0:1, ...], bda
        ]

        lss_volume = self.img_view_transformer(context, depth, view_trans_inputs)

        query_proposal = self.proposal_layer(view_trans_inputs, img_metas)

        if query_proposal.shape[1] == 2:
            proposal = torch.argmax(query_proposal, dim=1)
        else:
            proposal = query_proposal

        if depth is not None:
            mlvl_dpt_dists = [depth.unsqueeze(1)]
        else:
            mlvl_dpt_dists = None

        x = self.VoxFormer_head([context],
                                proposal,
                                cam_params=view_trans_inputs,
                                lss_volume=lss_volume,
                                img_metas=img_metas,
                                mlvl_dpt_dists=mlvl_dpt_dists)

        if hasattr(self, 'voxel_backbone'):
            x = self.voxel_backbone(x)
            x = self.voxel_neck(x)[0]

        return x, query_proposal, depth

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']
        gt_occ_1_2 = img_metas['gt_occ_1_2']

        # img encoder
        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)

        # tpv generator
        tpv_list, feats_all = self.tpv_generator(img_voxel_feats)

        # tpv aggregator
        x_3d, aggregator_weights = self.tpv_aggregator(tpv_list, img_voxel_feats)

        feats_all['feats3d_view_transformer'] = img_voxel_feats
        feats_all['feats3d_aggregator'] = x_3d[0]

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # loss
        losses = dict()
        losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs[-4][:, 0:1, ...], depth)
        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        # distillation
        losses_distill = {}
        if hasattr(self, 'teacher'):
            with torch.no_grad():
                tpv_list_teacher, output_teacher, aggregator_weights_teacher, feats_all_teacher = self.forward_teacher(data_dict)

            if self.ratio_logit_kl > 0:
                losses_distill_logit = self.distill_loss_logits(
                    logits_teacher=output_teacher['output_voxels'],
                    logits_student=output['output_voxels'],
                    target=gt_occ,
                )
                losses_distill.update(losses_distill_logit)

            if self.ratio_feats_numeric > 0 or self.ratio_feats_relation > 0:
                losses_distill_feature, feats_student, feats_teacher, masks = self.distill_loss_feature(
                    feats_teacher=feats_all_teacher,
                    feats_student=feats_all,
                    target=gt_occ_1_2,
                )
                losses_distill.update(losses_distill_feature)

            # save_tpv(tpv_list, tpv_list_teacher)
            # save_all_feats(feats_student, feats_teacher, masks)
            # save_weights(aggregator_weights, aggregator_weights_teacher)

            if self.ratio_aggregator_weights > 0:
                losses_distill_aggregator_weights = self.distill_loss_aggregator_weights(
                    aggregator_weights_teacher,
                    aggregator_weights,
                    gt_occ_1_2,
                    self.ratio_aggregator_weights,
                )
                losses_distill.update(losses_distill_aggregator_weights)

            losses.update(losses_distill)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {'losses': losses, 'pred': pred, 'gt_occ': gt_occ}
        return train_output

    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None

        # img encoder
        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)

        # tpv transformer
        tpv_list, _ = self.tpv_generator(img_voxel_feats)

        # tpv aggregator
        x_3d, aggregator_weights = self.tpv_aggregator(tpv_list, img_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ, 'tpv_list': tpv_list, 'tpv_weights': aggregator_weights}
        return test_output

    def forward_teacher(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None

        # lidar encoder
        lidar_voxel_feats = self.teacher.extract_lidar_feat(points=points, grid_ind=grid_ind)

        # tpv transformer
        tpv_lists, feats_all = self.teacher.tpv_generator(lidar_voxel_feats)

        # tpv aggregator
        x_3d, aggregator_weights = self.teacher.tpv_aggregator(tpv_lists, lidar_voxel_feats)
        feats_all['feats3d_view_transformer'] = lidar_voxel_feats
        feats_all['feats3d_aggregator'] = x_3d[0]

        # cls head
        output = self.teacher.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        return tpv_lists, output, aggregator_weights, feats_all

    def calculate_cosine_similarity(self, x, y):
        assert x.shape == y.shape, "输入特征的形状必须相同"
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, C, H * W)
        x_flat = x.reshape(B, C, -1)
        y_flat = y.reshape(B, C, -1)

        # normalize
        x_norm = F.normalize(x_flat, p=2, dim=1)
        y_norm = F.normalize(y_flat, p=2, dim=1)

        # bmm
        cosine_similarity_flat = torch.bmm(x_norm.permute(0, 2, 1), y_norm)

        return cosine_similarity_flat

    def distill_loss_logits(self, logits_teacher, logits_student, target):
        b, c, h, w, z = logits_teacher.shape
        logits_student_softmax = F.log_softmax(logits_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        logits_teacher_softmax = F.softmax(logits_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        # teacher_pred = logits_teacher_softmax.argmax(-1)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            # valid = (target[i] != 255) & (teacher_pred[i] == target[i])
            nonezero = (target[i] != 0)
            mask = valid * nonezero if not self.distill_kl_empty else valid
            logits_student_i = logits_student_softmax[i][mask]
            logits_teacher_i = logits_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(logits_student_i.unsqueeze(0), logits_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * self.ratio_logit_kl
        return dict(loss_distill_logits=loss)

    def distill_loss_feature(self, feats_teacher, feats_student, target):
        feats_teacher_list = []
        feats_student_list = []
        mask_list = []
        ratio_list = []

        target = target.to(torch.float32)
        target[target == 255] = 0

        # feats 2d
        if self.distill_2d_feature:
            # mask
            target_xy_mean = target.mean(dim=3)
            mask_xy = target_xy_mean != 0
            size_xy = (self.grid_size[0], self.grid_size[1])
            target_yz_mean = target.mean(dim=1)
            mask_yz = target_yz_mean != 0
            size_yz = (self.grid_size[1], self.grid_size[2])
            target_zx_mean = target.mean(dim=2)
            mask_zx = target_zx_mean != 0
            size_zx = (self.grid_size[0], self.grid_size[2])
            masks = [mask_xy, mask_yz, mask_zx]
            sizes = [size_xy, size_yz, size_zx]

            # tpv
            feats_backbone_teacher = feats_teacher['tpv_backbone']
            feats_backbone_student = feats_student['tpv_backbone']
            feats_neck_teacher = feats_teacher['tpv_neck']
            feats_neck_student = feats_student['tpv_neck']

            for i in range(3):
                mask = masks[i]
                size = sizes[i]

                # tpv backbone
                if self.distill_2d_backbone:
                    for j in range(len(feats_backbone_teacher[i])):
                        feat_student = feats_backbone_student[i][j]
                        feat_teacher = feats_backbone_teacher[i][j]
                        feat_student = F.interpolate(feat_student, size=size, mode='bilinear', align_corners=False)
                        feat_teacher = F.interpolate(feat_teacher, size=size, mode='bilinear', align_corners=False)
                        mask_ = mask.unsqueeze(1).expand_as(feat_student)

                        feats_student_list.append(feat_student)
                        feats_teacher_list.append(feat_teacher)
                        mask_list.append(mask_)
                        ratio_list.append(0.1)

                # tpv neck
                if self.distill_2d_neck:
                    for j in range(len(feats_neck_teacher[i])):
                        feat_student = feats_neck_student[i][j]
                        feat_teacher = feats_neck_teacher[i][j]
                        feat_student = F.interpolate(feat_student, size=size, mode='bilinear', align_corners=False)
                        feat_teacher = F.interpolate(feat_teacher, size=size, mode='bilinear', align_corners=False)
                        mask_ = mask.unsqueeze(1).expand_as(feat_student)

                        feats_student_list.append(feat_student)
                        feats_teacher_list.append(feat_teacher)
                        mask_list.append(mask_)

                        ratio = 1 if j == 0 else 0.5
                        ratio_list.append(ratio)

        # feats 3d
        if self.distill_3d_feature:
            if self.distill_aggregator:
                feats_teacher_list.append(feats_teacher['feats3d_aggregator'])
                feats_student_list.append(feats_student['feats3d_aggregator'])
                mask = (target != 0).unsqueeze(1).expand_as(feats_student['feats3d_aggregator'])
                mask_list.append(mask)
                ratio_list.append(1.5)
            if self.distill_view_transformer:
                feats_teacher_list.append(feats_teacher['feats3d_view_transformer'])
                feats_student_list.append(feats_student['feats3d_view_transformer'])
                mask = (target != 0).unsqueeze(1).expand_as(feats_student['feats3d_view_transformer'])
                mask_list.append(mask)
                ratio_list.append(0.5)

        losses_feature = {}

        # numeric loss
        if self.ratio_feats_numeric > 0:
            loss_numeric = 0
            for i in range(len(mask_list)):

                # l1 + mse
                # mask = mask_list[i]
                # ratio = ratio_list[i]
                # feat_student = feats_student_list[i][mask]
                # feat_teacher = feats_teacher_list[i][mask]
                # loss = 3 * F.l1_loss(feat_student, feat_teacher) + F.mse_loss(feat_student, feat_teacher)
                # loss = loss * ratio

                # cos sim
                if len(feats_student_list[i].shape) == 5:
                    b, c, h, w, z = feats_student_list[i].shape
                    mask = mask_list[i].permute(0, 2, 3, 4, 1)
                    feat_student = feats_student_list[i].permute(0, 2, 3, 4, 1)
                    feat_teacher = feats_teacher_list[i].permute(0, 2, 3, 4, 1)
                else:
                    b, c, h, w = feats_student_list[i].shape
                    mask = mask_list[i].permute(0, 2, 3, 1)
                    feat_student = feats_student_list[i].permute(0, 2, 3, 1)
                    feat_teacher = feats_teacher_list[i].permute(0, 2, 3, 1)

                feat_student = feat_student[mask].view(-1, c)
                feat_teacher = feat_teacher[mask].view(-1, c)
                loss = (1 - F.cosine_similarity(feat_student, feat_teacher)).mean()
                loss_numeric += loss
            loss_numeric = loss_numeric / len(mask_list) * self.ratio_feats_numeric
            losses_feature.update(dict(loss_distill_feature_numeric=loss_numeric))

        # relation loss
        if self.ratio_feats_relation > 0:
            assert self.distill_2d_feature is True
            loss_relation = 0
            # only neck[0]
            for i in range(3):
                feat_student = feats_neck_student[i][0]
                feat_teacher = feats_neck_teacher[i][0]
                cos_sim_student = self.calculate_cosine_similarity(feat_student, feat_student)
                cos_sim_teacher = self.calculate_cosine_similarity(feat_teacher, feat_teacher)
                loss = F.l1_loss(cos_sim_student, cos_sim_teacher)
                loss_relation += loss
            loss_relation = loss_relation * self.ratio_feats_relation
            losses_feature.update(dict(loss_distill_feature_relation=loss_relation))

        return losses_feature, feats_student_list, feats_teacher_list, mask_list

    def distill_loss_aggregator_weights(self, weights_teacher, weights_student, target, ratio):
        b, c, h, w, z = weights_teacher.shape
        weights_student_softmax = F.log_softmax(weights_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        weights_teacher_softmax = F.softmax(weights_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            weights_student_i = weights_student_softmax[i][mask]
            weights_teacher_i = weights_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(weights_student_i.unsqueeze(0), weights_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_weights=loss)
