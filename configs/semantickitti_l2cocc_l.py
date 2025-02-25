data_root = 'data/SemanticKitti/dataset'
stereo_depth_root = 'data/SemanticKitti/sequences_msnet3d_depth'
ann_file = 'data/SemanticKitti/labels'
camera_used = ['left']

dataset_type = 'SemanticKITTIDatasetLC'
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

empty_idx = 0

semantic_kitti_class_frequencies = [
    5.41773033e09,
    1.57835390e07,
    1.25136000e05,
    1.18809000e05,
    6.46799000e05,
    8.21951000e05,
    2.62978000e05,
    2.83696000e05,
    2.04750000e05,
    6.16887030e07,
    4.50296100e06,
    4.48836500e07,
    2.26992300e06,
    5.68402180e07,
    1.57196520e07,
    1.58442623e08,
    2.06162300e06,
    3.69705220e07,
    1.15198800e06,
    3.34146000e05,
]

# 20 classes with unlabeled
class_names = [
    'unlabeled',
    'car',
    'bicycle',
    'motorcycle',
    'truck',
    'other-vehicle',
    'person',
    'bicyclist',
    'motorcyclist',
    'road',
    'parking',
    'sidewalk',
    'other-ground',
    'building',
    'fence',
    'vegetation',
    'trunk',
    'terrain',
    'pole',
    'traffic-sign',
]
num_class = len(class_names)

# dataset config #
bda_aug_conf = dict(rot_lim=(-22.5, 22.5), scale_lim=(0.95, 1.05), flip_dx_ratio=0.5, flip_dy_ratio=0.5, flip_dz_ratio=0)

data_config = {
    'input_size': (384, 1280),
    'resize': (0., 0.),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# lidar
grid_size = [128, 128, 16]
coarse_ratio = 2

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti',
         data_config=data_config,
         load_stereo_depth=True,
         is_train=True,
         color_jitter=(0.4, 0.4, 0.4)),
    dict(
        type='LoadLidarPointsFromFiles_SemanticKitti',
        data_config=data_config,
        is_train=True,
    ),
    dict(
        type='LidarPointsPreProcess_SemanticKitti',
        data_config=data_config,
        point_cloud_range=point_cloud_range,
        occ_size=occ_size,
        coarse_ratio=coarse_ratio,
        is_train=True,
    ),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti'),
    dict(type='LoadSemKittiAnnotation',
         bda_aug_conf=bda_aug_conf,
         apply_bda=False,
         is_train=True,
         point_cloud_range=point_cloud_range),
    dict(type='CollectData',
         keys=['points', 'grid_ind', 'voxel_position_grid_coarse', 'gt_occ'],
         meta_keys=['pc_range', 'occ_size', 'gt_occ_1_2']),
]

trainset_config = dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=train_pipeline,
    split='train',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=False,
)

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti',
         data_config=data_config,
         load_stereo_depth=True,
         is_train=False,
         color_jitter=None),
    dict(
        type='LoadLidarPointsFromFiles_SemanticKitti',
        data_config=data_config,
        is_train=False,
    ),
    dict(
        type='LidarPointsPreProcess_SemanticKitti',
        data_config=data_config,
        point_cloud_range=point_cloud_range,
        occ_size=occ_size,
        coarse_ratio=coarse_ratio,
        is_train=False,
    ),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti'),
    dict(type='LoadSemKittiAnnotation',
         bda_aug_conf=bda_aug_conf,
         apply_bda=False,
         is_train=False,
         point_cloud_range=point_cloud_range),
    dict(type='CollectData',
         keys=['points', 'grid_ind', 'voxel_position_grid_coarse', 'gt_occ'],
         meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 'stereo_depth'])
]

testset_config = dict(type=dataset_type,
                      stereo_depth_root=stereo_depth_root,
                      data_root=data_root,
                      ann_file=ann_file,
                      pipeline=test_pipeline,
                      split='test',
                      camera_used=camera_used,
                      occ_size=occ_size,
                      pc_range=point_cloud_range)

data = dict(train=trainset_config, val=testset_config, test=testset_config)

train_dataloader_config = dict(batch_size=1, num_workers=4)

test_dataloader_config = dict(batch_size=1, num_workers=4)

# model params #
_dim_ = 128
voxel_out_channels = [_dim_]

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
Swin = dict(
    type='Swin',
    embed_dims=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4,
    in_channels=128,
    patch_size=4,
    strides=[1, 2, 2, 2],
    frozen_stages=-1,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.2,
    patch_norm=True,
    out_indices=[1, 2, 3],
    with_cp=False,
    convert_weights=True,
    init_cfg=dict(type='Pretrained', checkpoint='pretrain/swin_tiny_patch4_window7_224.pth'),
)

GeneralizedLSSFPN = dict(
    type='GeneralizedLSSFPN',
    in_channels=[192, 384, 768],
    out_channels=_dim_,
    start_level=0,
    num_outs=3,
    norm_cfg=dict(type='BN2d', requires_grad=True, track_running_stats=False),
    act_cfg=dict(type='LeakyReLU', inplace=True),
    upsample_cfg=dict(mode='bilinear', align_corners=False),
)

OccHead = dict(type='OccHead',
               in_channels=[sum(voxel_out_channels)],
               out_channel=num_class,
               empty_idx=0,
               num_level=1,
               with_cp=True,
               occ_size=occ_size,
               loss_weight_cfg={
                   "loss_voxel_ce_weight": 3,
                   "loss_voxel_sem_scal_weight": 1,
                   "loss_voxel_geo_scal_weight": 1,
               },
               conv_cfg=dict(type='Conv3d', bias=False),
               norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
               class_frequencies=semantic_kitti_class_frequencies)

tpv_generator = dict(
    type='TPVGenerator',
    embed_dims=_dim_,
    pooler='avg',
    global_encoder_backbone=Swin,
    global_encoder_neck=GeneralizedLSSFPN,
)

tpv_aggregator = dict(
    type='TPVAggregator',
    embed_dims=_dim_,
)

model = dict(
    type='LidarSegmentor',
    lidar_tokenizer=dict(
        type='LidarEncoder',
        grid_size=grid_size,
        in_channels=6,
        out_channels=128,
        fea_compre=None,
        base_channels=128,
        split=[8, 8, 8],
        track_running_stats=False,
    ),
    lidar_backbone=dict(type='CustomResNet3D',
                        numC_input=_dim_,
                        num_layer=[2, 2, 2],
                        num_channels=[_dim_, _dim_, _dim_],
                        stride=[1, 2, 2]),
    lidar_neck=dict(type='GeneralizedLSSFPN',
                    in_channels=[_dim_, _dim_, _dim_],
                    out_channels=_dim_,
                    start_level=0,
                    num_outs=3,
                    norm_cfg=norm_cfg,
                    conv_cfg=dict(type='Conv3d'),
                    act_cfg=dict(type='ReLU', inplace=True),
                    upsample_cfg=dict(mode='trilinear', align_corners=False)),
    tpv_generator=tpv_generator,
    tpv_aggregator=tpv_aggregator,
    pts_bbox_head=OccHead,
)
"""Training params."""
learning_rate = 3e-4
training_steps = 25000

optimizer = dict(type="AdamW", lr=learning_rate, weight_decay=0.01)

lr_scheduler = dict(type="OneCycleLR",
                    max_lr=learning_rate,
                    total_steps=training_steps + 10,
                    pct_start=0.05,
                    cycle_momentum=False,
                    anneal_strategy="cos",
                    interval="step",
                    frequency=1)
