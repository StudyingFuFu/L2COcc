import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from sklearn.decomposition import PCA

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 0
BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0

VOXEL_SIZE = [0.2, 0.2, 0.2]
POINT_CLOUD_RANGE = [0, -25.6, -2, 51.2, 25.6, 4.4]
SPTIAL_SHAPE = [256, 256, 32]
TGT_VOXEL_SIZE = [0.2, 0.2, 0.2]
TGT_POINT_CLOUD_RANGE = [0, -25.6, -2, 51.2, 25.6, 4.4]

colormap_to_colors = np.array(
    [
        [0, 0, 0, 255],  # 0 undefined
        [100, 150, 245, 255],  # 1 barrier  orange
        [100, 230, 245, 255],  # 2 bicycle  Blue
        [30, 60, 150, 255],  # 3 bus  Darkslategrey
        [80, 30, 180, 255],  # 4 car  Crimson
        [0, 0, 255, 255],  # 5 cons. Veh  Orangered
        [255, 30, 30, 255],  # 6 motorcycle  Darkorange
        [255, 40, 200, 255],  # 7 pedestrian  Darksalmon
        [150, 30, 90, 255],  # 8 traffic cone  Red
        [255, 0, 255, 255],  # 9 trailer  Slategrey
        [255, 150, 255, 255],  # 10 truck Burlywood
        [75, 0, 75, 255],  # 11 drive sur  Green
        [175, 0, 75, 255],  # 12 other lat  nuTonomy green
        [255, 200, 0, 255],  # 13 sidewalk
        [255, 120, 50, 255],  # 14 terrain
        [0, 175, 0, 255],  # 15 manmade
        [135, 60, 0, 255],  # 16 vegeyation
        [150, 240, 80, 255],  # 16 vegeyation
        [255, 240, 150, 255],  # 16 vegeyation
        [255, 0, 0, 255],  # 16 vegeyation
    ],
    dtype=np.float32)


def filter_points(points, range):
    mask = ((points[:, 0] >= range[0]) & (points[:, 0] <= range[3]) & (points[:, 1] >= range[1]) & (points[:, 1] <= range[4]) &
            (points[:, 2] >= range[2]) & (points[:, 2] <= range[5]))
    return points[mask]


def voxel2points(voxel, occ_show, voxelSize):
    """
    Args:
        voxel: (Dx, Dy, Dz)
        occ_show: (Dx, Dy, Dz)
        voxelSize: (dx, dy, dz)

    Returns:
        points: (N, 3) 3: (x, y, z)
        voxel: (N, ) cls_id
        occIdx: (x_idx, y_idx, z_idx)
    """
    occIdx = torch.where(occ_show)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + POINT_CLOUD_RANGE[0], \
                        occIdx[1][:, None] * voxelSize[1] + POINT_CLOUD_RANGE[1], \
                        occIdx[2][:, None] * voxelSize[2] + POINT_CLOUD_RANGE[2]),
                       dim=1)      # (N, 3) 3: (x, y, z)
    return points, voxel[occIdx], occIdx


def voxel_profile(voxel, voxel_size):
    """
    Args:
        voxel: (N, 3)  3:(x, y, z)
        voxel_size: (vx, vy, vz)

    Returns:
        box: (N, 7) (x, y, z - dz/2, vx, vy, vz, 0)
    """
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)  # (x, y, z - dz/2)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None], torch.tensor(voxel_size[1]).repeat(
        centers.shape[0])[:, None], torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]),
                    dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def my_compute_box_3d(center, size, heading_angle):
    """
    Args:
        center: (N, 3)  3: (x, y, z - dz/2)
        size: (N, 3)    3: (vx, vy, vz)
        heading_angle: (N, 1)
    Returns:
        corners_3d: (N, 8, 3)
    """
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def show_point_cloud(points: np.ndarray,
                     colors=True,
                     points_colors=None,
                     bbox3d=None,
                     voxelize=False,
                     bbox_corners=None,
                     linesets=None,
                     vis=None,
                     offset=[0, 0, 0],
                     large_voxel=True,
                     voxel_size=0.4):
    """
    :param points: (N, 3)  3:(x, y, z)
    :param colors: false 不显示点云颜色
    :param points_colors: (N, 4）
    :param bbox3d: voxel grid (N, 7) 7: (center, wlh, yaw=0)
    :param voxelize: false 不显示voxel边界
    :param bbox_corners: (N, 8, 3)  voxel grid 角点坐标, 用于绘制voxel grid 边界.
    :param linesets: 用于绘制voxel grid 边界.
    :return:
    """
    # breakpoint()
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points + offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3)) + offset)
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

    vis.add_geometry(mesh_frame)

    # ego_pcd = o3d.geometry.PointCloud()
    # ego_points = generate_the_ego_car()
    # ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
    # vis.add_geometry(ego_pcd)

    return vis


def show_occ(occ_state, occ_show, voxel_size, vis=None, offset=[0, 0, 0]):
    """
    Args:
        occ_state: (Dx, Dy, Dz), cls_id
        occ_show: (Dx, Dy, Dz), bool
        voxel_size: [0.4, 0.4, 0.4]
        vis: Visualizer
        offset:

    Returns:

    """
    colors = colormap_to_colors / 255
    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    # pcd: (N, 3)  3: (x, y, z)
    # labels: (N, )  cls_id
    _labels = labels % len(colors)
    pcds_colors = colors[_labels]  # (N, 4)

    bboxes = voxel_profile(pcd, voxel_size)  # (N, 7)   7: (x, y, z - dz/2, dx, dy, dz, 0)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])  # (N, 8, 3)
    # breakpoint()
    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6],
                          [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)  # (N, 12, 2)
    # (N, 12, 2) + (N, 1, 1) --> (N, 12, 2)   此时edges中记录的是bboxes_corners的整体id: (0, N*8).
    edges = edges + bases_[:, None, None]
    # breakpoint()

    vis = show_point_cloud(points=pcd.numpy(),
                           colors=True,
                           points_colors=pcds_colors,
                           voxelize=False,
                           bbox3d=bboxes.numpy(),
                           bbox_corners=bboxes_corners.numpy(),
                           linesets=edges.numpy(),
                           vis=vis,
                           offset=offset,
                           large_voxel=True,
                           voxel_size=0.2)
    return vis


def save_feature_map_as_image(feature_map, output_dir, name='map', method='pca', n_components=3):
    """
    Save feature map as an image.

    Args:
        feature_map (torch.Tensor): Feature map tensor of shape [B, N, C, H, W].
        output_dir (str): Directory to save the images.
        method (str): Method for visualization ('single_channel', 'average', 'pca', 'max_activation').
        n_components (int): Number of components for PCA (default is 3 for RGB).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Normalize feature map to [0, 1]
    def normalize(tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        return (tensor - tensor_min) / (tensor_max - tensor_min)

    B, N, C, H, W = feature_map.shape
    for b in range(B):
        for n in range(N):
            feature = feature_map[b, n]

            if method == 'single_channel':
                for c in range(C):
                    single_channel = feature[c:c + 1]
                    single_channel = normalize(single_channel)
                    single_channel_np = single_channel.cpu().numpy().transpose(1, 2, 0)
                    single_channel_np = np.flipud(single_channel_np)
                    single_channel_np = np.fliplr(single_channel_np)
                    img_path = os.path.join(output_dir, f'feature_{name}_b{b}_n{n}_c{c}.png')
                    plt.imsave(img_path, single_channel_np.squeeze(), cmap='gray')
                    print(f'Saved: {img_path}')

            elif method == 'average':
                avg_feature = torch.mean(feature, dim=0, keepdim=True)
                avg_feature = normalize(avg_feature)
                avg_feature_np = avg_feature.cpu().numpy().transpose(1, 2, 0)
                avg_feature_np = np.flipud(avg_feature_np)
                avg_feature_np = np.fliplr(avg_feature_np)
                img_path = os.path.join(output_dir, f'feature_{name}_b{b}_n{n}_avg.png')
                plt.imsave(img_path, avg_feature_np.squeeze(), cmap='gray')
                print(f'Saved: {img_path}')

            elif method == 'pca':
                feature_flattened = feature.reshape(C, H * W).cpu().numpy().T
                pca = PCA(n_components=n_components)
                pca_feature = pca.fit_transform(feature_flattened)
                pca_feature = pca_feature.T.reshape(n_components, H, W)
                pca_feature = normalize(torch.tensor(pca_feature))
                pca_feature_np = pca_feature.cpu().numpy().transpose(1, 2, 0)
                pca_feature_np = np.flipud(pca_feature_np)
                pca_feature_np = np.fliplr(pca_feature_np)
                img_path = os.path.join(output_dir, f'feature_{name}_b{b}_n{n}_pca.png')
                plt.imsave(img_path, pca_feature_np)
                print(f'Saved: {img_path}')

            elif method == 'max_activation':
                max_channel = torch.argmax(feature.mean(dim=(1, 2)))
                max_feature = feature[max_channel:max_channel + 1]
                max_feature = normalize(max_feature)
                max_feature_np = max_feature.cpu().numpy().transpose(1, 2, 0)
                max_feature_np = np.flipud(max_feature_np)
                max_feature_np = np.fliplr(max_feature_np)
                img_path = os.path.join(output_dir, f'feature_{name}_b{b}_n{n}_max.png')
                plt.imsave(img_path, max_feature_np.squeeze(), cmap='gray')
                print(f'Saved: {img_path}')


def save_tpv(tpv_list, save_folder, frame_id):

    # format to b,n,c,h,w
    feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
    save_feature_map_as_image(feat_xy.detach(), os.path.join(save_folder, 'xy'), frame_id, method='pca')

    feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
    save_feature_map_as_image(feat_yz.detach(), os.path.join(save_folder, 'yz'), frame_id, method='pca')

    feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
    save_feature_map_as_image(feat_zx.detach(), os.path.join(save_folder, 'zx'), frame_id, method='pca')

    return


def save_weights(tpv_weights, save_folder, frame_id):
    os.makedirs(save_folder, exist_ok=True)
    save_folder_xy = os.path.join(save_folder, 'xy')
    os.makedirs(save_folder_xy, exist_ok=True)
    save_folder_yz = os.path.join(save_folder, 'yz')
    os.makedirs(save_folder_yz, exist_ok=True)
    save_folder_zx = os.path.join(save_folder, 'zx')
    os.makedirs(save_folder_zx, exist_ok=True)

    tpv_weights = F.softmax(tpv_weights, dim=1)

    def get_weights_23d(weights):
        weights_23d = torch.cat(
            [
                weights[:, :, 3:],
                weights[:, :, :3].sum(dim=2, keepdim=True),
                torch.zeros_like(weights[:, :, 3:]),
            ],
            dim=2,
        )
        return weights_23d

    def save_img(weights, img_path):
        weights = np.fliplr(np.flipud(weights.detach().cpu().numpy()))
        plt.imsave(img_path, weights)
        print(f'Saved: {img_path}')

    tpv_weights_xy = tpv_weights.mean(dim=4).squeeze(0).permute(1, 2, 0)
    tpv_weights_xy_tpv = tpv_weights_xy[:, :, :3]
    tpv_weights_xy_23d = get_weights_23d(tpv_weights_xy)
    weights_tpv = tpv_weights_xy_tpv
    weights_23d = tpv_weights_xy_23d

    save_img(weights_tpv, save_folder_xy + f'/weights_tpv_{frame_id}.png')
    save_img(weights_23d, save_folder_xy + f'/weights_23d_{frame_id}.png')

    tpv_weights_yz = torch.flip(tpv_weights.mean(dim=2).squeeze(0).permute(2, 1, 0), dims=[-1])
    tpv_weights_yz_tpv = tpv_weights_yz[:, :, :3]
    tpv_weights_yz_23d = get_weights_23d(tpv_weights_yz)
    weights_tpv = tpv_weights_yz_tpv
    weights_23d = tpv_weights_yz_23d
    save_img(weights_tpv, save_folder_yz + f'/weights_tpv_{frame_id}.png')
    save_img(weights_23d, save_folder_yz + f'/weights_23d_{frame_id}.png')

    tpv_weights_zx = torch.flip(tpv_weights.mean(dim=3).squeeze(0).permute(2, 1, 0), dims=[-1])
    tpv_weights_zx_tpv = tpv_weights_zx[:, :, :3]
    tpv_weights_zx_23d = get_weights_23d(tpv_weights_zx)
    weights_tpv = tpv_weights_zx_tpv
    weights_23d = tpv_weights_zx_23d
    save_img(weights_tpv, save_folder_zx + f'/weights_tpv_{frame_id}.png')
    save_img(weights_23d, save_folder_zx + f'/weights_23d_{frame_id}.png')

    return
