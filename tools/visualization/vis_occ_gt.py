import os
import torch
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tools.visualization.utils import SPTIAL_SHAPE, VOXEL_SIZE, FREE_LABEL
from tools.visualization.utils import show_occ, get_color_map


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize lidar point cloud')
    parser.add_argument('--dataset', type=str, required=True, help='SemanticKitti or SSCBench-Kitti360')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence to visualize')
    parser.add_argument('--root_path', type=str, default='data/semanticKitti', help='Path to dataset')
    parser.add_argument('--save_path', type=str, default='vis', help='Path to save visualization')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of frames to visualize')
    return parser.parse_args()


def get_label_path(root_path, dataset, sequence):
    if dataset == 'SemanticKitti':
        pcd_path = os.path.join(root_path, 'labels', sequence)
    elif dataset == 'Kitti360':
        pcd_path = os.path.join(root_path, 'preprocess', 'labels', sequence)
    else:
        raise ValueError('Invalid dataset')
    return pcd_path


def main():
    args = parse_args()
    sequence = args.sequence
    root_path = args.root_path
    dataset = args.dataset
    save_path = os.path.join(args.save_path, dataset, 'occ_gt', sequence)
    os.makedirs(save_path, exist_ok=True)
    colors = get_color_map(dataset)

    label_path = get_label_path(root_path, dataset, sequence)
    files = os.listdir(label_path)
    files = [file for file in files if file.endswith('1_1.npy')]
    file_names = sorted(files)

    # Create a window for offscreen rendering (set visible=False)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=512, height=512)
    background_color = np.array([1, 1, 1])  # 白色背景
    vis.get_render_option().background_color = background_color

    cnt = 0
    for file_name in file_names:
        if cnt == args.num_frames:
            break
        cnt += 1

        file_path = os.path.join(label_path, file_name)
        occ = np.load(file_path).reshape(SPTIAL_SHAPE).astype(np.int32)
        voxel_show = np.logical_and(occ != FREE_LABEL, occ != 255)
        voxel_size = VOXEL_SIZE
        vis = show_occ(
            torch.from_numpy(occ),
            torch.from_numpy(voxel_show),
            voxel_size=voxel_size,
            colors=colors,
            vis=vis,
            offset=[0, occ.shape[0] * voxel_size[0] * 1.2 * 0, 0],
        )
        view_control = vis.get_view_control()
        # #斜着看
        # look_at = np.array([15, 0, 0])
        # front = np.array([-1, -0.3, 1])
        # up = np.array([0., 0., 1])
        # zoom = np.array([0.6])
        # 俯视
        look_at = np.array([25.6, 0, 0])
        front = np.array([0, 0, 1])
        up = np.array([1, 0, 1])
        zoom = np.array([0.5])
        view_control.set_lookat(look_at)
        view_control.set_front(front)
        view_control.set_up(up)
        view_control.set_zoom(zoom)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.line_width = 5

        vis.poll_events()
        vis.update_renderer()

        # frame_id = int(file_name.split('.')[0])
        rgb_image_path = os.path.join(save_path, file_name.replace('_1_1.npy', '.png'))

        image = vis.capture_screen_float_buffer(do_render=True)
        plt.imsave(rgb_image_path, np.asarray(image))

        vis.clear_geometries()

    vis.destroy_window()


if __name__ == '__main__':
    main()
