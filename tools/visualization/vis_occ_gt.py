import os
import torch
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tools.visualization.utils import SPTIAL_SHAPE, VOXEL_SIZE, FREE_LABEL
from tools.visualization.utils import show_occ, get_color_map, get_label_path, set_view_control, create_window


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Occ GT')
    parser.add_argument('--dataset', type=str, required=True, help='SemanticKitti or Kitti360')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence to visualize')
    parser.add_argument('--view', type=str, default='top_down', help='View to visualize')
    parser.add_argument('--root_path', type=str, default='data/semanticKitti', help='Path to dataset')
    parser.add_argument('--save_path', type=str, default='vis', help='Path to save visualization')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of frames to visualize')
    return parser.parse_args()


def main():
    args = parse_args()
    view = args.view
    dataset = args.dataset
    sequence = args.sequence
    root_path = args.root_path
    label_path = get_label_path(root_path, dataset, sequence)
    save_path = os.path.join(args.save_path, dataset, 'occ_gt', sequence, view)
    os.makedirs(save_path, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis = create_window(vis, view, dataset)

    background_color = np.array([1, 1, 1])  # white
    opt = vis.get_render_option()
    opt.background_color = background_color
    opt.line_width = 5
    colors = get_color_map(dataset)

    files = os.listdir(label_path)
    files = [file for file in files if file.endswith('1_1.npy')]
    file_names = sorted(files)
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
        set_view_control(vis, view, dataset, root_path, sequence)

        vis.poll_events()
        vis.update_renderer()

        rgb_image_path = os.path.join(save_path, file_name.replace('_1_1.npy', '.png'))
        image = vis.capture_screen_float_buffer(do_render=True)
        plt.imsave(rgb_image_path, np.asarray(image))

        vis.clear_geometries()

    vis.destroy_window()


if __name__ == '__main__':
    main()
