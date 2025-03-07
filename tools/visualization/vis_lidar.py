import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tools.visualization.utils import POINT_CLOUD_RANGE
from tools.visualization.utils import filter_points, set_view_control, create_window, get_pcd_path


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize lidar point cloud')
    parser.add_argument('--dataset', type=str, required=True, help='SemanticKitti or SSCBench-Kitti360')
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
    save_path = os.path.join(args.save_path, dataset, 'velodyne', sequence, view)
    os.makedirs(save_path, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis = create_window(vis, view, dataset)

    background_color = np.array([1, 1, 1])  # white
    opt = vis.get_render_option()
    opt.background_color = background_color
    opt.point_size = 2.0

    pcd_path = get_pcd_path(root_path, dataset, sequence)
    pcd_names = os.listdir(pcd_path)
    pcd_names.sort()
    cnt = 0
    for pcd_name in pcd_names:
        frame_id = int(pcd_name.split('.')[0])
        if frame_id % 5 != 0:
            continue
        if cnt == args.num_frames:
            break
        cnt += 1
        pcd = np.fromfile(os.path.join(pcd_path, pcd_name), dtype=np.float32).reshape(-1, 4)
        points = pcd[:, :3]

        # Filter points based on the specified range
        filtered_points = filter_points(points, POINT_CLOUD_RANGE)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

        vis.add_geometry(point_cloud)
        set_view_control(vis, view, dataset, root_path, sequence)
        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(do_render=True)
        rgb_image_path = os.path.join(save_path, pcd_name.replace('.bin', '.png'))
        plt.imsave(rgb_image_path, np.asarray(image))

        vis.remove_geometry(point_cloud)


if __name__ == '__main__':
    main()
