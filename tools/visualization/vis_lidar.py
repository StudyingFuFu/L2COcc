import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tools.visualization.utils import POINT_CLOUD_RANGE
from tools.visualization.utils import filter_points


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize lidar point cloud')
    parser.add_argument('--dataset', type=str, required=True, help='SemanticKitti or SSCBench-Kitti360')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence to visualize')
    parser.add_argument('--root_path', type=str, default='data/semanticKitti', help='Path to dataset')
    parser.add_argument('--save_path', type=str, default='vis', help='Path to save visualization')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of frames to visualize')
    return parser.parse_args()


def get_pcd_path(root_path, dataset, sequence):
    if dataset == 'SemanticKitti':
        pcd_path = os.path.join(root_path, 'dataset', 'sequences', sequence, 'velodyne')
    elif dataset == 'Kitti360':
        pcd_path = os.path.join(root_path, 'data_3d_raw', sequence, 'velodyne_points', 'data')
    else:
        raise ValueError('Invalid dataset')
    return pcd_path


def main():
    args = parse_args()
    sequence = args.sequence
    root_path = args.root_path
    dataset = args.dataset
    save_path = os.path.join(args.save_path, dataset, 'velodyne', sequence)
    os.makedirs(save_path, exist_ok=True)

    pcd_path = get_pcd_path(root_path, dataset, sequence)
    pcd_names = os.listdir(pcd_path)
    pcd_names.sort()

    # Create a window for offscreen rendering (set visible=False)
    vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=False)  # Set visible=False to prevent the window from appearing
    vis.create_window(width=512, height=512)  # Set visible=False to prevent the window from appearing
    render_option = vis.get_render_option()
    render_option.point_size = 3.0  # Set the desired point size

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
        vis.poll_events()
        vis.update_renderer()

        view_control = vis.get_view_control()

        # # view 1: 3rd person view
        # look_at = np.array([15, 0, 0])
        # front = np.array([-1, -0.3, 1])
        # up = np.array([0., 0., 1])
        # zoom = 0.6
        # view_control.set_lookat(look_at)
        # view_control.set_front(front)
        # view_control.set_up(up)
        # view_control.set_zoom(zoom)

        # view 2: top-down view
        up = np.array([1., 0., 1])
        zoom = 0.5
        view_control.set_up(up)
        view_control.set_zoom(zoom)

        vis.poll_events()
        vis.update_renderer()

        # Capture and save the image
        image = vis.capture_screen_float_buffer(do_render=True)

        rgb_image_path = os.path.join(save_path, pcd_name.replace('.bin', '.png'))
        plt.imsave(rgb_image_path, np.asarray(image))

        vis.remove_geometry(point_cloud)


if __name__ == '__main__':
    main()
