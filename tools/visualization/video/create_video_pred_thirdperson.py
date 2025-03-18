import os
import pdb
import cv2
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.visualization.utils import get_label_path, get_img_path


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize lidar point cloud')
    parser.add_argument('--dataset', type=str, required=True, help='SemanticKitti or SSCBench-Kitti360')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence to visualize')
    parser.add_argument('--vis_path', type=str, required=True, help='Path to visulization data')
    parser.add_argument('--root_path', type=str, default=True, help='Path to dataset')
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames to visualize')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second')

    return parser.parse_args()


def main():
    args = parse_args()
    fps = args.fps
    dataset = args.dataset
    vis_path = args.vis_path
    sequence = args.sequence
    root_path = args.root_path
    num_frames = args.num_frames

    label_path = get_label_path(root_path, dataset, sequence)
    input_cam_path = get_img_path(root_path, dataset, sequence)
    input_lidar_path = os.path.join(vis_path, 'velodyne', sequence, 'camera')
    occ_gt_path = os.path.join(vis_path, 'occ_gt', sequence, 'third_person')
    monoscene_path = os.path.join('vis/Monoscene/sequences', sequence, 'occ_pred', 'third_person')
    save_path = os.path.join(args.vis_path, 'videos', 'sequences', sequence)
    video_path = os.path.join(save_path, f'pred_third_person.mp4')
    os.makedirs(save_path, exist_ok=True)

    width = 1920
    height = 1080
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    files = os.listdir(label_path)
    files = [file for file in files if file.endswith('1_1.npy')]
    file_names = sorted(files)
    cnt = 0
    modals = ['lidar', 'cam', 'distill']
    for file_name in file_names:
        if cnt == num_frames:
            break
        cnt += 1
        frame_id = file_name.split('.')[0].replace('_1_1', '')

        img_path = os.path.join(input_cam_path, frame_id + '.png')
        pcd_path = os.path.join(input_lidar_path, frame_id + '.png')
        gt_path = os.path.join(occ_gt_path, frame_id + '.png')
        monoscene_pred_path = os.path.join(monoscene_path, frame_id + '.png')
        img = cv2.imread(img_path)
        pcd = cv2.imread(pcd_path)
        gt = cv2.imread(gt_path)
        monoscene = cv2.imread(monoscene_pred_path)
        COL_W = gt.shape[0]
        gap = None
        GAP_W = COL_W // 10

        assert img is not None and pcd is not None and gt is not None
        img_height, img_width = img.shape[:2]
        scale_ratio = COL_W / img_width
        new_height = int(img_height * scale_ratio)
        img = cv2.resize(img, (COL_W, new_height))
        pcd_height, pcd_width = pcd.shape[:2]
        scale_ratio = COL_W / pcd_width
        new_height = int(pcd_height * scale_ratio)
        pcd = cv2.resize(pcd, (COL_W, new_height))

        img_pcd = np.ones((COL_W, COL_W, 3), dtype=np.uint8) * 255
        img_pcd[COL_W // 5:COL_W // 5 + img.shape[0], :img.shape[1]] = img
        img_pcd[COL_W // 5 + img.shape[0]:COL_W // 5 + img.shape[0] + pcd.shape[0], :pcd.shape[1]] = pcd

        gap = np.ones((COL_W, GAP_W, 3), dtype=np.uint8) * 255
        canvas_0 = np.zeros((COL_W, 0, 3), dtype=np.uint8)
        canvas_0 = np.concatenate((canvas_0, img_pcd), axis=1)
        canvas_0 = np.concatenate((canvas_0, gap), axis=1)
        canvas_0 = np.concatenate((canvas_0, gt), axis=1)
        canvas_0 = np.concatenate((canvas_0, gap), axis=1)
        canvas_0 = np.concatenate((canvas_0, monoscene), axis=1)
        canvas_0 = np.concatenate((canvas_0, gap), axis=1)

        canvas_1 = np.zeros((COL_W, 0, 3), dtype=np.uint8)
        for modal in modals:
            occ_pred_path = os.path.join(vis_path, modal, 'sequences', sequence, 'occ_pred', 'third_person')
            occ_pred_path = os.path.join(occ_pred_path, frame_id + '.png')
            occ_pred = cv2.imread(occ_pred_path)
            canvas_1 = np.concatenate((canvas_1, occ_pred), axis=1)
            canvas_1 = np.concatenate((canvas_1, gap), axis=1)

        canvas = np.concatenate((canvas_0, canvas_1), axis=0)

        # font
        font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
        color = (0, 0, 0)

        # text1
        font_size = int(COL_W * 0.06)
        font = ImageFont.truetype(font_path, font_size)
        canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas)
        texts_positions = [
            ("(a) Input", (int(COL_W * 0.37), COL_W * 0.85)),
            ("(b) Ground Truth", (int(COL_W * 1.37), COL_W * 0.85)),
            ("(c) Monoscene", (int(COL_W * 2.49), COL_W * 0.85)),
            ("(d) L2COcc-L", (int(COL_W * 0.31), COL_W * 1.85)),
            ("(e) L2COcc-C", (int(COL_W * 1.40), COL_W * 1.85)),
            ("(f) L2COcc-D", (int(COL_W * 2.49), COL_W * 1.85)),
        ]
        for text, position in texts_positions:
            draw.text(position, text, font=font, fill=color)

        # text2
        font_size = int(COL_W * 0.04)
        font = ImageFont.truetype(font_path, font_size)
        texts_positions = [
            ("(Camera only, mIoU 11.08)", (int(COL_W * 2.46), int(COL_W * 0.92))),
            ("(LiDAR only, mIoU 23.37)", (int(COL_W * 0.27), int(COL_W * 1.92))),
            ("(Camera only, mIoU 17.03)", (int(COL_W * 1.37), int(COL_W * 1.92))),
            ("(Camera only, with LiDAR distillation, mIoU 18.18)", (int(COL_W * 2.26), int(COL_W * 1.92))),
        ]
        for text, position in texts_positions:
            draw.text(position, text, font=font, fill=color)
        canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

        row_0 = int(1.02 * COL_W)
        row_1 = int(1.17 * COL_W)
        canvas_0 = canvas[:row_0, ...]
        canvas_1 = canvas[row_0:row_1, ...]
        canvas_2 = canvas[row_1:, ...]
        canvas = np.concatenate((canvas_1[int(0.1 * COL_W):, ...], canvas_0), axis=0)
        canvas = np.concatenate((canvas, canvas_2), axis=0)
        canvas = canvas[:, :-GAP_W, :]

        # font
        font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
        font_size = int(COL_W * 0.10)
        font = ImageFont.truetype(font_path, font_size)

        # text3
        canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas)
        texts_positions = [
            ("Visualizations for SSC Predictions (" + dataset.replace('itti', 'ITTI') + ')', (int(COL_W * 0.6),
                                                                                              int(COL_W * 0.05))),
        ]
        for text, position in texts_positions:
            draw.text(position, text, font=font, fill=color)
        canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

        gap = np.ones((int(COL_W * 0.20), canvas.shape[1], 3), dtype=np.uint8) * 255
        canvas = np.concatenate((canvas, gap), axis=0)

        # cv2.imshow('Canvas with Text', canvas)
        # cv2.waitKey(0)
        # pdb.set_trace()
        # cv2.destroyAllWindows()

        # Resize
        aspect_ratio = canvas.shape[1] / canvas.shape[0]
        new_width = int(height * aspect_ratio)
        resized_canvas = cv2.resize(canvas, (new_width, height))
        final_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        x_offset = (width - new_width) // 2
        final_canvas[:, x_offset:x_offset + new_width] = resized_canvas

        # cv2.imshow('Final Canvas', final_canvas)
        # cv2.waitKey(0)
        # pdb.set_trace()
        # cv2.destroyAllWindows()

        # Write
        video.write(final_canvas)
    video.release()
    return


if __name__ == '__main__':
    main()
