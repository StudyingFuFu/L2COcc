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
    input_lidar_path = os.path.join(vis_path, 'velodyne', sequence, 'top_down')
    occ_gt_path = os.path.join(vis_path, 'occ_gt', sequence, 'top_down')
    save_path = os.path.join(args.vis_path, 'videos', 'sequences', sequence)
    video_path = os.path.join(save_path, f'tpv_and_weight.mp4')
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
        img = cv2.imread(img_path)
        pcd = cv2.imread(pcd_path)
        gt = cv2.imread(gt_path)

        assert img is not None and pcd is not None and gt is not None
        img_height, img_width = img.shape[:2]
        pcd_height, pcd_width = pcd.shape[:2]
        scale_ratio = pcd_width / img_width
        new_height = int(img_height * scale_ratio)
        img = cv2.resize(img, (pcd_width, new_height))

        COL_W = pcd_width
        gap = None
        GAP_W = COL_W // 20

        canvas_modals = []
        for modal in modals:
            canvas = np.zeros((0, COL_W, 3), dtype=np.uint8)

            tpv_path = os.path.join(vis_path, modal, 'sequences', sequence, 'tpv_feats')
            tpv_xy_path = os.path.join(tpv_path, 'xy', 'feature_' + frame_id + '_b0_n0_pca.png')
            tpv_yz_path = os.path.join(tpv_path, 'yz', 'feature_' + frame_id + '_b0_n0_pca.png')
            tpv_zx_path = os.path.join(tpv_path, 'zx', 'feature_' + frame_id + '_b0_n0_pca.png')
            tpv_xy = cv2.imread(tpv_xy_path)
            tpv_yz = cv2.imread(tpv_yz_path)
            tpv_zx = cv2.imread(tpv_zx_path)

            weight_path = os.path.join(vis_path, modal, 'sequences', sequence, 'tpv_weights')
            weight_xy_path = os.path.join(weight_path, 'xy', 'weights_tpv_' + frame_id + '.png')
            weight_yz_path = os.path.join(weight_path, 'yz', 'weights_tpv_' + frame_id + '.png')
            weight_zx_path = os.path.join(weight_path, 'zx', 'weights_tpv_' + frame_id + '.png')
            weight_xy = cv2.imread(weight_xy_path)
            weight_yz = cv2.imread(weight_yz_path)
            weight_zx = cv2.imread(weight_zx_path)

            assert tpv_xy is not None and tpv_yz is not None and tpv_zx is not None
            assert weight_xy is not None and weight_yz is not None and weight_zx is not None

            tpv_xy = cv2.resize(tpv_xy, (COL_W, int(tpv_xy.shape[0] * COL_W / tpv_xy.shape[1])))
            tpv_yz = cv2.resize(tpv_yz, (COL_W, int(tpv_yz.shape[0] * COL_W / tpv_yz.shape[1])))
            tpv_zx = cv2.resize(tpv_zx, (COL_W, int(tpv_zx.shape[0] * COL_W / tpv_zx.shape[1])))

            weight_xy = cv2.resize(weight_xy, (COL_W, int(weight_xy.shape[0] * COL_W / weight_xy.shape[1])))
            weight_yz = cv2.resize(weight_yz, (COL_W, int(weight_yz.shape[0] * COL_W / weight_yz.shape[1])))
            weight_zx = cv2.resize(weight_zx, (COL_W, int(weight_zx.shape[0] * COL_W / weight_zx.shape[1])))

            # gap_yz_zx = np.ones((img.shape[0] - 2 * tpv_yz.shape[0], COL_W, 3), dtype=np.uint8) * 255
            GAP_W_YZZX = img.shape[0] - 2 * tpv_yz.shape[0]
            gap = np.ones((GAP_W, COL_W, 3), dtype=np.uint8) * 255
            gap_yzzx = np.ones((GAP_W_YZZX, COL_W, 3), dtype=np.uint8) * 255

            canvas = np.concatenate((canvas, tpv_yz), axis=0)
            canvas = np.concatenate((canvas, gap_yzzx), axis=0)
            canvas = np.concatenate((canvas, tpv_zx), axis=0)
            canvas = np.concatenate((canvas, gap), axis=0)
            canvas = np.concatenate((canvas, tpv_xy), axis=0)
            canvas = np.concatenate((canvas, gap), axis=0)

            canvas = np.concatenate((canvas, weight_yz), axis=0)
            canvas = np.concatenate((canvas, gap_yzzx), axis=0)
            canvas = np.concatenate((canvas, weight_zx), axis=0)
            canvas = np.concatenate((canvas, gap), axis=0)
            canvas = np.concatenate((canvas, weight_xy), axis=0)
            canvas = np.concatenate((canvas, gap), axis=0)
            canvas_modals.append(canvas)

        canvas_input_gt = np.zeros((0, COL_W, 3), dtype=np.uint8)
        canvas_input_gt = np.concatenate((canvas_input_gt, img), axis=0)
        canvas_input_gt = np.concatenate((canvas_input_gt, gap), axis=0)
        canvas_input_gt = np.concatenate((canvas_input_gt, pcd), axis=0)

        white = (np.ones_like(canvas_modals[0]) * 255)[canvas_input_gt.shape[0]:, ...]
        canvas_input_gt = np.concatenate((canvas_input_gt, white))
        canvas_input_gt[-(COL_W + GAP_W):-GAP_W, ...] = gt

        assert canvas_input_gt.shape == canvas_modals[0].shape

        gap = np.ones((canvas_input_gt.shape[0], GAP_W, 3), dtype=np.uint8) * 255
        canvas = canvas_input_gt
        for modal in canvas_modals:
            canvas = np.concatenate((canvas, gap), axis=1)
            canvas = np.concatenate((canvas, modal), axis=1)

        # cv2.imshow('Canvas ', canvas)
        # cv2.waitKey(0)
        # pdb.set_trace()
        # cv2.destroyAllWindows()

        # font
        font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
        color = (0, 0, 0)

        # text1
        font_size = int(gap.shape[1] * 1.8)
        font = ImageFont.truetype(font_path, font_size)
        TEXT_POS_Y = int(GAP_W * 0.2)
        text_canvas = np.ones((int(GAP_W * 3.5), canvas.shape[1], 3), dtype=np.uint8) * 255
        text_canvas = Image.fromarray(cv2.cvtColor(text_canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(text_canvas)
        texts_positions_1 = [
            ("Input and GT", (int(COL_W * 0.25), TEXT_POS_Y)),
            ("L2COcc-L", (int(COL_W * 1.35), TEXT_POS_Y)),
            ("L2COcc-C", (int(COL_W * 2.40), TEXT_POS_Y)),
            ("L2COcc-D", (int(COL_W * 3.45), TEXT_POS_Y)),
        ]
        for text, position in texts_positions_1:
            draw.text(position, text, font=font, fill=color)

        # text2
        font_size = int(gap.shape[1] * 0.9)
        font = ImageFont.truetype(font_path, font_size)
        texts_positions_2 = [
            ("(LiDAR only)", (int(COL_W * 1.40), int(GAP_W * 2.1))),
            ("(Cam only)", (int(COL_W * 2.50), int(GAP_W * 2.1))),
            ("(Cam only, with LiDAR distillation)", (int(COL_W * 3.30), int(GAP_W * 2.1))),
        ]
        for text, position in texts_positions_2:
            draw.text(position, text, font=font, fill=color)
        text_canvas = cv2.cvtColor(np.array(text_canvas), cv2.COLOR_RGB2BGR)
        canvas = np.concatenate((text_canvas, canvas), axis=0)

        # text3
        text_canvas_height = canvas.shape[0]
        text_canvas_width = int(GAP_W * 4.7)
        text_canvas = np.ones((text_canvas_height, text_canvas_width, 3), dtype=np.uint8) * 255
        text_canvas = Image.fromarray(cv2.cvtColor(text_canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(text_canvas)
        if dataset == 'Kitti360':
            texts_positions_3 = [
                ("TPV-XY", (int(GAP_W * 0.5), int(GAP_W * 4.2))),
                ("TPV-YZ", (int(GAP_W * 0.5), int(GAP_W * 7.3))),
                ("TPV-ZX", (int(GAP_W * 0.5), int(GAP_W * 19.3))),
                ("Weight-XY", (int(GAP_W * 0.5), int(GAP_W * 32.2))),
                ("Weight-YZ", (int(GAP_W * 0.5), int(GAP_W * 35.0))),
                ("Weight-ZX", (int(GAP_W * 0.5), int(GAP_W * 48))),
            ]
        elif dataset == 'SemanticKitti':
            texts_positions_3 = [
                ("TPV-XY", (int(GAP_W * 0.5), int(COL_W * 0.21))),
                ("TPV-YZ", (int(GAP_W * 0.5), int(COL_W * 0.38))),
                ("TPV-ZX", (int(GAP_W * 0.5), int(COL_W * 1))),
                ("Weight-XY", (int(GAP_W * 0.5), int(COL_W * 1.61))),
                ("Weight-YZ", (int(GAP_W * 0.5), int(COL_W * 1.78))),
                ("Weight-ZX", (int(GAP_W * 0.5), int(COL_W * 2.4))),
            ]
        for text, position in texts_positions_3:
            draw.text(position, text, font=font, fill=color)
        text_canvas = cv2.cvtColor(np.array(text_canvas), cv2.COLOR_RGB2BGR)
        canvas = np.concatenate((canvas, text_canvas), axis=1)

        # font
        font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"  # Times New Roman 字体文件路径
        font_size = int(gap.shape[1] * 2.0)
        font = ImageFont.truetype(font_path, font_size)

        # text4
        text_canvas_height = int(GAP_W * 3)
        text_canvas_width = canvas.shape[1]
        text_canvas = np.ones((text_canvas_height, text_canvas_width, 3), dtype=np.uint8) * 255
        text_canvas = Image.fromarray(cv2.cvtColor(text_canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(text_canvas)
        texts_positions_4 = [
            (
                "Visualizations for TPV Features and TPV Weights (" + dataset.replace('itti', 'ITTI') + ')',
                (int(GAP_W * 17), int(0)),
            ),
        ]
        for text, position in texts_positions_4:
            draw.text(position, text, font=font, fill=color)
        text_canvas = cv2.cvtColor(np.array(text_canvas), cv2.COLOR_RGB2BGR)
        canvas = np.concatenate((text_canvas, canvas), axis=0)

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

        # write
        video.write(final_canvas)
    video.release()
    return


if __name__ == '__main__':
    main()
