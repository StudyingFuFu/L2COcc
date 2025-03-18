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
    input_lidar_topdown_path = os.path.join(vis_path, 'velodyne', sequence, 'top_down')
    input_lidar_thirdperson_path = os.path.join(vis_path, 'velodyne', sequence, 'third_person')
    occ_gt_path = os.path.join(vis_path, 'occ_gt', sequence)
    save_path = os.path.join(args.vis_path, 'videos', 'sequences', sequence)
    video_path = os.path.join(save_path, f'pred_gt.mp4')
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
        pcd_topdown_path = os.path.join(input_lidar_topdown_path, frame_id + '.png')
        pcd_thirdperson_path = os.path.join(input_lidar_thirdperson_path, frame_id + '.png')
        img = cv2.imread(img_path)
        pcd = cv2.imread(pcd_topdown_path)
        pcd_thirdperson = cv2.imread(pcd_thirdperson_path)

        assert img is not None and pcd is not None
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
            occ_pred_topdown_path = os.path.join(vis_path, modal, 'sequences', sequence, 'occ_pred', 'top_down')
            occ_pred_topdown_path = os.path.join(occ_pred_topdown_path, frame_id + '.png')
            occ_pred_topdown = cv2.imread(occ_pred_topdown_path)

            occ_pred_camera_path = os.path.join(vis_path, modal, 'sequences', sequence, 'occ_pred', 'camera')
            occ_pred_camera_path = os.path.join(occ_pred_camera_path, frame_id + '.png')
            occ_pred_camera = cv2.imread(occ_pred_camera_path)

            occ_pred_thirdperson_path = os.path.join(vis_path, modal, 'sequences', sequence, 'occ_pred', 'third_person')
            occ_pred_thirdperson_path = os.path.join(occ_pred_thirdperson_path, frame_id + '.png')
            occ_pred_thirdperson = cv2.imread(occ_pred_thirdperson_path)

            assert occ_pred_topdown is not None and occ_pred_camera is not None and occ_pred_thirdperson is not None
            occ_pred_camera = cv2.resize(occ_pred_camera, (img.shape[1], img.shape[0]))

            assert occ_pred_topdown.shape[1] == occ_pred_camera.shape[1]

            gap = np.ones((GAP_W, COL_W, 3), dtype=np.uint8) * 255
            canvas = np.concatenate((canvas, occ_pred_camera), axis=0)
            canvas = np.concatenate((canvas, gap), axis=0)
            canvas = np.concatenate((canvas, occ_pred_topdown), axis=0)
            canvas = np.concatenate((canvas, gap), axis=0)
            canvas = np.concatenate((canvas, occ_pred_thirdperson), axis=0)
            canvas_modals.append(canvas)

        canvas = np.zeros((0, COL_W, 3), dtype=np.uint8)
        occ_gt_topdown_path = os.path.join(occ_gt_path, 'top_down')
        occ_gt_topdown_path = os.path.join(occ_gt_topdown_path, frame_id + '.png')
        occ_gt_topdown = cv2.imread(occ_gt_topdown_path)
        occ_gt_camera_path = os.path.join(occ_gt_path, 'camera')
        occ_gt_camera_path = os.path.join(occ_gt_camera_path, frame_id + '.png')
        occ_gt_camera = cv2.imread(occ_gt_camera_path)
        occ_gt_thirdperson_path = os.path.join(occ_gt_path, 'third_person')
        occ_gt_thirdperson_path = os.path.join(occ_gt_thirdperson_path, frame_id + '.png')
        occ_gt_thirdperson = cv2.imread(occ_gt_thirdperson_path)

        assert occ_gt_topdown is not None and occ_gt_camera is not None and occ_gt_thirdperson is not None
        occ_gt_camera = cv2.resize(occ_gt_camera, (img.shape[1], img.shape[0]))

        assert occ_pred_topdown.shape[1] == occ_pred_camera.shape[1]

        gap = np.ones((GAP_W, COL_W, 3), dtype=np.uint8) * 255
        canvas = np.concatenate((canvas, occ_gt_camera), axis=0)
        canvas = np.concatenate((canvas, gap), axis=0)
        canvas = np.concatenate((canvas, occ_gt_topdown), axis=0)
        canvas = np.concatenate((canvas, gap), axis=0)
        canvas = np.concatenate((canvas, occ_gt_thirdperson), axis=0)
        canvas_modals.append(canvas)

        canvas_input_gt = np.zeros((0, COL_W, 3), dtype=np.uint8)
        canvas_input_gt = np.concatenate((canvas_input_gt, img), axis=0)
        canvas_input_gt = np.concatenate((canvas_input_gt, gap), axis=0)
        canvas_input_gt = np.concatenate((canvas_input_gt, pcd), axis=0)
        canvas_input_gt = np.concatenate((canvas_input_gt, gap), axis=0)
        canvas_input_gt = np.concatenate((canvas_input_gt, pcd_thirdperson), axis=0)

        assert canvas_input_gt.shape == canvas_modals[0].shape

        gap = np.ones((canvas_input_gt.shape[0], GAP_W, 3), dtype=np.uint8) * 255
        canvas = canvas_input_gt
        for modal in canvas_modals:
            canvas = np.concatenate((canvas, gap), axis=1)
            canvas = np.concatenate((canvas, modal), axis=1)

        # cv2.imshow('Canvas', canvas)
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
            ("Input", (int(COL_W * 0.30), TEXT_POS_Y)),
            ("L2COcc-L", (int(COL_W * 1.35), TEXT_POS_Y)),
            ("L2COcc-C", (int(COL_W * 2.40), TEXT_POS_Y)),
            ("L2COcc-D", (int(COL_W * 3.45), TEXT_POS_Y)),
            ("Ground Truth", (int(COL_W * 4.45), TEXT_POS_Y)),
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

        # font
        font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
        font_size = int(gap.shape[1] * 2.0)
        font = ImageFont.truetype(font_path, font_size)

        # text3
        text_canvas_height = int(GAP_W * 3)
        text_canvas_width = canvas.shape[1]
        text_canvas = np.ones((text_canvas_height, text_canvas_width, 3), dtype=np.uint8) * 255
        text_canvas = Image.fromarray(cv2.cvtColor(text_canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(text_canvas)
        texts_positions_4 = [
            (
                "Visualizations for SSC Predictions and Ground Truth(" + dataset.replace('itti', 'ITTI') + ')',
                (int(GAP_W * 23), int(0)),
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
        aspect_ratio = canvas.shape[0] / canvas.shape[1]
        new_height = int(width * aspect_ratio)
        resized_canvas = cv2.resize(canvas, (width, new_height))
        final_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        y_offset = (height - new_height) // 2
        final_canvas[y_offset:y_offset + new_height, :] = resized_canvas

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
