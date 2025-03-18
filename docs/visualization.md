## Visualization
1. Create directories for visualization.
    ``` bash
    mkdir -p vis/SemanticKitti/
    ```
2. Visualize LiDAR points and SSC ground-truth.
    ``` bash
    # LiDAR points
    python tools/visualization/vis_lidar.py --dataset SemanticKitti --root_path data/SemanticKitti --sequence 08 --view camera
    python tools/visualization/vis_occ_gt.py --dataset SemanticKitti --root_path data/SemanticKitti --sequence 08 --view camera
    # --view <mode>: Set the render view to "camera", "third_person" or "top_down".
    ```
3. Evaluation with saving the results.
    ``` bash
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --eval --ckpt_path ./ckpts/l2cocc_d_semantickitti_1822_4530.ckpt \
    --config_path configs/semantickitti_l2cocc_d.py \
    --log_folder semantickitti_l2cocc_d_eval --seed 7240 \
    --log_every_n_steps 100 --save_path pred --save_tpv

    mv pred vis/SemanticKitti/distill
    ```
4. Visualize SSC predictions.
    ``` bash
    python tools/visualization/vis_occ_pred.py --dataset SemanticKitti --pred_path vis/SemanticKitti/distill/ --root_path data/SemanticKitti --sequence 08 --view camera
    ```
5. Modify corresponding configuration options and repeat the above steps for L2COcc-c and L2COcc-l. The above steps are also applicable to the SSCBench-KITTI-360 dataset.
6. Scripts for video generation are available in `tools/visualization/video`. For example, to generate a video of TPV and predictions for sequence 08 of the SemanticKITTI dataset:
    ``` bash
    # install fonts
    sudo apt install texlive-full
    # create video
    python tools/visualization/create_video_{task}.py --dataset SemanticKitti --sequence 08 --vis_path vis/SemanticKitti --root_path data/SemanticKitti --num_frames 500 --fps 4
    # ffmpeg processing
    ffmpeg -i vis/SemanticKitti/videos/sequences/08/{task}.mp4 -vcodec libx264 -acodec aac -strict -2 -movflags faststart vis/SemanticKitti/videos/sequences/08/{task}_ffmpeg.mp4
    ```
    Before running the video generation script, it is necessary to generate visualization results for all L2COcc models from the specified rendering perspective.