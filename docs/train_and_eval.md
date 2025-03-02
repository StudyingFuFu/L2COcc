## Training Details

We train our L2COcc for 25 epochs on NVIDIA 4090 GPUs with batch sizes of 4 for both SemanticKITTI and
SSCBench-KITTI360 benchmarks. Training L2COcc-C, L2COcc-L, and L2COcc-D requires approximately 13GB, 9GB, and 14.5 GB GPU memory respectively.

## Pre-preparation
1. Create a symbolic link to the dataset path and create pretrain folder.
    ``` bash
    mkdir data
    ln -s /path/to/SemanticKITTI data/SemanticKitti
    ln -s /path/to/KITTI360 data/Kitti360
    ```
2. Download the pretrained weights ([efficientnet](https://github.com/StudyingFuFu/L2COcc/releases/download/v1.0/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth), [swin](https://github.com/StudyingFuFu/L2COcc/releases/download/v1.0/swin_tiny_patch4_window7_224.pth) and [geodepth](https://github.com/StudyingFuFu/L2COcc/releases/download/v1.0/pretrain_geodepth.pth)) and put them under the pretrain folder.
    ``` bash
    mv efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth pretrain
    mv swin_tiny_patch4_window7_224.pth pretrain
    mv pretrain_geodepth.pth pretrain
    ```
3. Download checkpoints of L2COcc-l ([SemanticKITTI](https://github.com/StudyingFuFu/L2COcc/releases/download/v1.0/l2cocc_l_semantickitti_2421_6066.ckpt), [KITTI360](https://github.com/StudyingFuFu/L2COcc/releases/download/v1.0/l2cocc_l_kitti360_2521_5760.ckpt)) and put them under the pretrain folder if you want to train L2COcc-d directly.
    ``` bash
    mv l2cocc_l_semantickitti_2421_6066.ckpt pretrain
    mv l2cocc_l_kitti360_2521_5760.ckpt pretrain
    ```
## Train

``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--config_path configs/semantickitti_l2cocc_d.py \
--log_folder semantickitti_l2cocc_d \
--seed 7240 \
--log_every_n_steps 100
```

The training logs and checkpoints will be saved under the log_folder

If you want to train other versions of L2COcc, modify the config_path and log_folder configuration options.

The typical training process starts with training L2COcc-l, followed by training L2COcc-d. Before training L2COcc-d, update the lidar_ckpt configuration option to the path of the checkpoint obtained from training L2COcc-l.

## Evaluation

Download the checkpoints we provide and put them under the ckpts folder.

``` bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--eval --ckpt_path ./ckpts/l2cocc_d_semantickitti_1822_4530.ckpt \
--config_path configs/semantickitti_l2cocc_d.py \
--log_folder semantickitti_l2cocc_d_eval --seed 7240 \
--log_every_n_steps 100
```
If you want to evaluate other versions of L2COcc, modify the ckpt_path, config_path and log_folder configuration options.

The evaluation metrics may have slight differences between multi-GPU and single-GPU evaluations. If using multi-GPU evaluation, please modify CUDA_VISIBLE_DEVICES accordingly. The reported results in the paper are based on single-GPU evaluation.

## Evaluation with Saving the Results

``` bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--eval --ckpt_path ./ckpts/l2cocc_d_semantickitti_1822_4530.ckpt \
--config_path configs/semantickitti_l2cocc_d.py \
--log_folder semantickitti_l2cocc_d_eval --seed 7240 \
--log_every_n_steps 100 --save_path pred --save_tpv
```
If you do not want to save the visualization results of the TPV feature maps and TPV aggregrating weights, remove --save_tpv from the command.

The results will be saved into the save_path.

## Submission

``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path ./ckpts/l2cocc_d_semantickitti_1822_4530.ckpt \
--config_path configs/semantickitti_l2cocc_d.py \
--log_folder semantickitti_l2cocc_d_eval --seed 7240 \
--log_every_n_steps 100 --save_path pred --test_mapping
```

## Pretrain
``` bash
# todo: add pretrain scripts
```

<!--  Using the following script to pretrain the depth net and context net.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--config_path configs/CGFormer-Efficient-Swin-SemanticKITTI-Pretrain.py \
--log_folder CGFormer-Efficient-Swin-SemanticKITTI-Pretrain \
--seed 7240 \
--pretrain \
--log_every_n_steps 100
```

Then using the organize_ckpt.py to extract weights for initialization.

```
python organize_ckpt.py --source_path logs/CGFormer-Efficient-Swin-SemanticKITTI-Pretrain/tensorboard/version_0/checkpoints/latest.ckpt --dst_path ckpts/efficientnet-seg-depth.pth
``` -->
