## Training Details

We train our L2COcc for 25 epochs on NVIDIA 4090 GPUs with batch sizes of 4 for both SemanticKITTI and
SSCBench-KITTI360 benchmarks. Training L2COcc-C, L2COcc-L, and L2COcc-D requires approximately 13GB, 9GB, and 14.5 GB GPU memory respectively.
Before start training, download the corresponding pretrained checkpoints ([efficientnet](https://github.com/pkqbajng/CGFormer/releases/download/v1.0/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth), [pretrain-ckpt](https://github.com/pkqbajng/CGFormer/releases/download/v1.0/efficientnet-seg-depth.pth) and [swin tiny](https://github.com/pkqbajng/CGFormer/releases/download/v1.0/swin_tiny_patch4_window7_224.pth)) and put them under the folder ckpts.

## Pre-preparation
``` bash
# dataset
mkdir data
ln -s /path/to/SemanticKITTI data/SemanticKitti
ln -s /path/to/KITTI360 data/Kitti360

# pretrained weights
mkdir pretrain

```

## Train

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--config_path configs/CGFormer-Efficient-Swin-SemanticKITTI.py \
--log_folder CGFormer-Efficient-Swin-SemanticKITTI \
--seed 7240 \
--log_every_n_steps 100
```

The training logs and checkpoints will be saved under the log_folder.

## Evaluation

Downloading the checkpoints from the model zoo and putting them under the ckpts folder.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path ./ckpts/CGFormer-Efficient-Swin-SemanticKITTI.ckpt \
--config_path configs/semantickitti_CGFormer.py \
--log_folder CGFormer-Efficient-Swin-SemanticKITTI-eval --seed 7240 \
--log_every_n_steps 100
```

## Evaluation with Saving the Results

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path ./ckpts/CGFormer-Efficient-Swin-SemanticKITTI.ckpt \
--config_path configs/semantickitti_CGFormer.py \
--log_folder CGFormer-Efficient-Swin-SemanticKITTI-eval --seed 7240 \
--log_every_n_steps 100 --save_path pred
```

The results will be saved into the save_path.

## Submission

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path ./ckpts/CGFormer-Efficient-Swin-SemanticKITTI.ckpt \
--config_path configs/CGFormer-Efficient-Swin-SemanticKITTI.py \
--log_folder CGFormer-Efficient-Swin-SemanticKITTI-eval --seed 7240 \
--log_every_n_steps 100 --save_path pred --test_mapping
```

## Pretrain

Using the following script to pretrain the depth net and context net.

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
```
