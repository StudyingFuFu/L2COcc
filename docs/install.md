# Step-by-step installation instructions

L2COcc is developed based on the official CGFormer codebase and the installation follows similar steps.

**a. Create a conda virtual environment and activate**


```shell
conda create -n l2cocc python=3.8 -y
conda activate l2cocc
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**

```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

or 

```shell
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

**c. Install mmcv, mmdet, and mmseg**

```shell
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**c. Install mmdet3d 0.17.1 and DFA3D**

We use the same mmdet3d and dfa3d libraries as [CGFormer](https://github.com/pkqbajng/CGFormer). 

```shell
cd packages
bash setup.sh
cd ../
```

**d. Install other dependencies, like timm, einops, torchmetrics, spconv, pytorch-lightning, etc.**

```shell
pip install -r docs/requirements.txt
```

