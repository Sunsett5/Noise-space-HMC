# Noise-space HMC (N-HMC)


## Getting started 

### 1) Clone the repository

```
git clone https://github.com/Sunsett5/Noise-space-HMC.git

cd Noise-space-HMC
```


### 2) Download pretrained checkpoint

From the [link](https://onedrive.live.com/?authkey=%21AOIJGI8FUQXvFf8&id=72419B431C262344%21103807&cid=72419B431C262344), download the checkpoint "celebahq_p2.pt" and paste it to ./models/;

From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh), download the checkpoint "ffhq_10m.pt" and paste it to ./models/;

From the [link](https://github.com/openai/guided-diffusion), download the checkpoint "lsun_bedroom.pt" and paste it to ./models/.

```
mkdir -p models
wget https://drive.google.com/file/d/1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh/view -P ./models
wget https://drive.google.com/file/d/1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21/view -P ./models
```

Prepare folder storing outputs from experiments.

```
mkdir -p exp/samples/ffhq
mkdir -p exp/samples/celeba_hq

```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

### 3) Download test datasets

```
mkdir -p exp/datasets/ffhq
mkdir -p exp/datasets/celeba_hq

```


### 4) Set environment

We use the external codes for motion-blurring and non-linear deblurring.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

Install dependencies

```
conda env create -f environment.yml
conda activate NHMC
```


## References
This repo is developed based on [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) and [BlindDPS](https://github.com/BlindDPS/blind-dps), especially for forward operations. Please also consider citing them if you use this repo. and [LLE](https://github.com/weigerzan/LLE_inverse_problem/tree)
