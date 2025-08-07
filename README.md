# Noise-space HMC (N-HMC)


## Getting started 

### 1) Clone the repository

```
git clone https://github.com/Sunsett5/Noise-space-HMC.git

cd Noise-space-HMC
```


### 2) Download pretrained checkpoint


```
pip3 install gdown
gdown https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O ./models/ffhq_10m.pt
gdown https://drive.google.com/uc?id=1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21 -O ./models/celeba_hq.ckpt

mkdir -p models/ldm
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./models/ldm
unzip models/ldm/ffhq.zip -d ./models/ldm

mkdir -p models/first_stage_models/vq-f4
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip -P ./models/first_stage_models/vq-f4
unzip models/first_stage_models/vq-f4/vq-f4.zip -d ./models/first_stage_models/vq-f4
```

Download the checkpoint "GOPRO_wVAE.pth"

```
gdown https://drive.google.com/uc?id=1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy -O ./experiments/pretrained/
```


Prepare folder storing outputs from experiments.

```
mkdir -p exp/samples/ffhq
mkdir -p exp/samples/celeba_hq

```

### 3) Download test datasets

```
mkdir -p exp/datasets/ffhq
mkdir -p exp/datasets/celeba_hq

```


### 4) Set environment

We use the external codes for motion-blurring, non-linear deblurring, and model_loader.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
sed -i 's/\bmodels\./bkse.models./g' bkse/models/kernel_encoding/kernel_wizard.py
sed -i 's/\bmodels\./bkse.models./g' bkse/models/kernel_encoding/image_base_model.py
sed -i 's/\bmodels\./bkse.models./g' bkse/models/backbones/resnet.py

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

Install dependencies. Change {DOWNLOAD_DIR} in sed command to your root location.

```
conda env create -f environment.yml
conda activate NHMC
sed -i 's/torch\._six\.string_classes/str/g' /{DOWNLOAD_DIR}/miniconda3/envs/NHMC/lib/python3.8/site-packages/torchvision/datasets/vision.py
sed -i "s/torch\.load(model_path, map_location='cpu')/torch\.load(model_path, map_location='cpu', weights_only=True)/" /{DOWNLOAD_DIR}/.local/lib/python3.8/site-packages/lpips/lpips.py
```

If encounter this bug "ImportError: cannot import name 'VectorQuantizer2' from 'taming.modules.vqvae.quantize'". Download [quantize.py](https://github.com/CompVis/stable-diffusion/issues/72). Then replace this file miniconda/envs/NHMC/lib/python3.8/site-packages/taming/modules/vqvae/quantize.py

## 5) Run experiment
Pixel Space
```
python3 main_sampling.py --ni --dataset ffhq --doc ffhq --algo hmc --timesteps 3 --deg inpaint_random --sigma_0 0.05  -i exp/samples/ffhq/inpaint_random/hmc --tau 1.0 --epsilon 0.05
```
- algo : ddnm, diffpir, dmps, pigdm, reddiff, dps, daps, dmplug, hmc
- timesteps : depends on algorithm. 3 for hmc, 3 for dmplug, 1000 for dps, 100 for diffpir
- deg : forward operator. 
    - sr4
    - sr16
    - hdr
    - random_inpaint
    - deblur_aniso
    - deblur_nonlinear
    - phase_retrieval
- sigma_0 : std dev of measurement noise (sigma_y)
- i : image output folder
For HMC only
- tau : length of 1 HMC update (default 1.0)
- epsilom : length of 1 leapfrog update (default 0.05)

Latent Space
```
python3 main_sampling_latent.py --ni --dataset ffhq --doc ffhq --algo hmc_latent --timesteps 3 --deg inpaint_random --sigma_0 0.05  -i exp/samples/ffhq/inpaint_random/hmc_latent --tau 1.0 --epsilon 0.05
```

- algo : resample_original, hmc_latent
- timesteps : depends on algorithm. 3 for hmc_latent, 500 for resample



## References
This repo is developed based on [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) and [BlindDPS](https://github.com/BlindDPS/blind-dps), especially for forward operations. Please also consider citing them if you use this repo. and [LLE](https://github.com/weigerzan/LLE_inverse_problem/tree)
