import random
import math
import argparse, os, yaml
import torch
import torchvision.utils as tvu
import numpy as np
from scipy.stats import invgamma
from scipy.stats import norm
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from util.early_stop import EarlyStop
from guided_diffusion.unet import create_model
import lpips
from tqdm import tqdm
import time
from omegaconf import OmegaConf

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from models.diffusion import Model
from guided_diffusion.unet_ffhq import create_model as create_model_ffhq
from algos.ddnm import DDNM
from algos.dps import DPS
from algos.dps_refine import DPS_refine
from algos.reddiff import RED_diff
from algos.diffpir import DiffPIR
from algos.daps.daps import DAPS
from algos.unconditional import Unconditional

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def init_model(opt, config, model_config, device):
    if config.model_type == 'simple':    
        model = Model(config)
        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif config.data.dataset == "LSUN":
            name = f"lsun_{config.data.category}"
        elif config.data.dataset == 'CelebA_HQ':
            name = 'celeba_hq'
        else:
            raise ValueError
        if name != 'celeba_hq':
            ckpt = get_ckpt_path(f"ema_{name}", prefix=opt.exp)
            print("Loading checkpoint {}".format(ckpt))
        elif name == 'celeba_hq':
            #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
            ckpt = 'models/celeba_hq.ckpt'
            if not os.path.exists(ckpt):
                download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
        else:
            raise ValueError
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        model = torch.nn.DataParallel(model)

    elif config.model_type == 'openai':
        raise NotImplementedError("OpenAI model is not implemented yet (need to change path to the model)")
        config_dict = vars(config.model)
        model = create_model(**config_dict)
        if config.model.use_fp16:
            model.convert_to_fp16()
        if config.model.class_cond:
            ckpt = os.path.join(opt.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (config.data.image_size, config.data.image_size))
            if not os.path.exists(ckpt):
                download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (config.data.image_size, config.data.image_size), ckpt)
        else:
            ckpt = os.path.join(opt.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
            if not os.path.exists(ckpt):
                download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        model.eval()
        model = torch.nn.DataParallel(model)

    elif config.model_type == 'ffhq':
        model = create_model_ffhq(**model_config)
        model = model.to(device)
        model.eval()

    return model


def init_algo(opt, model, H_funcs=None, sigma_0=0.01, deg=None, betas=None):
    if opt.algo == 'ddnm':
        algo = DDNM(model, H_funcs, sigma_0)
    elif opt.algo == 'ddrm':
        algo = DDRM(model, H_funcs, sigma_0)
    elif opt.algo == 'dps':
        if 'celeba' in opt.config:
            if 'cs' in deg:
                lam = 1.0
            elif deg == 'deblur_nonlinear':
                lam = 1.0
            elif 'hdr' in deg:
                lam = 1.0
            elif 'phase' in deg:
                lam = 0.4
            elif 'deblur_aniso' in deg:
                lam = 1.0
            elif 'box' in deg:
                lam = 1.0
            elif 'sr4' in deg:
                lam = 1.0
            else:
                lam = 1.0
        elif 'ffhq' in opt.config:
            if 'cs' in deg:
                lam = 1.0
            elif deg == 'deblur_nonlinear':
                lam = 1.0
            elif 'deblur_aniso' in deg:
                lam = 1.0
            elif 'inpaint' in deg:
                lam = 1.0
            elif 'phase' in deg:
                lam = 0.4
            elif 'sr4' in deg:
                lam = 0
            else:
                lam = 1.0
        else:
            lam = 1.0
        algo = DPS(model, H_funcs, sigma_0, lam=lam)
    elif opt.algo == 'reddiff':
        if 'celeba' in opt.config:
            if 'inp' in deg:
                if 'box' in deg:
                    eta = 0.4
                else:
                    eta = 0.5
            elif 'cs' in deg:
                eta = 0.5
            elif deg == 'deblur_nonlinear':
                eta = 0.2
            elif 'hdr' in deg:
                eta = 0.1
            elif 'sr_bicubic' in deg:
                eta = 3.0
            elif 'sr4' in deg:
                eta = 7.0
            elif 'deblur_aniso' in deg:
                eta = 0.5
            else:
                eta = 1.0
        elif 'ffhq' in opt.config:
            if 'cs' in deg:
                eta = 0.5
            elif deg == 'deblur_nonlinear':
                eta = 0.018
            elif 'deblur_aniso' in deg:
                eta = 0.05
            elif 'inpaint' in deg:
                eta = 0.1
            elif 'sr4' in deg:
                eta = 0.15
            elif 'sr16' in deg:
                eta = 1.5
            elif 'hdr' in deg:
                eta = 0.015
            elif 'phase' in deg:
                eta = 0.5
            else:
                eta = 0.5
        else:
            eta = 1.0
        algo = RED_diff(model, H_funcs, sigma_0, eta=eta)
    elif opt.algo == 'diffpir':
        lam = 7.0
        algo = DiffPIR(model, H_funcs, sigma_0, lam=lam)
    elif opt.algo == 'daps':
        daps_cfg = OmegaConf.load("configs/daps/edm_daps.yaml")
        if 'sr4' in opt.deg:
            mcmc_num_steps = 100
            lr = 1e-4
            lr_min_ratio = 1e-2
        elif 'sr16' in opt.deg:
            mcmc_num_steps = 100
            lr = 1e-4
            lr_min_ratio = 1e-2
        elif ('inpaint' in opt.deg) and  ('random' in opt.deg):
            mcmc_num_steps = 100
            lr = 1e-4
            lr_min_ratio = 1e-2
        elif 'aniso' in opt.deg:
            mcmc_num_steps = 100
            lr = 1e-4
            lr_min_ratio = 1e-2
        elif 'nonlinear' in opt.deg:
            mcmc_num_steps = 100
            lr = 5e-5
            lr_min_ratio = 1e-2
        elif 'phase' in opt.deg:
            mcmc_num_steps = 100
            lr = 5e-5
            lr_min_ratio = 1e-2
        elif 'hdr' in opt.deg:
            mcmc_num_steps = 100
            lr = 2e-5
            lr_min_ratio = 1e-2
        else:
            mcmc_num_steps = 100
            lr = 1e-4
            lr_min_ratio = 1e-2

        timesteps = opt.timesteps
        algo = DAPS(daps_cfg, model, H_funcs, sigma_0, timesteps, mcmc_num_steps, lr, lr_min_ratio, betas=betas)

    elif opt.algo == 'dmplug_adam' or opt.algo == 'hmc' or opt.algo == 'hmc_window' or opt.algo == 'hmc_blind_noise':
        algo = Unconditional(model, H_funcs, sigma_0)
    else:
        raise NotImplementedError

    #algo_dps = DPS_refine(model, H_funcs, sigma_0, lam=0.8)

    from obs_functions.Hfuncs import Deblurring
    #sigma_refine = 0.75
    sigma_refine = opt.sigma_refine
    lam_refine = opt.lam_refine
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma_refine)**2]))
    kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
    refine_H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, config.data.image_size, device)
    algo_dps = DPS(model, refine_H_funcs, 0, lam=lam_refine)
    
    return algo, algo_dps, refine_H_funcs

def prepare_measurement(opt, task_config, device):
    ## get degradation matrix ##
    deg = opt.deg
    H_funcs = None
    if 'sr' in deg:
        if deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from obs_functions.Hfuncs import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                            config.data.channels, config.data.image_size, device, stride = factor)
        else:
            # Super-Resolution
            blur_by = int(deg[2:])
            from obs_functions.Hfuncs import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, device)
    elif 'inp' in deg:
        if 'box' in deg:
            missing = torch.zeros([config.data.image_size, config.data.image_size, config.data.channels])
            left = random.randint(16, 112)
            up = random.randint(16, 112)
            #left = 64
            #up = 64
            missing[left:left+128, up:up+128, :] = 1.0
            missing = missing.view(-1).to(device).long()
            missing = torch.nonzero(missing).squeeze() 
        else:
            # Random inpainting
            missing_r = 3 * torch.randperm(config.data.image_size**2)[:int(config.data.image_size**2 * 0.92)].to(device).long()
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        from obs_functions.Hfuncs import Inpainting
        H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, device)
    elif 'deblur_gauss' in deg:
        # Gaussian Deblurring
        from obs_functions.Hfuncs import Deblurring
        sigma = 1.0
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, config.data.image_size, device)
    elif 'phase' in deg:
        # Phase Retrieval
        from obs_functions.Hfuncs import PhaseRetrievalOperator
        H_funcs = PhaseRetrievalOperator(oversample=2.0, device=device)
    elif 'hdr' in deg:
        # HDR
        from obs_functions.Hfuncs import HDR
        H_funcs = HDR()   
    elif 'cs' in deg:
        compress_by = int(deg[2:])
        from obs_functions.Hfuncs import WalshHadamardCS
        H_funcs = WalshHadamardCS(config.data.channels, config.data.image_size, compress_by, torch.randperm(config.data.image_size**2, device=device), device)
    elif deg == 'deblur_aniso':
        from obs_functions.Hfuncs import Deblurring2D
        sigma = 20
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
        sigma = 1
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
        H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, config.data.image_size, device)
    elif deg == 'deblur_nonlinear':
        from obs_functions.Hfuncs import NonlinearBlurOperator
        H_funcs = NonlinearBlurOperator(device, opt_yml_path='./bkse/options/generate_blur/default.yml')
    elif deg == 'color':
        from obs_functions.Hfuncs import Colorization
        H_funcs = Colorization(config.data.image_size, device)
    else:
        print("ERROR: degradation type not supported")
        quit()

    # for linear observations
    # if 'sr' in deg or 'inp' in deg or 'deblur_gauss' in deg:
    opt.sigma_0 = 2 * opt.sigma_0 #to account for scaling to [-1,1]
    sigma_0 = opt.sigma_0

    return H_funcs, sigma_0, deg


def sample_image(opt, config=None, model_config=None, device='cuda'):
    H_funcs, sigma_0, deg = prepare_measurement(opt, config, device)
    model = init_model(opt, config, model_config, device)
    betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

    algo, algo_dps, refine_H_funcs = init_algo(opt, model, H_funcs, sigma_0, deg, betas)

    #get original images and corrupted y_0
    dataset, test_dataset = get_dataset(opt, config)
    
    device_count = torch.cuda.device_count()
    
    if opt.subset_start >= 0 and opt.subset_end > 0:
        assert opt.subset_end > opt.subset_start
        test_dataset = torch.utils.data.Subset(test_dataset, range(opt.subset_start, opt.subset_end))
    else:
        opt.subset_start = 0
        opt.subset_end = len(test_dataset)

    print(f'Dataset has size {len(test_dataset)}')    
    
    def seed_worker(worker_id):
        worker_seed = opt.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(opt.seed)
    if 'phase' in opt.deg:
        if config.sampling.batch_size > 1:
            key = input('Recommend using batch size 1. Current batch size is {}, switch to 1? [y/n]'.format(config.sampling.batch_size))
            if key == 'y':
                config.sampling.batch_size = 1
                print('switch to 1')
            else:
                print('keep using {}'.format(config.sampling.batch_size))

    val_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        
    # step size
    if opt.default_lr: # using default step size to reproduce the metrics
        raise NotImplementedError("Default step size is not implemented yet")
        N = 1
        steps=opt.timesteps
        if 'imagenet' in opt.config:
            dataset_name = 'imagenet'
        elif 'celeba' in opt.config:
            dataset_name = 'celeba'
        elif 'ffhq' in opt.config:
            dataset_name = 'ffhq'
        else:
            dataset_name = 'unknown'
        # print(deg)
        # print(steps)
        # print(sigma_0)
        # print(dataset_name)
        #lr = get_default_lr(deg, steps, sigma_0, dataset_name)
    else:
        lr = opt.lr
        N = opt.N

    print(f'Start from {opt.subset_start}')
    idx_init = opt.subset_start
    idx_so_far = opt.subset_start
    avg_psnr = 0.0
    avg_best_psnr = 0.0
    avg_ssim = 0.0
    avg_best_ssim = 0.0
    avg_lpips = 0.0
    avg_best_lpips = 0.0
    std_psnr = 0.0
    std_ssim = 0.0
    std_lpips = 0.0
    pbar = tqdm(val_loader)
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    
    for i_img, (x_orig, classes) in enumerate(pbar):

        #if i_img <= 5:
        #    continue

        x_orig = x_orig.to(device)
        x_orig = data_transform(config, x_orig)

        y_0 = H_funcs.H(x_orig).detach()
        y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
        y_pinv = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, config.data.image_size, config.data.image_size)
        os.makedirs(opt.image_folder, exist_ok=True)

        for i in range(len(y_0)):
            tvu.save_image(
                inverse_data_transform(config, y_pinv[i]), os.path.join(opt.image_folder, f"y0_{idx_so_far + i}.png")
            )
            tvu.save_image(
                inverse_data_transform(config, x_orig[i]), os.path.join(opt.image_folder, f"orig_{idx_so_far + i}.png")
            )

        x = torch.randn(
                        y_0.shape[0],
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=device,
                    )
        

        if opt.algo == 'dmplug_adam' or opt.algo == 'hmc' or opt.algo == 'hmc_window' or opt.algo == 'hmc_blind_noise':
            skip = (opt.num_timesteps) // (opt.timesteps+1)
        else:
            skip = opt.num_timesteps // opt.timesteps
        seq = list(range(skip, opt.num_timesteps, skip))
        seq_next = [-1] + list(seq[:-1])
        skip_dps = 1
        seq_dps = list(range(skip_dps, opt.num_timesteps, skip_dps))
        seq_dps_next = [-1] + list(seq_dps[:-1])
        xt = x
        n = x.shape[0]
        
        b = torch.from_numpy(betas).float().to(device)
        
        if opt.algo == 'dmplug_adam':
            xt, xt_best = dmplug_adam(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig)
        elif opt.algo == 'hmc':
            xt = hmc(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig)
            if opt.refine:
                x_0 = refine_H_funcs.H(xt[-1:])
                xt = iterative_sampling(x, n, b, seq_dps, seq_dps_next, algo_dps, opt, x_0, tqdm_disable=True)
        elif opt.algo == 'hmc_window':
            xt = hmc_window(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig)
        elif opt.algo == 'hmc_blind_noise':
            xt = hmc_blind_noise(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig)
        else:
            with torch.no_grad():
                xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True)

        with torch.no_grad():

            final = torch.stack([inverse_data_transform(config, y) for y in xt])
            if opt.algo == 'dmplug_adam':
                final_best = torch.stack([inverse_data_transform(config, y) for y in xt_best])

            if len(xt) > 1:
                std = final.std(dim=0)  
                std_mean = std.mean(dim=0)
                std_plot = (std_mean - std_mean.min()) / (std_mean.max() - std_mean.min())
                # Create subplots
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                im = ax.imshow(std_plot.cpu().numpy(), cmap='hot')
                fig.colorbar(im, ax=ax, label="Std Dev")
                ax.set_title("Pixel-wise Std Dev Across Samples")
                ax.axis('off')

                # Save to PNG
                plt.tight_layout()
                plt.savefig(os.path.join(opt.image_folder, f"std_dev_map_{idx_so_far}.png"), dpi=300)
                plt.close()


            metrics_sum = [[], [], []]
            for j in range(len(final)):
                if j == len(final) - 1:
                    tvu.save_image(
                        final[j], os.path.join(opt.image_folder, f"{idx_so_far}.png")
                    )
                orig = inverse_data_transform(config, x_orig[0])
                mse = torch.mean((final[j].to(device) - orig) ** 2)
                PSNR = 10 * torch.log10(1 / mse)
                SSIM = ssim(final[j].detach().cpu().numpy(), orig.detach().cpu().numpy(), data_range=final[j].detach().cpu().numpy().max() - final[j].detach().cpu().numpy().min(), channel_axis=0)
                LPIPS = loss_fn_vgg(2*orig-1.0, 2*torch.tensor(final[j]).to(torch.float32).cuda()-1.0)[0,0,0,0]
                metrics_sum[0].append(PSNR.item())
                metrics_sum[1].append(SSIM)
                metrics_sum[2].append(LPIPS.item())

            avg_psnr += np.mean(metrics_sum[0])
            avg_ssim += np.mean(metrics_sum[1])
            avg_lpips += np.mean(metrics_sum[2])

            idx_so_far += y_0.shape[0]
            num_idx = idx_so_far - idx_init

            if len(xt) == 1:
                pbar.set_description("PSNR:{:.4f}, SSIM:{:.5f}, LPIPS:{:.5f}".format(avg_psnr / num_idx, avg_ssim / num_idx, avg_lpips / num_idx))
            else:
                std_psnr += np.std(metrics_sum[0], ddof=1)
                std_ssim += np.std(metrics_sum[1], ddof=1)
                std_lpips += np.std(metrics_sum[2], ddof=1)
                pbar.set_description("PSNR:{:.4f} ({:.4f}), SSIM:{:.5f} ({:.5f}), LPIPS:{:.5f} ({:.5f})".format(
                    avg_psnr / num_idx, std_psnr / (i_img+1),
                    avg_ssim / num_idx, std_ssim / (i_img+1),
                    avg_lpips / num_idx, std_lpips / (i_img+1)))
                
            if opt.algo == 'dmplug_adam':
                mse = torch.mean((final_best[0].to(device) - orig) ** 2)
                PSNR = 10 * torch.log10(1 / mse)
                SSIM = ssim(final_best[0].detach().cpu().numpy(), orig.detach().cpu().numpy(), data_range=final_best[0].detach().cpu().numpy().max() - final_best[0].detach().cpu().numpy().min(), channel_axis=0)
                LPIPS = loss_fn_vgg(2*orig-1.0, 2*torch.tensor(final_best[0]).to(torch.float32).cuda()-1.0)[0,0,0,0]
                avg_best_psnr += PSNR.item()
                avg_best_ssim += SSIM
                avg_best_lpips += LPIPS.item()

    avg_psnr = avg_psnr / num_idx
    avg_ssim = avg_ssim / num_idx
    avg_lpips = avg_lpips / num_idx
    std_psnr = std_psnr / (i_img+1)
    std_ssim = std_ssim / (i_img+1)
    std_lpips = std_lpips / (i_img+1)
    print("Total Average PSNR: {:.3f} ({:.4f})".format(avg_psnr, std_psnr))
    print("Total Average SSIM: {:.5f} ({:.5f})".format(avg_ssim, std_ssim))
    print("Total Average LPIPS: {:.5f} ({:.5f})".format(avg_lpips, std_lpips))
    print("Number of samples: {}".format(num_idx))

    if opt.algo == 'dmplug_adam':
        avg_best_psnr = avg_best_psnr / num_idx
        avg_best_ssim = avg_best_ssim / num_idx
        avg_best_lpips = avg_best_lpips / num_idx
        print("Total Average Best PSNR: {:.3f}".format(avg_best_psnr))
        print("Total Average Best SSIM: {:.5f}".format(avg_best_ssim))
        print("Total Average Best LPIPS: {:.5f}".format(avg_best_lpips))

    return avg_psnr, avg_ssim, avg_lpips

def dmplug_adam(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig):
    x = x.requires_grad_()
    params_group1 = {'params': x, 'lr': 1e-2}
    optimizer = torch.optim.Adam([params_group1])

    orig_pic = []
    for j in range(len(x_orig)):
        orig_pic.append(inverse_data_transform(config, x_orig[j]))
    psnr_list = []

    best_psnr = 0
    epochs = 10000
    buffer_size = 50
    patience = 300
    earlystop = EarlyStop(size=buffer_size,patience=patience)
    variance_history = []
    running = True

    xt_early = None
    xt_best = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
        x_save = [inverse_data_transform(config, y) for y in xt.detach()]
        for j in range(len(x_save)):
            r_img_np = xt.detach().reshape(-1)
            earlystop.update_img_collection(r_img_np)
            img_collection = earlystop.get_img_collection()
            if running and len(img_collection) == buffer_size:
                ave_img = sum(img_collection)/len(img_collection)
                variance = []
                for tmp in img_collection:
                    variance.append(((ave_img - tmp) ** 2).sum())
                cur_var = sum(variance)/len(variance)
                cur_epoch = epoch
                variance_history.append(cur_var)
                #if earlystop.stop == False:
                earlystop.stop = earlystop.check_stop(cur_var, cur_epoch)
                if earlystop.stop:
                    #print(f"Early stopping at epoch {epoch}, variance: {cur_var}")
                    xt_early = xt.detach().clone()
                    running = False
            #tvu.save_image(
            #    x_save[j], os.path.join(opt.image_folder, f"dmplug_{epoch}.png")
            #)
            mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            psnr_list.append(psnr.item())
            if psnr > best_psnr:
                best_psnr = psnr
                xt_best = xt.detach().clone()
            elif best_psnr - psnr > 1.0 and not running:
                 return (xt_early, xt_best)
            print('Best PSNR:', best_psnr.item(), 'PSNR:', psnr.item(), 'count:', earlystop.wait_count)
        error = y_0 - H_funcs.H(xt)
        loss = torch.sum(error**2)
        loss.backward()
        optimizer.step()
        

    """ # plot the PSNR
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(psnr_list, label='PSNR')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR')
    plt.savefig(os.path.join(opt.image_folder, 'dmplug_psnr.png')) """

    if xt_early is None:
        xt_early = xt.detach().clone()

    return (xt_early, xt_best)


def hmc_adaptive_mass(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig):

    orig_pic = []
    for j in range(len(x_orig)):
        orig_pic.append(inverse_data_transform(config, x_orig[j]))

    """ init_time = time.time()
    x = warm_start(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, opt.sigma_0, orig_pic) * math.sqrt(0.8) + torch.randn_like(x) * math.sqrt(0.2)
    xt = iterative_sampling(
        x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True
    ).clip(-1, 1)
    x_save = [inverse_data_transform(config, y) for y in xt.detach()]
    for j in range(len(x_save)):
        tvu.save_image(
            x_save[j], os.path.join(opt.image_folder, f"warm_start.png")
        )
        mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
        psnr = 10 * torch.log10(1 / mse)

        print('warm_start', 'PSNR:', psnr.item())
    print('Warm start time:', time.time() - init_time) """

    x = x.detach().requires_grad_()
    epsilon = opt.epsilon
    L = opt.L
    if 'phase' in opt.deg:
        warm_up = 75
        sampling = 10
        gamma = 0.98
        total_epochs = warm_up + 3 * sampling
    else:
        warm_up = 30
        sampling = 10
        gamma = 0.95
        total_epochs = warm_up + 4 * sampling

    M_diag = torch.ones_like(x).reshape(-1)
    std_diag = torch.sqrt(M_diag)
    inv_M_diag = 1.0 / M_diag

    psnr_list = []
    loss_list = []
    final_img_list = []
    sigma_y_list = []

    d_y = y_0.view(-1).numel()

    accepted = 0
    rejected = 0
    epsilon_decay = 0

    variance_grad = torch.zeros_like(x, device=device, requires_grad=False)
    beta = 0.95
    lam = 0.5
    rho = 0.85

    epoch = 0

    while epoch < total_epochs:
        if 'phase' in opt.deg:
            if epoch <= warm_up:
                sigma_y = opt.sigma_0 + 20 * (1 - epoch / warm_up) ** 2
        else:
            if epoch <= warm_up:
                sigma_y = opt.sigma_0 + (1.7-opt.sigma_0) * (1 - epoch / warm_up) ** 3
                #sigma_y = opt.sigma_0

            elif epoch == warm_up + 1:
                sigma_y = opt.sigma_0
                if epsilon > 0.005:
                    epsilon = 0.005
                    pass

        # initialize momentum
        p = torch.randn_like(x, device=device) * std_diag.view_as(x)
       
        xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
        xt_start = xt.detach().clone()
        loss = torch.sum((y_0 - H_funcs.H(xt))**2)
        current_loss = loss.detach()
        loss_grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
        total_current_loss = (1/2) * torch.sum(x.detach()**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * current_loss

        H = total_current_loss + (1/2) * torch.sum(inv_M_diag.view_as(x) * p**2)

        x_proposal = x.detach().clone().requires_grad_(True)

        early_stop = False

        for l in range(L):

            # update momentum
            p = p - (epsilon / 2) * (x_proposal.detach() + 1/(2 * sigma_y**2) * loss_grad)

            x_proposal = x_proposal + epsilon * inv_M_diag.view_as(x) * p 
            x_proposal = x_proposal.detach().requires_grad_(True)

            xt = iterative_sampling(x_proposal, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
            predicted_meas = H_funcs.H(xt)
            loss = torch.sum((y_0 - predicted_meas)**2)
            loss_grad = torch.autograd.grad(loss, x_proposal, retain_graph=False)[0]
            posterior_grad = x_proposal.detach() + 1/(2 * sigma_y**2) * loss_grad
            variance_grad = beta * variance_grad + (1 - beta) * posterior_grad**2

            p = p - (epsilon/2) * posterior_grad

            """ H_e = (1/2) * torch.sum(x_proposal.detach()**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * loss.detach() + (1/2) * torch.sum(inv_M_diag.view_as(x) * p**2)
            
            delta_H = H_e - H
            #print('Delta_H', delta_H)
            if delta_H > 100:
                #print('delta_H too large, stopping early')
                early_stop = True
                accept = False
                break """

        if not early_stop:
            proposal_loss = loss.detach()
            total_proposal_loss = (1/2) * torch.sum(x_proposal.detach()**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * proposal_loss
            H_proposal = total_proposal_loss + (1/2) * torch.sum(inv_M_diag.view_as(x) * p**2)
            delta_H = H_proposal - H
            acceptance_ratio = min(torch.tensor([1], device=device), torch.exp(-delta_H))
            accept = torch.rand(1).item() < acceptance_ratio.item()

        if accept:
            if warm_up < epoch <=  warm_up + 1 * sampling:
                M_diag_current = (1-lam) * variance_grad + lam * torch.ones_like(variance_grad, device=device)
                M_diag = torch.exp((1-rho) * torch.log(M_diag_current.view(-1)) + rho * torch.log(M_diag))
                M_diag = M_diag.clip(min=1)
                std_diag = torch.sqrt(M_diag)
                inv_M_diag = 1.0 / M_diag 
                
            accepted += 1
            rejected = 0
            x_accept = xt.detach().clone()

            epoch += 1
            if epoch >= warm_up + 3 * sampling:
                final_img_list.append(x_accept[0])

            x = x_proposal.detach().clone().requires_grad_(True)
            
            x_save = [inverse_data_transform(config, y) for y in xt.detach()]
            for j in range(len(x_save)):
                #if epoch % 10 == 0:
                #tvu.save_image(
                #    x_save[j], os.path.join(opt.image_folder, f"hmc_{epoch}.png")
                #)
                mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                psnr_list.append(psnr.item())
                sigma_y_list.append(sigma_y)
                loss_list.append(loss.item())
                print('epoch', epoch, 'PSNR:', psnr.item(), 'sigma_y:', sigma_y, 'epsilon', epsilon)
        else:
            print(rejected, 'rejected')
            if epoch <= warm_up + 1 * sampling:
                epsilon = epsilon * gamma

    print(len(final_img_list), 'final images collected')
    return torch.stack(final_img_list), x.detach().requires_grad_(True)

def hmc(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig):

    orig_pic = []
    for j in range(len(x_orig)):
        orig_pic.append(inverse_data_transform(config, x_orig[j]))

    x = x.detach().requires_grad_()
    epsilon = opt.epsilon
    L = opt.L
    if 'phase' in opt.deg:
        warm_up = 100
        sampling = 10
        gamma_accept = 1.0
        gamma_reject = 0.95
        total_epochs = warm_up + 2 * sampling
    else:
        warm_up = 30
        sampling = 10
        gamma_accept = 0.98
        gamma_reject = 0.95
        total_epochs = warm_up + 2 * sampling

    M_diag = torch.ones_like(x).reshape(-1)
    std_diag = torch.sqrt(M_diag)
    inv_M_diag = 1.0 / M_diag

    psnr_list = []
    loss_list = []
    final_img_list = []
    sigma_y_list = []

    d_y = y_0.view(-1).numel()
    d = x.view(-1).numel()

    accepted = 0
    rejected = 0
    epsilon_decay = 0


    epoch = 0

    if 'phase' in opt.deg:
        schedule = np.arange(warm_up)/warm_up
        sigma_start = 10
        sigma_mid = 5
        sigma_0 = opt.sigma_0
        k= 2/3
        sigma_y_schedule = sigma_schedule(schedule, sigma_start, sigma_mid, sigma_0, k)

    while epoch < total_epochs:
        if 'phase' in opt.deg:
            if epoch < warm_up:
                sigma_y = sigma_y_schedule[epoch]
            else:
                sigma_y = opt.sigma_0
        else:
            if epoch < warm_up:
                sigma_y = opt.sigma_0 + (3.0-opt.sigma_0) * (1 - epoch / warm_up) ** 4
                #sigma_y = opt.sigma_0

            elif epoch == warm_up:
                sigma_y = opt.sigma_0
                if epsilon > 0.01:
                    epsilon = 0.01
                    pass

        # initialize momentum
        p = torch.randn_like(x, device=device) * std_diag.view_as(x)
       
        xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
        xt_start = xt.detach().clone()
        loss = torch.sum((y_0 - H_funcs.H(xt))**2)
        current_loss = loss.detach()
        loss_grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
        total_current_loss = (1/2) * torch.sum(x.detach()**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * current_loss

        H = total_current_loss + (1/2) * torch.sum(inv_M_diag.view_as(x) * p**2)

        x_proposal = x.detach().clone().requires_grad_(True)

        early_stop = False

        for l in range(L):

            # update momentum
            p = p - (epsilon / 2) * (x_proposal.detach() + 1/(2 * sigma_y**2) * loss_grad)

            x_proposal = x_proposal + epsilon * inv_M_diag.view_as(x) * p 
            x_proposal = x_proposal.detach().requires_grad_(True)

            xt = iterative_sampling(x_proposal, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
            predicted_meas = H_funcs.H(xt)
            loss = torch.sum((y_0 - predicted_meas)**2)
            loss_grad = torch.autograd.grad(loss, x_proposal, retain_graph=False)[0]
            posterior_grad = x_proposal.detach() + 1/(2 * sigma_y**2) * loss_grad

            p = p - (epsilon/2) * posterior_grad

            #H_e = (1/2) * torch.sum(x_proposal.detach()**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * loss.detach() + (1/2) * torch.sum(inv_M_diag.view_as(x) * p**2)
            
            #delta_H = H_e - H
            #print('Delta_H', delta_H)
            #proposal_loss = loss.detach()
            #total_proposal_loss = (1/2) * torch.sum(x_proposal.detach()**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * proposal_loss
            #print('ratio', total_proposal_loss.item() / total_current_loss.item())
            #if (epoch < warm_up) and (total_proposal_loss/total_current_loss > 1.02):
            ##    print('stopping early')
            #    early_stop = True
            #    accept = False
            #    break

        if not early_stop:
            proposal_loss = loss.detach()
            total_proposal_loss = (1/2) * torch.sum(x_proposal.detach()**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * proposal_loss
            H_proposal = total_proposal_loss + (1/2) * torch.sum(inv_M_diag.view_as(x) * p**2)
            delta_H = H_proposal - H
            if (epoch < warm_up) and (total_proposal_loss < total_current_loss):
                accept = True
            else:
                acceptance_ratio = min(torch.tensor([1], device=device), torch.exp(-delta_H))
                accept = torch.rand(1).item() < acceptance_ratio.item()

        if accept:
            rejected = 0
            if epoch < warm_up:
                epsilon = epsilon * gamma_accept
            x_accept = xt.detach().clone()

            if epoch >= warm_up + sampling:
                final_img_list.append(x_accept[0])
            epoch += 1

            x = x_proposal.detach().clone().requires_grad_(True)
            
            if opt.verbose:
                x_save = [inverse_data_transform(config, y) for y in xt.detach()]
                for j in range(len(x_save)):
                    tvu.save_image(
                        x_save[j], os.path.join(opt.image_folder, f"hmc_{epoch}.png")
                    )
                    mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
                    psnr = 10 * torch.log10(1 / mse)
                    psnr_list.append(psnr.item())
                    sigma_y_list.append(sigma_y)
                    loss_list.append(loss.item())
                    print('epoch', epoch, 'PSNR:', psnr.item(), 'sigma_y:', sigma_y, 'epsilon', epsilon, 'ratio', torch.linalg.norm(x).item()/math.sqrt(d))
        else:
            rejected += 1
            if rejected >= 10 and epoch >= warm_up + 1 * sampling:
                break
            if opt.verbose: print(rejected, 'rejected')
            if epoch < warm_up + 1 * sampling:
                epsilon = epsilon * gamma_reject
            elif epoch == warm_up + 1 * sampling:
                epsilon = epsilon * 0.75

    if len(final_img_list) == 0:
        final_img_list.append(x_accept[0])

    return torch.stack(final_img_list)

def sigma_schedule(x, sigma_start, sigma_mid, sigma_0, k):
    """
    Piecewise annealing schedule for sigma_y.

    Parameters
    ----------
    x : float or np.ndarray
        Normalized position in [0, 1].
    sigma_start : float
        Starting sigma value at x=0.
    sigma_mid : float
        Sigma value at transition point k.
    sigma_0 : float
        Final sigma value at x=1.
    k : float
        Transition point between linear and quadratic [0, 1].
    """
    x = np.asarray(x)

    # Precompute slopes for the linear part
    m_linear = (sigma_mid - sigma_start) / k if k != 0 else 0

    sigma = np.zeros_like(x, dtype=float)

    # First part:
    mask1 = x <= k
    t = x[mask1]/(k)
    sigma[mask1] = sigma_mid + (sigma_start - sigma_mid) * (1 - t)

    # Second part:
    mask2 = ~mask1
    t = (x[mask2] - k) / (1 - k)  # normalize from 0 to 1 over the second part
    sigma[mask2] = sigma_0 + (sigma_mid - sigma_0) * (1-t)**1.5

    return sigma


def warm_start(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, sigma_y, orig_pic):
    x_param = torch.nn.Parameter(x.detach().clone())  

    # Define optimizer
    optimizer = torch.optim.LBFGS(
        [x_param], 
        lr=1.0,       # Acts as max step size; LBFGS will do line search
        max_iter=5,  # Max number of iterations per call
        history_size=10
    )

    # Define closure for LBFGS
    def closure():
        optimizer.zero_grad()
        
        xt = iterative_sampling(
            x_param, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True
        ).clip(-1, 1)
        
        # Measurement prediction
        predicted_meas = H_funcs.H(xt)
        
        # Loss = prior term + likelihood term
        meas_loss = torch.sum((y_0 - predicted_meas)**2)
        total_loss = (0.5 * torch.sum(x_param**2) 
                    + (0.5 / sigma_y**2) * meas_loss)
        
        total_loss.backward()

        x_save = [inverse_data_transform(config, y) for y in xt.detach()]
        for j in range(len(x_save)):
            mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
            psnr = 10 * torch.log10(1 / mse)

            print('warm_start', 'PSNR:', psnr.item())

        return total_loss

    # Run a fixed number of LBFGS steps
    for _ in range(10):  # Outer loop to allow line search to converge multiple times
        optimizer.step(closure)

    # Use the optimized value for HMC initialization
    return x_param.detach().clone().requires_grad_(True)

def hmc_window(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig):
    tau = opt.tau
    tau_decay = 1
    epsilon = opt.epsilon
    L = max(1,math.floor(tau/epsilon))
    W = math.ceil(L//2)
    epochs = 40
    sampling = 20

    orig_pic = []
    for j in range(len(x_orig)):
        orig_pic.append(inverse_data_transform(config, x_orig[j]))
    psnr_list = []
    loss_list = []
    final_img_list = []
    sigma_y_list = []
    epsilon_list = []

    d = x.view(-1).numel()
    x = x.detach().requires_grad_()

    xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
    x_save = [inverse_data_transform(config, y) for y in xt.detach()]
    for j in range(len(x_save)):
        tvu.save_image(
            x_save[j], os.path.join(opt.image_folder, f"hmc_.png")
        )
        mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
        psnr = 10 * torch.log10(1 / mse)
        print('start', 'PSNR:', psnr.item())

    accepted = 0
    rejected = 0
    low_accept = 0

    epoch = 0

    while epoch < epochs + 2 * sampling:
        if epoch < epochs:
            sigma_y = opt.sigma_0 + 1.6 * (1 - epoch / epochs) ** 1
            #tau = 0.5 #sigma_y ** (2/3) * tau_decay  #(0.1 + (opt.tau-0.1) * (1 - epoch / epochs))
            #epsilon = 0.05 * tau  #(0.005 + (opt.epsilon-0.005) * (1 - epoch / epochs))
            L = max(1,math.floor(tau/epsilon))
        elif epoch >= epochs:
            sigma_y = opt.sigma_0
            #if tau > 0.1:
            #    tau = 0.1
            #    epsilon = 0.005

        # initialize momentum
        p = torch.randn_like(x, device=device) * math.sqrt(opt.m)
        xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
        loss = torch.sum((y_0 - H_funcs.H(xt))**2)
        loss_grad = torch.autograd.grad(loss, x, retain_graph=False)[0]

        H = (1/2) * torch.sum(x**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * loss.detach() + (1/2)* torch.sum(p * p, dim=(1, 2, 3)) * opt.m**(-1)
        H_list = [H.item()]
        x_list = [x.detach().clone()]

        for l in range(1, L+1):

            # update momentum
            p = p - (epsilon / 2) * (x.detach() + 1/(2 * sigma_y**2) * loss_grad)

            x = x + epsilon * opt.m**(-1) * p 
            x = x.detach().requires_grad_(True)

            xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
            loss = torch.sum((y_0 - H_funcs.H(xt))**2)
            loss_grad = torch.autograd.grad(loss, x, retain_graph=False)[0]

            p = p - (epsilon / 2) * (x.detach() + 1/(2 * sigma_y**2) * loss_grad)
            H = (1/2) * torch.sum(x**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * loss.detach() + (1/2)* torch.sum(p * p, dim=(1, 2, 3)) * opt.m**(-1)

            #if epoch >= 1:
            if l < W or l >= L-W+1:
                H_list.append(H.item())
                x_list.append(x.detach().clone())

        H_min = min(H_list)
        exp_H_list = [math.exp(-(H - H_min)) for H in H_list]
        exp_H_current = sum(exp_H_list[:W])
        exp_H_prop = sum(exp_H_list[-W:])

        H_current_min = min(H_list[:W])
        H_prop_min = min(H_list[-W:])
        probs_current = [math.exp(-(H - H_current_min)) for H in H_list[:W]]
        probs_prop = [math.exp(-(H - H_prop_min)) for H in H_list[-W:]]

        x_current = x_list[:W][torch.multinomial(torch.tensor(probs_current), 1).item()]
        x_prop = x_list[-W:][torch.multinomial(torch.tensor(probs_prop), 1).item()]

        if exp_H_current > 0:
            accept_prob = exp_H_prop / exp_H_current
        else:
            accept_prob = 1
        accept = torch.rand(1).item() < accept_prob
        print("Acceptance ratio:", accept_prob)
        if accept_prob < 0.6:
            low_accept += 1
            if low_accept > 1:
                #tau_decay *= 0.95
                tau = tau * 0.85
                print('                    Acceptance too low, annealing tau:', tau)
                epsilon = epsilon * 0.85
                low_accept = 0
        else:
            low_accept = 0
            
        if accept:
            
            x_accept = xt.detach().clone()
            if epoch >= epochs + sampling:
                final_img_list.append(x_accept[0])
            epoch += 1
            x = x_prop.detach().clone().requires_grad_(True)
            x_save = [inverse_data_transform(config, y) for y in xt.detach()]
            for j in range(len(x_save)):
                #tvu.save_image(
                #    x_save[j], os.path.join(opt.image_folder, f"hmc_{epoch}.png")
                #)
                mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                psnr_list.append(psnr.item())
                sigma_y_list.append(sigma_y)
                epsilon_list.append(epsilon)
                print('epoch', epoch, 'PSNR:', psnr.item(), 'sigma_y:', sigma_y, 'tau:', tau, 'epsilon', epsilon)
        else:
            x = x_current.detach().clone().requires_grad_(True)


    skip = 0
    # plot the PSNR, loss in the same graph
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(sigma_y_list[skip:], 'g-', label='sigma_y')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('sigma_y', color='g')

    ax2 = ax1.twinx()
    ax2.plot(epsilon_list[skip:], 'b-', label='epsilon')
    ax2.set_ylabel('epsilon', color='b')

    # save
    ax1.set_title('HMC Sampling: sigma_y, epsilon over Epochs')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)


    # Optional: Combine legends
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.savefig(os.path.join(opt.image_folder, 'hmc_window.png'), bbox_inches='tight')

    return torch.stack(final_img_list) #x_accept

def iterative_sampling(xt, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=False):

    x0_t_last = None

    for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), disable=tqdm_disable):
        t = (torch.ones(n) * i).to(xt.device)
        next_t = (torch.ones(n) * j).to(xt.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        if opt.algo == 'reddiff':
            x0_t, add_up = algo.cal_x0(xt, x0_t_last, t, at, at_next, y_0, opt.noise)
        else:
            x0_t, add_up = algo.cal_x0(xt, t, at, at_next, y_0, opt.noise)

        x0_t_last = x0_t
        xt_next = algo.map_back(x0_t, y_0, add_up, at_next, at)
        xt = xt_next
    return xt


def gaussian_log_prob(x_from, x_to, grad_from, tau):
    mean = x_from - 0.5 * tau**2 * grad_from
    diff = x_to - mean
    return -0.5 * (diff.pow(2).sum() / tau**2)

def sphere_update(x, grad, r, alpha, preserve_norm=True):
    """
    Performs a geodesic gradient descent step on the sphere.
    
    Args:
        x (torch.Tensor): Current point on the sphere
        grad (torch.Tensor): Gradient at x, same shape as x.
        alpha (float): Step size (arc length in radians).
    
    Returns:
        torch.Tensor: Updated point on the sphere, same shape as x.
    """
    with torch.no_grad():
        # Flatten to vector
        x_flat = x.view(-1)
        g_flat = grad.view(-1)

        # Project gradient onto tangent space of the sphere
        inner = torch.dot(g_flat, x_flat) / (x_flat.norm() ** 2)
        g_tan = g_flat - inner * x_flat

        # If tangent vector is (numerically) zero, no update
        if g_tan.norm() < 1e-12:
            return x
        
        # Geodesic update on the sphere of radius ||x||
        u = g_tan / g_tan.norm()  # unit tangent direction

        if preserve_norm:
            x_new = math.cos(alpha / r) * x_flat - math.sin(alpha / r) * r * u
        else:
            x_new = x_flat - math.sin(alpha) * u

    return x_new.view_as(x)

def cosine_similarity(x, y, eps=1e-8):
    dot = torch.sum(x * y)
    norm_x = torch.norm(x)
    norm_y = torch.norm(y)
    return dot / (norm_x * norm_y + eps)

import torch

def expected_clipped_sse(s, sigma_y, a=-1.0, b=1.0):
    """
    Compute E[(clip(s + n, a, b) - s)^2] where n ~ N(0, sigma_y^2), in PyTorch.

    Args:
        s       : tensor of shape (...) predicted measurements
        sigma_y : scalar float (can be torch.tensor) noise stddev > 0
        a, b    : clipping bounds (floats)

    Returns:
        Tensor same shape as s: per-pixel expected squared error
    """
    # Standardize bounds
    t_a = (a - s) / sigma_y
    t_b = (b - s) / sigma_y

    # Normal pdf and cdf in torch
    phi_a = torch.exp(-0.5 * t_a**2) / torch.sqrt(torch.tensor(2.0 * torch.pi, device=s.device))
    phi_b = torch.exp(-0.5 * t_b**2) / torch.sqrt(torch.tensor(2.0 * torch.pi, device=s.device))

    Phi_a = 0.5 * (1.0 + torch.erf(t_a / torch.sqrt(torch.tensor(2.0, device=s.device))))
    Phi_b = 0.5 * (1.0 + torch.erf(t_b / torch.sqrt(torch.tensor(2.0, device=s.device))))

    # Left tail: clipped to a
    left = (a - s)**2 * Phi_a

    # Middle: unclipped region
    middle = (sigma_y**2) * (Phi_b - Phi_a - t_b * phi_b + t_a * phi_a)

    # Right tail: clipped to b
    right = (b - s)**2 * (1.0 - Phi_b)

    sum_squared_error = (left + middle + right).sum()

    return sum_squared_error


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=5678, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset",
        default="celeba"
    )

    parser.add_argument(
        "--default_lr", action="store_true", help="Using the best step sizes to reproduce the results in the paper"
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="?",
        help="Step-Size",
        default=1.0
    )
    parser.add_argument(
        "--N", type=int, default=1, help="N repeats"
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation Type"
    )
    parser.add_argument(
        "--sigma_0", type=float, required=True, help="True Measurement Noise"
    )
    parser.add_argument(
        "--L", type=float, default=20, help="L for HMC"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.05, help="Epsilon for HMC"
    )
    parser.add_argument(
        "--noise", type=str, default="ddpm", help="Type of Noise"
    )
    parser.add_argument(
        "--m", type=float, default=1.0, help="Mass for HMC"
    )

    parser.add_argument(
        "--num_timesteps",
        type=int,
        nargs="?",
        help="Maximum timestep for beta schedule",
        default=1000
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="?",
        help="Number of timesteps for actual sampling",
        default=10
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "--algo",
        type=str,
        nargs="?",
        help="Algorithm to use for sampling",
        default='resample'
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="exp/samples/ffhq/00000",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--sigma_refine",
        type=float,
        default=0.75,
        help="Sigma for refinement step in NHMC"
    )
    parser.add_argument(
        "--lam_refine",
        type=float,
        default=0.1,
        help="Lambda for refinement step in NHMC"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Whether to tune the hyperparameters for NHMC"
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        default=False,
        help="Whether to perform a refinement step after sampling"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to print detailed information during sampling"
    )
    return parser

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

if __name__ == "__main__":
    # Load configurations
    parser = get_parser()
    device = torch.device("cuda")
    opt, unknown = parser.parse_known_args()
    #config = load_yaml('configs/config_{}.yaml')
    opt.config = 'configs/config_{}.yml'.format(opt.dataset)
    with open(opt.config, "r") as f:
        config = yaml.safe_load(f)
    model_config = config['model']
    config = dict2namespace(config)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    if opt.tune:
        opt.subset_start = 1
        opt.subset_end = 4
        min_lpips = float('inf')
        for _ in range(100):
            opt.sigma_refine = np.random.uniform(0.2, 2.0)
            opt.lam_refine = np.random.uniform(0.01, 0.5)
            avg_psnr, avg_ssim, avg_lpips = sample_image(opt=opt, config=config, model_config=model_config, device=device)
            print(f"Sigma: {opt.sigma_refine}, Lambda: {opt.lam_refine}, PSNR: {avg_psnr}, SSIM: {avg_ssim}, LPIPS: {avg_lpips}")

            if avg_lpips < min_lpips:
                min_lpips = avg_lpips
                best_sigma = opt.sigma_refine
                best_lam = opt.lam_refine
                print(f"             New best: Sigma: {best_sigma}, Lambda: {best_lam}, LPIPS: {min_lpips}")
    else:
        sample_image(opt=opt, config=config, model_config=model_config, device=device)