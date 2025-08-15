import torch
from algos.base_algo import Base_Algo
from torch.nn import Parameter
from optim.sf_adamw import AdamWScheduleFree
import tqdm
import time
from algos.daps.mcmc import MCMCSampler
#from algos.daps.ode_solver import EDMScheduler, DiffusionPFODE

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

class DAPS(Base_Algo):
    def __init__(self, daps_cfg, model, H_funcs, sigma_0, timesteps, mcmc_num_steps, lr, lr_min_ratio, cls_fn=None, betas = []):
        super().__init__(model, H_funcs, sigma_0, cls_fn)
        self.daps_cfg = daps_cfg
        self.order = daps_cfg.diffusion_scheduler_config.num_steps
        self.timesteps = timesteps
        self.betas = torch.tensor(betas, dtype=torch.float32).cuda()
        self.sigma_0 = sigma_0
        self.mcmc_num_steps = mcmc_num_steps
        self.lr = lr
        self.lr_min_ratio = lr_min_ratio

    @ torch.no_grad()
    def ode(self, xt, t, classes=None):
        n = xt.shape[0]
        skip = t // (self.order - 1)
        #print(self.order, skip)
        if skip > 0:
            seq = range(0, t, skip)
        else:
            seq = [0]
        # print(list(seq))
        # print(seq)
        seq = list(seq)[1:] + [t]
        seq_next = [-1] + list(seq[:-1])
        b = self.betas
        for i, j in zip(reversed(seq), reversed(seq_next)):
            # steps.append(i)
            t = (torch.ones(n) * i).to(xt.device)
            next_t = (torch.ones(n) * j).to(xt.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            if self.cls_fn == None:
                et = self.model(xt, t)
            else:
                et = self.model(xt, t, self.classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * self.cls_fn(x,t,self.classes)
            if et.size(1) == 6:
                et = et[:, :3]
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = x0_t.clip(-1, 1)
            xt_next = at_next.sqrt() * x0_t + (1-at_next).sqrt() * et
            xt = xt_next
            # print(xt.norm())
        return xt

    @ torch.no_grad() 
    def cal_x0(self, xt, t, at, at_next, y_0, noise='ddpm', T=1000, classes=None):
        """ with torch.no_grad():
            diffusion_scheduler = EDMScheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
            sampler = DiffusionPFODE(model, diffusion_scheduler, device=xt.device, solver='euler')
            x0hat = sampler.sample(xt) """

        x0hat = self.ode(xt, int(t[0]), classes=classes)

        mcmc_sampler = MCMCSampler(self.mcmc_num_steps, self.lr, lr_min_ratio=self.lr_min_ratio, tau=self.sigma_0)

        sigma = (1-at[0,0,0,0]).sqrt()
        x0y = mcmc_sampler.sample(xt, x0hat, self.H_funcs.H, y_0, sigma, 1-t.item()/T)

        add_up = (1-at_next).sqrt() * torch.randn_like(x0hat)

        return x0y, add_up
    
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        xt_next = at_next.sqrt() * x0_t + add_up
        return xt_next