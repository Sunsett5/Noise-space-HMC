from torch.autograd import grad
from abc import ABC, abstractmethod
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import torch
import warnings

class Scheduler(ABC):
    """
    Abstract base class for diffusion scheduler.

    Schedulers manage time steps, noise scales (sigma), scaling factors, and coefficients 
    used in diffusion stochastic/ordinary differential equations (SDEs/ODEs).
    """

    def __init__(self, num_steps):
        self.num_steps = num_steps + 1 # include the initial step

    def discretize(self, time_steps):
        sigma_steps = self.get_sigma(time_steps[:-1])
        sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])
        self.sigma_steps = sigma_steps

    def tensorize(self, data):
        if isinstance(data, (int, float)):
            return torch.tensor(data).float()
        if isinstance(data, list):
            return torch.tensor(data).float()
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        if isinstance(data, torch.Tensor):
            return data.float()
        raise ValueError(f"Data type {type(data)} is not supported.") 

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Noise Scheduling & Scaling Function 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @abstractmethod
    def get_scaling(self, t):
        pass
    
    def get_sigma(self, t):
        pass
    
    def get_scaling_derivative(self, t):
        pass

    def get_sigma_derivative(self, t):
        pass

    def get_sigma_inv(self, sigma):
        pass
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Time & Sigma Range Function
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_t_min(self):
        pass

    def get_t_max(self):
        pass

    def get_discrete_time_steps(self, num_steps):
        pass

    def get_sigma_max(self):
        return self.get_sigma(self.get_t_max())

    def get_sigma_min(self):
        return self.get_sigma(self.get_t_min())
    
    def get_prior_sigma(self):
        # simga(t_max) * scaling(t_max)
        return self.get_sigma_max() * self.get_scaling(self.get_t_max())

    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # For Iterating Over the Discretized Scheduler
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __iter__(self):
        self.pbar = range(self.num_steps)
        self.pbar_iter = iter(self.pbar)
        return self

    def __next__(self):
        try:
            step = next(self.pbar_iter)
            time, scaling, sigma, scaling_factor, factor = self.time_steps[step], self.scaling_steps[step], \
                self.sigma_steps[step], self.scaling_factor_steps[step], self.factor_steps[step]
            return self.pbar, time, scaling, sigma, factor, scaling_factor
        except StopIteration:
            raise StopIteration

class EDMScheduler(Scheduler):
    """
        EDM (Elucidating the Design Space of Diffusion-Based Generative Models) Scheduler.
    """

    def __init__(self, num_steps, sigma_max=100, sigma_min=1e-2, timestep='poly-7'):
        super().__init__(num_steps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        p = int(timestep.split('-')[1])
        self.time_steps_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p

        # get time_steps
        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_sigma(self, t):
        # sigma(t) = t
        return self.tensorize(t)

    def get_scaling(self, t):
        # s(t) = 1
        return torch.ones_like(self.tensorize(t))

    def get_sigma_derivative(self, t):
        # sigma'(t) = 1
        return torch.ones_like(self.tensorize(t))

    def get_scaling_derivative(self, t):
        # s'(t) = 0
        return torch.zeros_like(self.tensorize(t))
    
    def get_sigma_inv(self, sigma):
        return self.tensorize(sigma)

    def get_t_min(self):
        return self.tensorize(self.sigma_min)
    
    def get_t_max(self):
        return self.tensorize(self.sigma_max)

    def get_discrete_time_steps(self, num_steps):
        steps = np.linspace(0, 1, num_steps)
        time_steps = np.array([self.time_steps_fn(s) for s in steps])
        return torch.from_numpy(time_steps)


class DiffusionPFODE:
    """
    Diffusion Probability Flow ODE (PF-ODE) for sampling and likelihood computation.

    Implements forward and reverse sampling based on diffusion models, using numerical ODE solvers.
    """
    def __init__(self, model, scheduler, device, solver='euler'):
        self.model = model
        self.scheduler = scheduler
        self.solver = solver
        self.device = device
    
    def derivative(self, xt, t):
        # refer to Eq. (4) in EDM paper (https://arxiv.org/abs/2206.00364)
        st = self.scheduler.get_scaling(t)
        dst = self.scheduler.get_scaling_derivative(t)
        sigma_t = self.scheduler.get_sigma(t)
        dsigma_t = self.scheduler.get_sigma_derivative(t)
        return dst / st * xt - st * dsigma_t * sigma_t * self.model.score(xt/st, sigma=sigma_t)

    def sample(self, xT, num_steps=None, return_traj=False, requires_grad=False):
        # reverse PF-ODE, from prior Gaussian to data
        if num_steps is None:
            num_steps = self.scheduler.num_steps
        
        shape = xT.shape
        def _derivative_wrapper(t, xt):
            xt = xt.view(*shape)
            deriv = self.derivative(xt, t)
            return deriv.flatten(1)
        
        time_steps = self.scheduler.get_discrete_time_steps(num_steps).to(xT.device)
        if requires_grad:
            xT.requires_grad_(True)
            x_ode_traj = odeint_adjoint(_derivative_wrapper, xT.flatten(1), time_steps, rtol=1e-3, atol=1e-3, method=self.solver, adjoint_params=(xT))
        else:
            x_ode_traj = odeint(_derivative_wrapper, xT.flatten(1), time_steps, rtol=1e-3, atol=1e-3, method=self.solver) # [num_steps, B, D]
        x_ode_traj = x_ode_traj.view(num_steps, *shape)
        
        if return_traj:
            return x_ode_traj
        else:
            return x_ode_traj[-1]

