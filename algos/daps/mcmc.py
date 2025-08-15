import torch
import numpy as np
import torch.nn as nn


class MCMCSampler(nn.Module):
    """
    Monte Carlo sampler class for diffusion processes.

    Supports Langevin dynamics, Hamiltonian Monte Carlo (HMC) and Metropolis-Hastings (MH) methods.

    Attributes:
        num_steps (int): Number of sampling steps.
        lr (float): Initial learning rate.
        tau (float): Standard deviation for data-fitting term.
        lr_min_ratio (float): Minimum learning rate ratio.
    """

    def __init__(self, num_steps, lr, lr_min_ratio=0.01, tau=0.01):
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio

    @ torch.enable_grad()
    def score_fn(self, x, x0hat, xt, operator, measurement, sigma):
        """
        Computes the conditional score function \nabla_x \log p(x_0 = x | x_t, y).

        Returns:
            Tuple containing:
                - Current score estimate.
                - Data-fitting loss.
        """
        x.requires_grad_(True) 
        data_fitting_error = measurement - operator(x)
        data_fitting_loss = torch.sum(data_fitting_error**2)
        data_fitting_grad = torch.autograd.grad(outputs=data_fitting_loss, inputs=x)[0]
        data_term = -data_fitting_grad / self.tau ** 2
        x.requires_grad_(False)
        xt_term = (xt - x) / sigma ** 2
        prior_term = (x0hat - xt).detach() / sigma ** 2
        return data_term + xt_term + prior_term, data_fitting_loss

    def mc_update(self, x, cur_score, lr, epsilon):
        """ Performs a single Monte Carlo update step (Langevin or HMC)."""
        x_new = x + lr * cur_score + np.sqrt(2 * lr) * epsilon
        return x_new


    def sample(self, xt, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        """
        Main method for performing MCMC sampling.

        Args:
            xt (torch.Tensor): Current noisy latent tensor.
            x0hat (torch.Tensor): Initial estimate of x0 from PF-ODE.
            operator (Operator): Measurement operator.
            measurement (torch.Tensor): Measurement data.
            sigma (float): Noise scale at current timestep.
            ratio (float): Ratio to adjust learning rate scheduling.
            record (bool): Whether to record trajectory.
            verbose (bool): Verbosity flag.

        Returns:
            torch.Tensor: Sampled latent tensor.
        """
      
        lr = self.get_lr(ratio)

        x = x0hat.clone().detach()
        pbar = range(self.num_steps)
        for _ in pbar:
            cur_score, _ = self.score_fn(x, x0hat, xt, operator, measurement, sigma)
            epsilon = torch.randn_like(x)

            x = self.mc_update(x, cur_score, lr, epsilon)

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x) 

        return x.detach()

    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr
