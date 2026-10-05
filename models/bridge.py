"""Brownian-bridge diffusion process used by TDBM.

The forward process is a discrete-time Brownian bridge that interpolates between
a contrast-free frame ``x0`` and a contrast-filled frame ``y``::

    q(x_t | x0, y) = N(x_t; (1 - m_t) x0 + m_t y, sigma_t^2 I)

with ``m_t = t / T`` and ``sigma_t^2 = eta * (m_t - m_t^2)``, so that the process
starts at ``x0`` (t = 0), ends at ``y`` (t = T) and has maximal variance at the
midpoint.  The reverse process is learned by ``denoise_fn`` (an attention U-Net),
which is additionally conditioned on a temporal window of neighbouring frames.

The bridge formulation follows BBDM (Li et al., CVPR 2023,
https://github.com/xuekt98/BBDM); TDBM adds the temporal conditioning and the
vessel-aware contrastive objective on top of it.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from .utils import default, extract


class BrownianBridgeModel(nn.Module):
    """Brownian-bridge diffusion model wrapping a conditional denoising U-Net.

    Args:
        unet: Denoising network. Called as ``unet(x_t, timesteps=t, context=c)``.
        num_timesteps: Length ``T`` of the bridge (training schedule).
        sample_step: Number of reverse steps ``K`` used at inference time when
            ``skip_sample`` is enabled (DDIM-style acceleration). ``K = 20`` is the
            default used throughout the paper; ``K = 50`` gives the best quality.
        image_size: Spatial resolution the network expects.
        mt_type: Schedule for ``m_t``; one of ``{"linear", "sin"}``.
        eta: Scales the stochasticity of the reverse transitions.
        loss_type: Reconstruction loss; one of ``{"l1", "l2"}``.
        objective: Prediction target; one of ``{"grad", "noise", "ysubx"}``.
        skip_sample: If True, sample with ``sample_step`` strided steps instead of
            all ``num_timesteps`` steps.
        sample_type: Spacing of the strided steps; one of ``{"linear", "cosine"}``.
        condition_key: ``"cond"`` to condition on the temporal context, ``"nocond"``
            to disable conditioning entirely.
    """

    def __init__(
        self,
        unet,
        num_timesteps: int = 1000,
        sample_step: int = 20,
        image_size: int = 256,
        mt_type: str = "linear",
        max_var: float = 1.0,
        eta: float = 1.0,
        loss_type: str = "l2",
        objective: str = "grad",
        skip_sample: bool = True,
        sample_type: str = "linear",
        condition_key: str = "cond",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mt_type = mt_type
        self.max_var = max_var
        self.eta = eta
        self.skip_sample = skip_sample
        self.sample_type = sample_type
        self.sample_step = sample_step
        self.steps = None
        self.register_schedule()

        self.loss_type = loss_type
        self.objective = objective

        self.image_size = image_size
        self.condition_key = condition_key

        self.denoise_fn = unet

    # ------------------------------------------------------------------ setup

    def register_schedule(self):
        """Pre-compute the bridge coefficients and the reverse step schedule."""
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError(f"Unknown mt_type: {self.mt_type}")
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2.0 * (m_t - m_t**2) * self.max_var
        variance_tminus = np.append(0.0, variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1.0 - m_t) / (1.0 - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("m_t", to_torch(m_t))
        self.register_buffer("m_tminus", to_torch(m_tminus))
        self.register_buffer("variance_t", to_torch(variance_t))
        self.register_buffer("variance_tminus", to_torch(variance_tminus))
        self.register_buffer("variance_t_tminus", to_torch(variance_t_tminus))
        self.register_buffer("posterior_variance_t", to_torch(posterior_variance_t))

        self.steps = self._build_sampling_steps()

    def _build_sampling_steps(self):
        """Return the (descending) list of timesteps visited by the reverse loop."""
        if not self.skip_sample:
            return torch.arange(self.num_timesteps - 1, -1, -1)

        if self.sample_type == "linear":
            if self.sample_step < 1:
                return torch.Tensor([1, 0]).long()
            midsteps = torch.arange(
                self.num_timesteps - 1,
                1,
                step=-((self.num_timesteps - 1) / (self.sample_step - 2)),
            ).long()
            return torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
        if self.sample_type == "cosine":
            steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
            steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.0) / 2.0 * self.num_timesteps
            return torch.from_numpy(steps)
        raise NotImplementedError(f"Unknown sample_type: {self.sample_type}")

    def set_sample_step(self, sample_step: int):
        """Change the number of reverse steps ``K`` (used for the acceleration study)."""
        self.sample_step = sample_step
        self.skip_sample = True
        self.steps = self._build_sampling_steps()

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    # -------------------------------------------------------------- training

    def forward(self, x, y, context=None):
        """Sample a random timestep and return the bridge loss for ``(x, y)``."""
        context = None if self.condition_key == "nocond" else default(context, y)
        b, c, h, w = x.shape
        assert h == self.image_size and w == self.image_size, (
            f"height and width of image must be {self.image_size}, got {h}x{w}"
        )
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        return self.p_losses(x, y, t, context=context)

    def p_losses(self, x0, y, t, noise=None, context=None):
        """Bridge reconstruction loss.

        Args:
            x0: Target (contrast-free) image, the ``t = 0`` end of the bridge.
            y: Source (contrast-filled) image, the ``t = T`` end of the bridge.
            t: Timestep indices, shape ``[B]``.
            noise: Optional standard Gaussian noise; drawn if omitted.
            context: Temporal conditioning window, shape ``[B, 2*delta+1, H, W]``.

        Returns:
            ``(loss, log_dict)`` where ``log_dict["x0_recon"]`` is the predicted ``x0``.
        """
        noise = default(noise, lambda: torch.randn_like(x0))
        context = None if self.condition_key == "nocond" else default(context, y)

        x_t, objective = self.q_sample(x0, y, t, noise)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        if self.loss_type == "l1":
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == "l2":
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError(f"Unknown loss_type: {self.loss_type}")

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        return recloss, {"loss": recloss, "x0_recon": x0_recon}

    def q_sample(self, x0, y, t, noise=None):
        """Draw ``x_t ~ q(x_t | x0, y)`` and the corresponding regression target."""
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == "grad":
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == "noise":
            objective = noise
        elif self.objective == "ysubx":
            objective = y - x0
        else:
            raise NotImplementedError(f"Unknown objective: {self.objective}")

        return (1.0 - m_t) * x0 + m_t * y + sigma_t * noise, objective

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        """Invert the training target to recover the predicted ``x0``."""
        if self.objective == "grad":
            return x_t - objective_recon
        if self.objective == "noise":
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            return (x_t - m_t * y - sigma_t * objective_recon) / (1.0 - m_t)
        if self.objective == "ysubx":
            return y - objective_recon
        raise NotImplementedError(f"Unknown objective: {self.objective}")

    # ------------------------------------------------------------- sampling

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        """Return the full forward trajectory (diagnostics only)."""
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc="q sampling loop"):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        """One reverse step from ``steps[i]`` to ``steps[i + 1]``."""
        t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
        if clip_denoised:
            x0_recon.clamp_(-1.0, 1.0)

        if self.steps[i] == 0:
            return x0_recon, x0_recon

        n_t = torch.full((x_t.shape[0],), self.steps[i + 1], device=x_t.device, dtype=torch.long)
        m_t = extract(self.m_t, t, x_t.shape)
        m_nt = extract(self.m_t, n_t, x_t.shape)
        var_t = extract(self.variance_t, t, x_t.shape)
        var_nt = extract(self.variance_t, n_t, x_t.shape)
        sigma2_t = (var_t - var_nt * (1.0 - m_t) ** 2 / (1.0 - m_nt) ** 2) * var_nt / var_t
        sigma_t = torch.sqrt(sigma2_t) * self.eta

        noise = torch.randn_like(x_t)
        x_tminus_mean = (
            (1.0 - m_nt) * x0_recon
            + m_nt * y
            + torch.sqrt((var_nt - sigma2_t) / var_t)
            * (x_t - (1.0 - m_t) * x0_recon - m_t * y)
        )
        return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False, progress=True):
        """Run the reverse bridge from ``y`` back to the contrast-free domain."""
        context = None if self.condition_key == "nocond" else default(context, y)

        iterator = range(len(self.steps))
        if progress:
            iterator = tqdm(iterator, desc="sampling loop time step", total=len(self.steps), leave=False)

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in iterator:
                img, x0_recon = self.p_sample(imgs[-1], y, context, i, clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs

        img = y
        for i in iterator:
            img, _ = self.p_sample(img, y, context, i, clip_denoised)
        return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False, progress=True):
        """Predict the contrast-free frame corresponding to ``y``."""
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step, progress)
