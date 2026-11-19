"""Continuous-time rate RNN of Eq. (1), simulated for many independent networks at once.

    tau dh/dt = -h + W_rec phi(h) + W_in u(t) + W_ctx c(t) + eta(t),      y(t) = W_out phi(h(t))

with phi(x) = log(1 + exp(x)) (softplus), integrated with the Euler method:

    h[t+1] = (1 - a) h[t] + a (W_rec phi(h[t]) + W_in u[t] + W_ctx c[t]) + sqrt(2a) sigma_rec xi[t],   a = dt / tau
    y[t]   = W_out phi(h[t+1])

Every parameter carries a leading "network" dimension S, so S networks (e.g. the 20 seeds of one
training condition) are trained and analysed in a single batched computation. The networks never
interact: the loss is a sum over networks and both optimisers used here act element-wise.

Three parametrisations of the recurrent weights are supported:
    "full"   W_rec = W                      (unconstrained)
    "dale"   W_rec = |W| D, D = diag(+-1)   (Dale's law, 80% excitatory / 20% inhibitory; Song et al., 2016)
    "rank1"  W_rec = m n^T / N              (rank-one network used for validation hypothesis V)

Time-series tensors are laid out time-first, [T, S, B, C], so that each time step is one contiguous block.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

VARIANTS = ("full", "dale", "rank1")

# exp() is evaluated on min(h, _CLAMP); above the clamp softplus(h) = h to float32 precision.
_CLAMP = 20.0


def softplus(h: Tensor) -> Tensor:
    return torch.nn.functional.softplus(h, threshold=_CLAMP)


class NoiseSource:
    """Independent standard-normal samples (float32) from a seeded NumPy generator.

    NumPy's float32 ziggurat sampler is roughly three times faster than torch.randn on CPU, and the
    private noise eta(t) is by far the largest consumer of random numbers in this project.
    """

    def __init__(self, seed: int):
        self.rng = np.random.Generator(np.random.SFC64(seed))

    def normal(self, *shape: int) -> Tensor:
        return torch.from_numpy(self.rng.standard_normal(shape, dtype=np.float32))


class BatchedRNN(nn.Module):
    """S independent rate networks of N units each."""

    def __init__(
        self,
        seeds: Sequence[int],
        n_units: int = 128,
        variant: str = "full",
        tau: float = 100.0,
        dt: float = 10.0,
        g: float = 1.0,
        frac_exc: float = 0.8,
        w_in_std: float = 0.5,
        w_out_std: float | None = None,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.seeds = [int(s) for s in seeds]
        self.n_networks = len(self.seeds)
        self.n_units = n_units
        self.variant = variant
        self.tau = float(tau)
        self.dt = float(dt)
        self.g = float(g)
        self.frac_exc = float(frac_exc)
        S, N = self.n_networks, n_units
        if w_out_std is None:
            w_out_std = 0.5 / math.sqrt(N)

        n_exc = int(round(frac_exc * N))
        sign = torch.ones(N)
        sign[n_exc:] = -1.0
        self.register_buffer("dale_sign", sign)

        w_in = torch.empty(S, N, 2)
        w_ctx = torch.empty(S, N, 2)
        w_out = torch.empty(S, 2, N)
        if variant == "rank1":
            m = torch.empty(S, N)
            n = torch.empty(S, N)
        else:
            w_raw = torch.empty(S, N, N)

        # Each network is initialised from its own seed, independently of which other
        # networks happen to share the batch.
        for s, seed in enumerate(self.seeds):
            gen = torch.Generator().manual_seed(seed)
            if variant == "full":
                w_raw[s] = torch.randn(N, N, generator=gen) * (g / math.sqrt(N))
            elif variant == "dale":
                raw = torch.randn(N, N, generator=gen).abs()
                raw[:, n_exc:] *= n_exc / (N - n_exc)  # inhibition balances excitation on average
                radius = torch.linalg.eigvals(raw * sign).abs().max()
                w_raw[s] = raw * (g / radius)
            else:
                m[s] = torch.randn(N, generator=gen)
                n[s] = torch.randn(N, generator=gen)
            w_in[s] = torch.randn(N, 2, generator=gen) * w_in_std
            w_ctx[s] = torch.randn(N, 2, generator=gen) * w_in_std
            w_out[s] = torch.randn(2, N, generator=gen) * w_out_std

        if variant == "rank1":
            self.m = nn.Parameter(m)
            self.n = nn.Parameter(n)
        else:
            self.w_raw = nn.Parameter(w_raw)
        self.w_in = nn.Parameter(w_in)
        self.w_ctx = nn.Parameter(w_ctx)
        self.w_out = nn.Parameter(w_out)

    # ------------------------------------------------------------------ weights
    @property
    def alpha(self) -> float:
        return self.dt / self.tau

    def w_rec(self) -> Tensor:
        """Effective recurrent weights [S, N, N]; entry [s, i, j] is the weight from unit j to unit i."""
        if self.variant == "full":
            return self.w_raw
        if self.variant == "dale":
            return self.w_raw.abs() * self.dale_sign
        return self.m.unsqueeze(2) * self.n.unsqueeze(1) / self.n_units

    def w_x(self) -> Tensor:
        """Input weights [S, N, 4] for the channels (u_A, u_B, c_A, c_B)."""
        return torch.cat([self.w_in, self.w_ctx], dim=2)

    def config(self) -> dict:
        return dict(
            seeds=self.seeds, n_units=self.n_units, variant=self.variant, tau=self.tau, dt=self.dt,
            g=self.g, frac_exc=self.frac_exc,
        )

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "BatchedRNN":
        model = cls(**ckpt["model_config"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    # --------------------------------------------------------------- simulation
    @torch.no_grad()
    def simulate(
        self,
        x: Tensor,
        sigma_rec: float = 0.0,
        noise: NoiseSource | None = None,
        h0: Tensor | None = None,
        return_h: bool = False,
        interventions: dict[int, tuple[Tensor, Tensor]] | None = None,
    ):
        """Run the networks on inputs x [T, S, B, 4]; returns outputs y [T, S, B, 2].

        h0            initial state [S, B, N] (default: zeros).
        return_h      also return the states h[t+1], shape [T, S, B, N].
        interventions {t: (mask [S, B] bool, h_new [S, B, N])}: before input step t is consumed, the state
                      of the masked sequences is overwritten with h_new (used for the E5 state resets).
        """
        T, S, B, _ = x.shape
        N, a = self.n_units, self.alpha
        w = self.w_rec()
        wat = (a * w).transpose(1, 2).contiguous()
        wxt = (a * self.w_x()).transpose(1, 2).contiguous()
        wot = self.w_out.transpose(1, 2).contiguous()
        h = x.new_zeros(S, B, N) if h0 is None else h0.clone()
        r = softplus(h)
        y = x.new_empty(T, S, B, 2)
        hs = x.new_empty(T, S, B, N) if return_h else None
        scale = math.sqrt(2.0 * a) * sigma_rec
        if scale > 0 and noise is None:
            raise ValueError("sigma_rec > 0 requires a NoiseSource")
        for t in range(T):
            if interventions is not None and t in interventions:
                mask, h_new = interventions[t]
                h = torch.where(mask.unsqueeze(-1), h_new, h)
                r = softplus(h)
            pre = torch.bmm(x[t], wxt)
            if scale > 0:
                pre.add_(noise.normal(S, B, N), alpha=scale)
            pre.baddbmm_(r, wat)
            h.mul_(1.0 - a).add_(pre)
            r = softplus(h)
            torch.bmm(r, wot, out=y[t])
            if return_h:
                hs[t] = h
        return (y, hs) if return_h else y

    # ----------------------------------------------------------------- training
    def bptt(self, x: Tensor, target: Tensor, sigma_rec: float = 0.0, noise: NoiseSource | Tensor | None = None):
        """Mean-squared-error loss of every network and its exact gradient (backpropagation through time).

        x [T, S, B, 4], target [T, S, B, 2]. Returns (loss [S], y [T, S, B, 2]) and writes the gradient of
        sum_s loss[s] to the .grad fields of the parameters.

        The backward pass is written out by hand instead of going through autograd. This is the same
        computation (tests/test_model.py checks the gradients against autograd), but it avoids building a
        graph with thousands of nodes per step and is several times faster on CPU:

            delta[t] = dL/dh[t] = phi'(h[t]) * (W_out^T gy[t-1] + a W_rec^T delta[t+1]) + (1 - a) delta[t+1]
            dL/dW_rec = a sum_t delta[t+1] r[t]^T,   dL/dW_x = a sum_t delta[t+1] x[t]^T,   dL/dW_out = sum_t gy[t] r[t+1]^T

        `noise` is either a NoiseSource (fresh private noise is drawn) or an explicit tensor [T, S, B, N]
        of standard-normal samples (used by the gradient test).
        """
        T, S, B, _ = x.shape
        N, a = self.n_units, self.alpha
        w = self.w_rec()
        scale = math.sqrt(2.0 * a) * sigma_rec
        with torch.no_grad():
            wa = (a * w).contiguous()
            wat = wa.transpose(1, 2).contiguous()
            wxt = (a * self.w_x()).transpose(1, 2).contiguous()
            w_out = self.w_out.detach()

            if scale > 0:
                if noise is None:
                    raise ValueError("sigma_rec > 0 requires noise")
                xi = noise if isinstance(noise, Tensor) else noise.normal(T, S, B, N)

            # ---- forward pass, storing rates and gains
            rates = x.new_empty(T + 1, S, B, N)   # rates[t] = phi(h[t])
            gains = x.new_empty(T, S, B, N)       # gains[t] = phi'(h[t+1])
            h = x.new_zeros(S, B, N)
            rates[0] = softplus(h)
            e = x.new_empty(S, B, N)
            p = x.new_empty(S, B, N)
            for t in range(T):
                pre = torch.bmm(x[t], wxt)
                if scale > 0:
                    pre.add_(xi[t], alpha=scale)
                pre.baddbmm_(rates[t], wat)
                h.mul_(1.0 - a).add_(pre)
                torch.clamp(h, max=_CLAMP, out=e).exp_()      # e = exp(min(h, clamp))
                torch.add(e, 1.0, out=p)                       # p = 1 + e
                torch.div(e, p, out=gains[t])                  # phi'(h) = e / (1 + e)
                torch.log(p, out=rates[t + 1])                 # phi(h)  = log(1 + e)
                if float(h.max()) > _CLAMP:                    # exact continuation above the clamp
                    rates[t + 1].add_((h - _CLAMP).clamp_(min=0.0))

            # ---- readout and loss
            rp = rates.permute(1, 0, 2, 3).contiguous()        # [S, T+1, B, N]
            r_prev = rp[:, :T].reshape(S, T * B, N)            # r[0..T-1]
            r_post = rp[:, 1:].reshape(S, T * B, N)            # r[1..T]
            yp = torch.bmm(r_post, w_out.transpose(1, 2))      # [S, T*B, 2]
            err = yp - target.permute(1, 0, 2, 3).reshape(S, T * B, 2)
            loss = err.pow(2).mean(dim=(1, 2))
            gy = err * (2.0 / (T * B * 2))                     # dL/dy
            d_w_out = torch.bmm(gy.transpose(1, 2), r_post)
            gy_t = gy.view(S, T, B, 2).permute(1, 0, 2, 3).contiguous()   # [T, S, B, 2]

            # ---- backward pass
            deltas = x.new_empty(T, S, B, N)                   # deltas[t] = dL/dh[t+1]
            delta = x.new_zeros(S, B, N)
            tmp = x.new_empty(S, B, N)
            for t in range(T - 1, -1, -1):
                torch.bmm(gy_t[t], w_out, out=tmp)
                tmp.baddbmm_(delta, wa)
                tmp.mul_(gains[t])
                torch.add(tmp, delta, alpha=1.0 - a, out=deltas[t])
                delta = deltas[t]
            dp = deltas.permute(1, 0, 2, 3).reshape(S, T * B, N).transpose(1, 2)   # [S, N, T*B]
            d_w_rec = torch.bmm(dp, r_prev).mul_(a)
            d_w_x = torch.bmm(dp, x.permute(1, 0, 2, 3).reshape(S, T * B, 4)).mul_(a)
            y = yp.view(S, T, B, 2).permute(1, 0, 2, 3)

        self.w_out.grad = d_w_out
        self.w_in.grad = d_w_x[:, :, :2].contiguous()
        self.w_ctx.grad = d_w_x[:, :, 2:].contiguous()
        if self.variant == "full":
            self.w_raw.grad = d_w_rec
        else:
            # chain rule through the parametrisation of W_rec (|W| D or m n^T / N)
            for prm in (self.w_raw,) if self.variant == "dale" else (self.m, self.n):
                prm.grad = None
            w.backward(d_w_rec)
        return loss, y

    def forward_autograd(self, x: Tensor, noise: Tensor | None = None, sigma_rec: float = 0.0) -> Tensor:
        """Reference implementation of the dynamics with autograd enabled (used only in tests)."""
        T, S, B, _ = x.shape
        a = self.alpha
        w = self.w_rec()
        wx = self.w_x()
        scale = math.sqrt(2.0 * a) * sigma_rec
        h = torch.zeros(S, B, self.n_units, dtype=x.dtype)
        ys = []
        for t in range(T):
            r = softplus(h)
            drive = torch.einsum("sij,sbj->sbi", w, r) + torch.einsum("sic,sbc->sbi", wx, x[t])
            h = (1.0 - a) * h + a * drive
            if scale > 0:
                h = h + scale * noise[t]
            ys.append(torch.einsum("soi,sbi->sbo", self.w_out, softplus(h)))
        return torch.stack(ys, dim=0)
