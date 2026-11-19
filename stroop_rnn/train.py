"""Training with a controlled learning history.

Phase 1 (pre-training)  k epochs on task A alone.
Phase 2 (joint)         a fixed number of epochs on both tasks, with rho task-A trials per task-B trial.

All training uses back-to-back sequences of six trials, so activity carries over between trials from
the first gradient step on. In the input-norm control condition the two stimulus input vectors are
rescaled to a common norm after every update, in both phases.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import torch

from .behavior import responses
from .config import Condition, TaskConfig, TrainConfig
from .model import BatchedRNN, NoiseSource
from .task import TASK_A, TASK_B, make_sequences


def equalise_input_norms(model: BatchedRNN) -> None:
    """Rescale w_in,A and w_in,B of every network to their mean norm."""
    with torch.no_grad():
        norms = model.w_in.norm(dim=1)                       # [S, 2]
        target = norms.mean(dim=1, keepdim=True)
        model.w_in.mul_((target / norms).unsqueeze(1))


def clip_per_network(model: BatchedRNN, max_norm: float) -> torch.Tensor:
    """Clip the gradient norm of each network separately; returns the norms before clipping. [S]"""
    params = [p for p in model.parameters() if p.grad is not None]
    sq = sum(p.grad.flatten(1).pow(2).sum(1) for p in params)
    norm = sq.sqrt()
    factor = (max_norm / norm.clamp(min=1e-12)).clamp(max=1.0)
    for p in params:
        p.grad.mul_(factor.view(-1, *([1] * (p.grad.dim() - 1))))
    return norm


def make_optimizer(model: BatchedRNN, cond: Condition, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cond.optimizer == "adam":
        lr = cfg.lr_rank1 if cond.variant == "rank1" else cfg.lr_adam
        return torch.optim.Adam(model.parameters(), lr=lr)
    if cond.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr_sgd, momentum=cfg.momentum)
    raise ValueError(f"unknown optimizer {cond.optimizer!r}")


def train_condition(
    cond: Condition,
    seeds: list[int],
    task_cfg: TaskConfig = TaskConfig(),
    train_cfg: TrainConfig = TrainConfig(),
    data_seed: int = 0,
    log=None,
    on_log=None,
) -> dict:
    """Train len(seeds) networks under one learning history; returns a checkpoint dictionary.

    log     optional function that receives one progress line every `train_cfg.log_every` steps.
    on_log  optional function (step, total_steps, make_checkpoint) called at the same moments; it can
            save intermediate checkpoints or raise an exception to abort the run.
    """
    model = BatchedRNN(seeds, variant=cond.variant, tau=task_cfg.tau, dt=task_cfg.dt)
    if cond.norm_control:
        equalise_input_norms(model)
    opt = make_optimizer(model, cond, train_cfg)
    gen = torch.Generator().manual_seed(data_seed)
    noise = NoiseSource(data_seed + 1)

    S = model.n_networks
    pre_steps = cond.k * train_cfg.steps_per_epoch
    total_steps = pre_steps + train_cfg.joint_epochs * train_cfg.steps_per_epoch
    p_a_joint = cond.rho / (1.0 + cond.rho)
    horizon = task_cfg.steps(task_cfg.train_horizon_ms)

    hist = {k: [] for k in ("step", "loss", "acc_A", "acc_B", "grad_norm", "norm_in_A", "norm_in_B")}
    acc = {"loss": torch.zeros(S), "gn": torch.zeros(S), "n": 0,
           "hit": torch.zeros(2, S), "cnt": torch.zeros(2, S)}
    t0 = time.time()

    def make_checkpoint() -> dict:
        return {
            "condition": cond.as_dict(),
            "model_config": model.config(),
            "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "task_config": asdict(task_cfg),
            "train_config": asdict(train_cfg),
            "data_seed": data_seed,
            "history": {k: np.asarray(v) for k, v in hist.items()},
            "elapsed_s": time.time() - t0,
        }

    for step in range(total_steps):
        p_a = 1.0 if step < pre_steps else p_a_joint
        seq = make_sequences(task_cfg, (S, train_cfg.batch_seqs), gen, p_a=p_a, horizon=horizon)
        loss, y = model.bptt(seq.x, seq.target, sigma_rec=task_cfg.sigma_rec, noise=noise)
        grad_norm = clip_per_network(model, train_cfg.clip)
        opt.step()
        if cond.norm_control:
            equalise_input_norms(model)

        # running statistics for the log
        acc["loss"] += loss
        acc["gn"] += grad_norm
        acc["n"] += 1
        fc = responses(y, seq, task_cfg, thresholds=())["final_correct"].float()
        for task in (TASK_A, TASK_B):
            m = (seq.valid & (seq.context == task)).float()
            acc["hit"][task] += (fc * m).sum(dim=(1, 2))
            acc["cnt"][task] += m.sum(dim=(1, 2))

        if (step + 1) % train_cfg.log_every == 0 or step + 1 == total_steps:
            rate = acc["hit"] / acc["cnt"].clamp(min=1)
            rate[acc["cnt"] == 0] = float("nan")
            norms = model.w_in.detach().norm(dim=1)
            hist["step"].append(step + 1)
            hist["loss"].append((acc["loss"] / acc["n"]).numpy().copy())
            hist["grad_norm"].append((acc["gn"] / acc["n"]).numpy().copy())
            hist["acc_A"].append(rate[TASK_A].numpy().copy())
            hist["acc_B"].append(rate[TASK_B].numpy().copy())
            hist["norm_in_A"].append(norms[:, 0].numpy().copy())
            hist["norm_in_B"].append(norms[:, 1].numpy().copy())
            if log is not None:
                phase = "pre" if step < pre_steps else "joint"
                log(
                    f"{cond.name:28s} step {step + 1:5d}/{total_steps} [{phase:5s}] "
                    f"loss {hist['loss'][-1].mean():.4f}  acc A {np.nanmean(hist['acc_A'][-1]):.3f} "
                    f"B {np.nanmean(hist['acc_B'][-1]):.3f}  |g| {hist['grad_norm'][-1].mean():.2f}  "
                    f"{time.time() - t0:6.0f}s"
                )
            acc = {"loss": torch.zeros(S), "gn": torch.zeros(S), "n": 0,
                   "hit": torch.zeros(2, S), "cnt": torch.zeros(2, S)}
            if on_log is not None:
                on_log(step + 1, total_steps, make_checkpoint)

    return make_checkpoint()
