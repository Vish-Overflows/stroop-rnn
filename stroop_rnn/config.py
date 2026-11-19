"""All design constants of the study in one place.

Times are in milliseconds throughout. Task A is the "word" task (report feature A, the feature that is
overlearned in the extension), task B is the "colour" task (report feature B).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TaskConfig:
    dt: float = 10.0                          # Euler step
    tau: float = 100.0                        # unit time constant
    stim_ms: float = 500.0                    # stimulus duration
    hold_ms: tuple[float, float] = (200.0, 1000.0)  # hold period (uniform): cue on, stimulus off, choice held
    csi_ms: tuple[float, float] = (100.0, 800.0)   # cue-stimulus interval (uniform)
    rci_ms: tuple[float, float] = (100.0, 600.0)   # response-cue interval (uniform); RSI = RCI + CSI
    signal: float = 1.0                       # s: feature values are drawn from {-s, 0, +s}
    sigma_in: float = 3.0                     # s.d. of the stimulus noise added at every time step
    sigma_rec: float = 0.05                   # s.d. of the private noise eta(t)
    n_trials: int = 6                         # trials per sequence
    train_horizon_ms: float = 12500.0         # training sequences are cut or padded to this length

    def steps(self, ms: float) -> int:
        return int(round(ms / self.dt))

    @property
    def window_steps(self) -> int:
        """Window in which the response time is measured: the stimulus plus the shortest hold period."""
        return self.steps(self.stim_ms) + self.steps(self.hold_ms[0])


def task_config_from(saved: dict) -> TaskConfig:
    """Rebuild the TaskConfig that was stored in a checkpoint."""
    return TaskConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})


@dataclass(frozen=True)
class TrainConfig:
    steps_per_epoch: int = 100                # one "epoch" = 100 gradient steps
    joint_epochs: int = 20                    # joint training on both tasks
    batch_seqs: int = 16                      # sequences per gradient step (16 x 6 = 96 trials)
    lr_adam: float = 1e-3
    lr_rank1: float = 3e-3                    # Adam learning rate of the rank-one validation networks
    lr_sgd: float = 0.1
    momentum: float = 0.9
    clip: float = 1.0                         # per-network gradient-norm clipping
    log_every: int = 50
    snapshot_epochs: tuple[int, ...] = (0, 1, 2, 5, 10)   # joint epochs after which a snapshot is kept


@dataclass(frozen=True)
class Condition:
    """One learning history."""

    k: int = 0                    # pre-training epochs on task A alone
    rho: int = 1                  # task-A : task-B trial ratio during joint training
    norm_control: bool = False    # rescale input weights after every update so that |w_in,A| = |w_in,B|
    optimizer: str = "adam"       # "adam" or "sgd" (SGD with momentum)
    variant: str = "full"         # "full", "dale" or "rank1"

    @property
    def name(self) -> str:
        parts = [self.variant, f"k{self.k}", f"rho{self.rho}"]
        if self.norm_control:
            parts.append("norm")
        if self.optimizer != "adam":
            parts.append(self.optimizer)
        return "_".join(parts)

    def as_dict(self) -> dict:
        return {**asdict(self), "name": self.name}


K_LEVELS = (0, 5, 10)
RHO_LEVELS = (1, 2, 4)
N_SEEDS = 20
RT_THRESHOLDS = (0.3, 0.5, 0.7)
MAIN_THRESHOLD = 0.5
MIN_ACCURACY = 0.90
FIXED_POINT_TOL = 1e-4      # ||F(h)|| in 1/ms
SLOW_EIG_TOL = 0.1          # |lambda| tau


def training_conditions(variant: str) -> list[Condition]:
    """The 15 learning histories trained for each network variant.

    9  k x rho grid (Adam, unconstrained input norms)
    4  input-norm control at the corners of the grid
    2  second optimiser (SGD with momentum) at the baseline and at the strongest overlearning
    """
    k_max, rho_max = K_LEVELS[-1], RHO_LEVELS[-1]
    conds = [Condition(k=k, rho=rho, variant=variant) for k in K_LEVELS for rho in RHO_LEVELS]
    conds += [
        Condition(k=k, rho=rho, norm_control=True, variant=variant)
        for k in (0, k_max) for rho in (1, rho_max)
    ]
    conds += [
        Condition(k=0, rho=1, optimizer="sgd", variant=variant),
        Condition(k=k_max, rho=rho_max, optimizer="sgd", variant=variant),
    ]
    return conds


def all_conditions() -> list[Condition]:
    """15 conditions x 2 variants, plus the rank-one validation networks (hypothesis V)."""
    return training_conditions("full") + training_conditions("dale") + [Condition(variant="rank1")]


def network_seeds(cond_index: int, n: int = N_SEEDS, base: int = 20261022) -> list[int]:
    """Independent, fixed seeds for the networks of one condition."""
    return [base + 1000 * cond_index + i for i in range(n)]
