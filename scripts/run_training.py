"""Train the networks of the study: 15 learning histories x 20 seeds x 2 variants, plus rank-one networks.

Every condition is one job that trains its 20 networks in a single batched computation on one CPU
core; jobs run in parallel in a process pool. Finished conditions are skipped, so the script can be
interrupted and restarted.

    python scripts/run_training.py                       # everything (runs/main)
    python scripts/run_training.py --only full_k0_rho1   # selected conditions
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import torch

from stroop_rnn.config import Condition, TaskConfig, TrainConfig, all_conditions, network_seeds
from stroop_rnn.train import train_condition

ROOT = Path(__file__).resolve().parents[1]


def run_job(job: dict) -> tuple[str, float]:
    torch.set_num_threads(1)
    cond = Condition(**job["cond"])
    out_dir = Path(job["out_dir"])
    log_path = out_dir / "logs" / f"{cond.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")
        print(msg, flush=True)

    def save(ckpt: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)

    # snapshots during joint training (epoch 0 = end of the pre-training on task A)
    train_cfg = job["train_cfg"]
    pre_steps = cond.k * train_cfg.steps_per_epoch
    snapshot_steps = {pre_steps + e * train_cfg.steps_per_epoch: e for e in train_cfg.snapshot_epochs}

    def on_log(step: int, total: int, make_checkpoint) -> None:
        if (out_dir / "STOP").exists():      # cooperative way to abort a run: touch <out>/STOP
            raise RuntimeError(f"{cond.name}: stop file found at step {step}")
        if step in snapshot_steps and step < total:
            save(make_checkpoint(), out_dir / "snapshots" / f"{cond.name}_e{snapshot_steps[step]:02d}.pt")
        every = job["snapshot_every"]
        if every and step % every == 0 and step < total:
            save(make_checkpoint(), out_dir / "partial" / f"{cond.name}.pt")

    seeds = network_seeds(job["index"], job["n_seeds"])
    ckpt = train_condition(
        cond, seeds, job["task_cfg"], job["train_cfg"], data_seed=seeds[0] + 500, log=log, on_log=on_log
    )
    save(ckpt, out_dir / f"{cond.name}.pt")
    return cond.name, ckpt["elapsed_s"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/main", help="output directory (relative to the repository)")
    ap.add_argument("--only", nargs="*", default=None, help="condition names to train (default: all)")
    ap.add_argument("--seeds", type=int, default=None, help="networks per condition (default: 20)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="retrain conditions that already have a checkpoint")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also keep the latest intermediate checkpoint in <out>/partial, updated every this many steps")
    # overrides used for pilot runs
    ap.add_argument("--joint-epochs", type=int, default=None)
    ap.add_argument("--sigma-in", type=float, default=None)
    ap.add_argument("--sigma-rec", type=float, default=None)
    ap.add_argument("--hold-ms", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--horizon-ms", type=float, default=None)
    ap.add_argument("--lr-adam", type=float, default=None)
    ap.add_argument("--lr-sgd", type=float, default=None)
    ap.add_argument("--batch-seqs", type=int, default=None)
    args = ap.parse_args()

    task_cfg, train_cfg = TaskConfig(), TrainConfig()
    task_over = dict(sigma_in=args.sigma_in, sigma_rec=args.sigma_rec,
                     hold_ms=None if args.hold_ms is None else tuple(args.hold_ms),
                     train_horizon_ms=args.horizon_ms)
    train_over = dict(joint_epochs=args.joint_epochs, lr_adam=args.lr_adam, lr_sgd=args.lr_sgd,
                      batch_seqs=args.batch_seqs)
    task_cfg = replace(task_cfg, **{k: v for k, v in task_over.items() if v is not None})
    train_cfg = replace(train_cfg, **{k: v for k, v in train_over.items() if v is not None})

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = all_conditions()
    jobs = []
    for index, cond in enumerate(conds):
        if args.only is not None and cond.name not in args.only:
            continue
        if (out_dir / f"{cond.name}.pt").exists() and not args.force:
            print(f"skip {cond.name} (checkpoint exists)")
            continue
        n_seeds = args.seeds if args.seeds is not None else len(network_seeds(index))
        cond_dict = {k: v for k, v in cond.as_dict().items() if k != "name"}
        jobs.append(dict(cond=cond_dict, index=index, n_seeds=n_seeds, task_cfg=task_cfg,
                         train_cfg=train_cfg, out_dir=str(out_dir), snapshot_every=args.snapshot_every))
    if args.only is not None:
        unknown = set(args.only) - {c.name for c in conds}
        if unknown:
            raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    # longest jobs first, so that the pool stays evenly loaded
    jobs.sort(key=lambda j: -(j["cond"]["k"] + train_cfg.joint_epochs))
    print(f"{len(jobs)} conditions to train with {args.workers} workers -> {out_dir}", flush=True)

    t0 = time.time()
    with mp.get_context("spawn").Pool(min(args.workers, max(len(jobs), 1))) as pool:
        for i, (name, elapsed) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] finished {name} in {elapsed / 60:.1f} min "
                  f"(wall {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
