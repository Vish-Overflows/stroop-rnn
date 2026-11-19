"""Run the dynamical-systems analysis and experiments E1-E5 on every trained condition.

One job per checkpoint (single-threaded); jobs run in parallel in a process pool and finished jobs
are skipped, so the script can be interrupted and restarted.

    python scripts/run_analysis.py                        # all checkpoints in runs/main
    python scripts/run_analysis.py --only full_k0_rho1    # selected conditions
    python scripts/run_analysis.py --snapshots            # behaviour during joint training (snapshots)

Output: <runs>/analysis/<condition>.pt (and <runs>/analysis/snapshots/<condition>_eXX.pt), each a
dictionary of NumPy arrays whose first dimension runs over the networks of the condition.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from stroop_rnn import experiments as ex
from stroop_rnn.config import task_config_from
from stroop_rnn.dynamics import analyse_checkpoint
from stroop_rnn.model import BatchedRNN

ROOT = Path(__file__).resolve().parents[1]


def run_job(job: dict) -> tuple[str, float, str]:
    torch.set_num_threads(1)
    t0 = time.time()
    path, out_path = Path(job["path"]), Path(job["out"])
    try:
        ckpt = torch.load(path, weights_only=False)
        model = BatchedRNN.from_checkpoint(ckpt)
        cfg = task_config_from(ckpt["task_config"])
        result = {
            "condition": ckpt["condition"], "seeds": np.asarray(model.seeds), "task_config": ckpt["task_config"],
            "history": ckpt["history"], "train_steps": int(ckpt["history"]["step"][-1]),
        }
        result["battery"] = ex.behaviour_battery(model, cfg, n_base=job["n_base"])
        if job["full"]:
            rows, details = analyse_checkpoint(model, cfg)
            result["dynamics"] = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
            result["details"] = details
            h_neutral = np.stack([d["h_neutral"] for d in details])
            result["psychometric"] = ex.psychometric(model, cfg, h_neutral, n_rep=job["psy_rep"])
            result["step"] = ex.step_responses(model, cfg, details)
            result["perturbations"] = ex.perturbations(model, cfg, details)
            result["relaxation"] = ex.relaxation(model, cfg, details)
            result["interventions"] = ex.interventions(model, cfg, details, n_rep=job["e5_rep"])
            result["long_rsi"] = ex.long_rsi_repeats(model, cfg, n_rep=job["long_rep"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".pt.tmp")
        torch.save(result, tmp)
        tmp.replace(out_path)
        return path.stem, time.time() - t0, "ok"
    except Exception:                                            # keep the other jobs running
        return path.stem, time.time() - t0, traceback.format_exc()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/main", help="directory with the trained checkpoints")
    ap.add_argument("--only", nargs="*", default=None, help="condition names (default: all checkpoints found)")
    ap.add_argument("--snapshots", action="store_true",
                    help="analyse the snapshots taken during joint training (behaviour only) instead of the final networks")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-base", type=int, default=None, help="base sequences of the behavioural battery (x 9 versions)")
    ap.add_argument("--psy-rep", type=int, default=40)
    ap.add_argument("--e5-rep", type=int, default=60)
    ap.add_argument("--long-rep", type=int, default=24)
    args = ap.parse_args()

    runs = ROOT / args.runs
    if args.snapshots:
        paths = sorted((runs / "snapshots").glob("*.pt"))
        out_dir = runs / "analysis" / "snapshots"
        n_base = args.n_base or 100
    else:
        paths = sorted(runs.glob("*.pt"))
        out_dir = runs / "analysis"
        n_base = args.n_base or 400
    if args.only is not None:
        paths = [p for p in paths if p.stem in args.only or p.stem.rsplit("_e", 1)[0] in args.only]
    jobs = []
    for p in paths:
        out = out_dir / p.name
        if out.exists() and not args.force:
            continue
        jobs.append(dict(path=str(p), out=str(out), full=not args.snapshots, n_base=n_base, psy_rep=args.psy_rep,
                         e5_rep=args.e5_rep, long_rep=args.long_rep))
    print(f"{len(jobs)} checkpoints to analyse with {args.workers} workers -> {out_dir}", flush=True)
    if not jobs:
        return
    t0 = time.time()
    failed = 0
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for i, (name, elapsed, status) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            if status == "ok":
                print(f"[{i}/{len(jobs)}] {name} done in {elapsed / 60:.1f} min (wall {(time.time() - t0) / 60:.1f} min)", flush=True)
            else:
                failed += 1
                print(f"[{i}/{len(jobs)}] {name} FAILED after {elapsed / 60:.1f} min\n{status}", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min, {failed} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
