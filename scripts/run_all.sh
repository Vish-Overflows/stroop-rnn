#!/usr/bin/env bash
# Reproduce the whole study: training -> analysis -> validation -> statistics -> figures.
#
#   scripts/run_all.sh                   everything (the training takes several hours on a 10-core laptop)
#   scripts/run_all.sh --skip-training   start from the trained networks in runs/main
#   scripts/run_all.sh --wait            wait for a training run that is already in progress, then continue
#   scripts/run_all.sh --figures-only    only regenerate statistics and figures from results/
#
# Environment variables: WORKERS (default 9), RUNS (runs/main), RESULTS (results), FIGURES (figures).
# Every step skips work that is already done, so the script can be restarted after an interruption.

set -u
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
WORKERS=${WORKERS:-9}
RUNS=${RUNS:-runs/main}
RESULTS=${RESULTS:-results}
FIGURES=${FIGURES:-figures}
MODE=${1:-all}

export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/runs/.mplconfig"
mkdir -p "$RUNS" "$MPLCONFIGDIR"
status=0

step() {                       # step <label> <command...>: run one stage, report its duration, remember failures
    local label=$1 t0=$SECONDS
    shift
    echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $label"
    if "$@"; then
        echo "--- $label: ok ($(( (SECONDS - t0) / 60 )) min)"
    else
        echo "--- $label: FAILED with exit code $? ($(( (SECONDS - t0) / 60 )) min)"
        status=1
    fi
}

case "$MODE" in
    all)
        step "training" "$PY" scripts/run_training.py --out "$RUNS" --workers "$WORKERS"
        ;;
    --wait)
        echo "=== $(date '+%Y-%m-%d %H:%M:%S')  waiting for the training in $RUNS"
        until grep -q "^all done" "$RUNS/train_stdout.log" 2>/dev/null; do sleep 60; done
        ;;
    --skip-training | --figures-only) ;;
    *)
        echo "unknown option $MODE" >&2
        exit 2
        ;;
esac

if [ "$MODE" != "--figures-only" ]; then
    echo "trained conditions: $(ls "$RUNS" | grep -c '\.pt$')"
    step "analysis of the trained networks" "$PY" scripts/run_analysis.py --runs "$RUNS" --workers "$WORKERS"
    step "behaviour during training (snapshots)" "$PY" scripts/run_analysis.py --runs "$RUNS" --snapshots --workers "$WORKERS"
    step "E0 validation" "$PY" scripts/e0_validation.py --runs "$RUNS" --out "$RESULTS"
    step "collecting results" "$PY" scripts/collect_results.py --runs "$RUNS" --out "$RESULTS"
fi
step "statistics" "$PY" scripts/run_stats.py --results "$RESULTS"
step "figures" "$PY" scripts/make_figures.py --results "$RESULTS" --out "$FIGURES"

echo "=== $(date '+%Y-%m-%d %H:%M:%S')  pipeline finished with status $status"
[ "$MODE" = "--figures-only" ] || touch "$RUNS/PIPELINE_DONE"
exit $status
