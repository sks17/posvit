#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/config.sh"
mkdir -p logs

submit() {                      # submit <command> [dependency-jobid]
    local cmd="$1" dep="${2:-}"
    local scope; scope=$(posvit meta "$cmd" --field scope)
    local gpu;   gpu=$(posvit meta "$cmd" --field needs_gpu)
    local args=(--account="${HYAK_ACCOUNT}" --job-name="posvit_${cmd}"
                --time="$(posvit meta "$cmd" --field time)"
                --mem="$(posvit meta "$cmd" --field mem)")
    if [ "$gpu" = "True" ]; then
        args+=(--partition="${GPU_PARTITION}" --gpus-per-node="${GPU_SPEC}")
    else
        args+=(--partition="${CPU_PARTITION}")
    fi
    # DERIVED from the registry — not a range typed into a #SBATCH comment.
    [ "$scope" = "per-model" ] && args+=(--array="$(posvit array-spec "$cmd")")
    [ -n "$dep" ] && args+=(--dependency=afterok:"$dep")
    sbatch --parsable "${args[@]}" "${HERE}/job.slurm" "$cmd"
}

prep=$(submit verify)
mask=$(submit masks   "$prep")
eval_=$(submit evaluate "$mask")
met=$(submit metrics  "$eval_")
int=$(submit intervene "$eval_")
ctl=$(submit controls "$int")
submit mechanisms "$ctl"
echo "submitted; watch: squeue -u \$USER"