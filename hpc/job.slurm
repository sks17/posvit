#!/bin/bash
# PosViT HPC entry script.
# Runs one posvit command, with optional SLURM array index.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

set -euo pipefail

# Load local job configuration.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${HERE}/config.sh" ]; then
    echo "ERROR: ${HERE}/config.sh not found." >&2
    echo "  cp ${HERE}/config.example.sh ${HERE}/config.sh && \$EDITOR ${HERE}/config.sh" >&2
    exit 1
fi
source "${HERE}/config.sh"

# Activate conda environment and run command.
CMD="${1:?usage: sbatch job.slurm <posvit-command>}"
module load conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_PREFIX}"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    posvit "$CMD" --index "${SLURM_ARRAY_TASK_ID}"
else
    posvit "$CMD"
fi