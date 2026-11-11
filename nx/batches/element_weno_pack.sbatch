#!/usr/bin/env bash
# Element-WENO tasks packed on one exclusive nx node: four pinned slots of 16 cores (two per socket).
# Usage (from the repository root): sbatch --export=ALL,TASKFILE=/path/tasks.tsv nx/batches/element_weno_pack.sbatch
#   TASKFILE: "NAME<TAB>COMMAND" per line (scripts/element_weno/pack_node.py); logs land next to it.
# Measured (docs/ELEMENT_WENO.md, "Hardware"): 4 x 16 cores give 2.5 times the throughput of single 32-core runs.
#SBATCH --job-name=ew-pack
#SBATCH --constraint=nx
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=ew-pack-%j.log
set -euo pipefail
: "${TASKFILE:?set TASKFILE}"
REPO=${REPO:-$SLURM_SUBMIT_DIR} # submit from the repository root (Slurm copies this script)
export PATH=/home/warehouse15/bberg/spack/opt/spack/linux-zen3/numactl-2.0.19-jqkle7zj62c4ihp7wg7ug6lruygup5td/bin:$PATH
python3 "$REPO/scripts/element_weno/pack_node.py" "$TASKFILE" --slot-cores "${SLOT_CORES:-16}"
