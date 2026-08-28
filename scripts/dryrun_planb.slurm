#!/bin/bash
# dryrun_planb.slurm  --  bwUniCluster 3.0 MINI-Dry-Run der Plan-B-Kette (WF-02)
# ---------------------------------------------------------------------------
# STATUS: UNVERIFIZIERT. Dieses Skript wurde nie submittet. Es entstand in einer
# Umgebung ohne Cluster-Zugang (kein sbatch, kein enroot, kein SSH zu uc3), so
# dass weder Queue-Namen noch Modul-Namen noch Pfade gegen die reale Maschine
# geprueft sind. Vor dem ersten Lauf: Partition mit `sinfo` verifizieren und
# einen Trockenlauf mit `sbatch --test-only` machen. Bis dahin ist das hier
# eine Vorlage, kein lauffaehiger Job.
# ---------------------------------------------------------------------------
# Winziger Crash-Freiheits-Test der 3-Stufen-Pipeline (chain_e2e.py):
# S1 Vertices -> S2 Pointer-Faces -> S3 HO-Geometrie -> Chain-Inferenz ->
# Galerie. KEIN Volllauf, keine Qualitaetsaussage -- es zaehlt nur: laeuft die
# Pipeline auf einem sauberen Checkout ohne Crash durch (WF-02-Frage)?
#
# Zwei Modi (env MODE):
#   MODE=train (Default)  Mini-Training von Grund auf (d=32, 1 Epoche, LIMIT
#                         Meshes) + Chain-Inferenz + Galerie + Checkpoint-Save.
#   MODE=load             Nur Chain-Inferenz aus vorhandenen 3-Kopf-Checkpoints
#                         (--load-dir) -- testet, ob Checkpoints auf einer
#                         sauberen Maschine laden (WF-02-Kernfrage).
#
# bwUniCluster 3.0 GPU-Queues -- Namen/Limits mit `sinfo` verifizieren:
#   sinfo -o "%P %l %G %c %m" | grep -i a100
#   gpu_a100_short   Kurzjob/dev, 30 min max (getestet: Jobs 6122195/6122568/
#                    6123500, A100-PCIE-40GB)  <- diese
#   gpu_a100_il      48 h max, 4x A100 (nur fuer Volllauf, nicht Dry-Run)
#
# Setup (einmalig, auf der LOGIN-Node -- nur dort gibt es Internet):
#   ws_allocate meshtron 60                    # falls noch kein Workspace
#   cd $(ws_find meshtron)/meshtron            # Repo im Workspace
#   export UV_CACHE_DIR=$PWD/.uv-cache         # Cache auf /work (Hardlinks)
#   uv sync                                    # .venv aus pyproject.toml + uv.lock
#
# Aufruf (Script liegt im quadmesh-Superrepo unter scripts/, muss IM
# meshtron-Verzeichnis laufen -- MESHTRON_DIR erzwingt das Verzeichnis):
#   cp scripts/dryrun_planb.slurm $(ws_find meshtron)/meshtron/
#   cd $(ws_find meshtron)/meshtron
#   sbatch dryrun_planb.slurm                          # Modus train
#   MODE=load sbatch dryrun_planb.slurm                # Modus load (s.u.)
#   MESHTRON_DIR=$(ws_find meshtron)/meshtron sbatch dryrun_planb.slurm
#
# Modus load erwartet 3-Kopf-Checkpoints (meta.pt + s1_vertex.pt +
# s2_pointer.pt + s3_geom.pt) unter LOAD_DIR, Default runs_planb/e2e_6123500
# (Cluster-Run vom 2026-08-03, d_model=256). Vorher pruefen:
#   ls $(ws_find meshtron)/meshtron/runs_planb/e2e_6123500/
# Falls weg: erst MODE=train laufen lassen und LOAD_DIR auf das neue
# runs_planb/dryrun_<jobid>/ zeigen lassen.
#
# Logs -> slurm_logs/dryrun_<jobid>.{out,err}
# Ergebnis -> ${SAVE_DIR}/{meta,s1_vertex,s2_pointer,s3_geom}.pt + Galerie-PNG
# ---------------------------------------------------------------------------

#SBATCH --job-name=meshtron-dryrun
#SBATCH --partition=gpu_a100_short
#SBATCH --output=slurm_logs/dryrun_%j.out
#SBATCH --error=slurm_logs/dryrun_%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000mb

set -euo pipefail
cd "${MESHTRON_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p slurm_logs

# --- Environment: uv-managed .venv (aus pyproject.toml + uv.lock) ------------
# torch cu128 wheels bringen CUDA-Runtime mit -> nur NVIDIA-Treiber noetig.
# `uv sync` EINMAL auf der LOGIN-Node laufen lassen (Internet!); der Block
# unten ist nur Fallback und schlaegt auf der Compute-Node ohne Internet fehl.
ENV_PATH=${ENV_PATH:-$PWD/.venv}
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
if [[ ! -f "${ENV_PATH}/bin/activate" ]]; then
  echo "kein venv unter ${ENV_PATH} -> versuche 'uv sync' ..."
  if ! command -v uv >/dev/null 2>&1; then
    echo "FEHLER: 'uv' nicht im PATH. Einmalig installieren:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh    # dann Shell neu"
    echo "Danach auf der LOGIN-Node (Internet!):  cd '$PWD' && uv sync"
    exit 1
  fi
  uv sync                       # erstellt $PWD/.venv aus uv.lock
fi
source "${ENV_PATH}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Mini-Konfiguration (per env ueberschreibbar) ----------------------------
MODE=${MODE:-train}             # train | load
DATA=${DATA:-domain_data_aug.pt}
SEED=${SEED:-0}
LR=${LR:-5e-4}
SAVE_DIR=${SAVE_DIR:-runs_planb/dryrun_${SLURM_JOB_ID:-local}}
OUT=${OUT:-figures/e2e/dryrun_gallery_${SLURM_JOB_ID:-local}.png}
LOAD_DIR=${LOAD_DIR:-runs_planb/e2e_6123500}   # nur MODE=load

# train-Modus: winzig (Ziel: Crash-Freiheit, keine Qualitaet)
DMODEL=${DMODEL:-32}
BATCH=${BATCH:-4}
EP1=${EP1:-1}
EP2=${EP2:-1}
EP3=${EP3:-1}
LIMIT=${LIMIT:-64}              # nur erste K Meshes (Beispiel-Bau verkuerzen).
                                # LIMIT>=20 halten: n_val = LIMIT/10 >= 2 ...
EVAL_N=${EVAL_N:-8}
GALLERY=${GALLERY:-8}           # ... und GALLERY=8 (Cap 2 je Facecount) ->
                                # >=2 Galerie-Beispiele. Bei GENAU 1 Beispiel
                                # crasht die Galerie in chain_e2e.py (1-Spalten-
                                # subplots-Array wird von np.atleast_2d falsch
                                # orientiert; axes[1,c] -> IndexError).

echo "[$(date -Is)] host=$(hostname) node=${SLURMD_NODENAME:-?} part=${SLURM_JOB_PARTITION:-?} gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "  mode=$MODE data=$DATA seed=$SEED save-dir=$SAVE_DIR out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ "$MODE" == "load" ]]; then
  # d_model MUSS zum Checkpoint passen (meta.pt-Assert in chain_e2e.py);
  # e2e_6123500 wurde mit d=256 trainiert.
  DMODEL=${DMODEL_LOAD:-256}
  if [[ ! -f "${LOAD_DIR}/meta.pt" ]]; then
    echo "FEHLER: ${LOAD_DIR}/meta.pt nicht gefunden -- LOAD_DIR setzen oder"
    echo "zuerst MODE=train laufen lassen (liefert ${SAVE_DIR}/)."
    exit 1
  fi
  CMD=(python -u chain_e2e.py
    --data "$DATA"
    --d-model "$DMODEL"
    --load-dir "$LOAD_DIR"
    --eval-n "$EVAL_N"
    --gallery "$GALLERY"
    --seed "$SEED"
    --limit "$LIMIT"
    --out "$OUT")
else
  CMD=(python -u chain_e2e.py
    --data "$DATA"
    --d-model "$DMODEL"
    --batch "$BATCH"
    --ep1 "$EP1" --ep2 "$EP2" --ep3 "$EP3"
    --lr "$LR"
    --seed "$SEED"
    --limit "$LIMIT"
    --eval-n "$EVAL_N"
    --gallery "$GALLERY"
    --save-dir "$SAVE_DIR"
    --out "$OUT")
fi

echo "+ ${CMD[*]}"
srun "${CMD[@]}"
echo "[$(date -Is)] fertig (mode=$MODE)."
