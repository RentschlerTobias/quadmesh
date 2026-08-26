#!/bin/sh
# Generic shell template for running domain_partition data generation on a
# cluster node. Copy and adapt to your scheduler (SLURM, SGE, ...).
#
# USER INPUT REQUIRED: cluster paths vary. Set the following environment
# variables before running, e.g.:
#   export QUADMESH_WORKDIR=/path/to/domain_partition
#   export QUADMESH_PYTHON_ENV=/path/to/venv/bin/activate
#   ./queuing.sh

set -e

cd "${QUADMESH_WORKDIR:?Please set QUADMESH_WORKDIR}"
source "${QUADMESH_PYTHON_ENV:?Please set QUADMESH_PYTHON_ENV}"
export OSLO_LOCK_PATH=~/tmp

export FOAM_SIGFPE=false

python data_generator.py > output.log
