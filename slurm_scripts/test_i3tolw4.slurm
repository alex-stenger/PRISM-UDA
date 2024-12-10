#! /bin/bash
#SBATCH -p publicgpu -A miv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --constraint="gpuv100|gpua40|gpua100"
#SBATCH -o jobs/test_i3tolw4.out
hostname
source deactivate
module load python/python-3.8.18
source ~/venv/hrda/bin/activate
python --version
echo 'START'

TEST_ROOT=work_dirs/local-basic/241209_1501_i3tolw4_7790a
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.py"
CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
SHOW_DIR="${TEST_ROOT}/preds"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1
