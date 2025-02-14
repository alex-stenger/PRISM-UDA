#! /bin/bash
#SBATCH -p public -A miv
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -o jobs/test_lw4multitoi3multi_res.out
hostname
source deactivate
echo 'START'

TEST_ROOT=work_dirs/local-basic/250213_1444_lw4multitoi3multi_8d1ae
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.py"
CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
SHOW_DIR="${TEST_ROOT}/preds"
module load python/python-3.11.4 && activate deep-learning
python get_results.py --pred_path $SHOW_DIR --gt_path /home2020/home/miv/astenger/data/segdiff/i3multi/test/lbl/labels/
