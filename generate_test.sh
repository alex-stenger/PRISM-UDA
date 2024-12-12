#!/bin/bash

# Directories
CONFIG_DIR="configs/mic"
WORK_DIR="work_dirs/local-basic"
OUTPUT_DIR="slurm_scripts"
LAUNCH_SCRIPT="launch_test.sh"

# Create the output directory
mkdir -p $OUTPUT_DIR

# Create or overwrite the launch script
echo "#!/bin/bash" > $LAUNCH_SCRIPT
echo "" >> $LAUNCH_SCRIPT
echo "# Launch all generated SLURM test scripts" >> $LAUNCH_SCRIPT
echo "" >> $LAUNCH_SCRIPT

# Iterate through each config file
for CONFIG_FILE in $CONFIG_DIR/*.py; do
    # Extract the "name" field from the config
    NAME=$(grep -Po "(?<=name = ')[^']*" $CONFIG_FILE)
    
    # Find the latest work directory for this name
    LATEST_DIR=$(ls -d $WORK_DIR/*_${NAME}_* 2>/dev/null | sort -r | head -n 1)
    
    if [ -z "$LATEST_DIR" ]; then
        echo "No work directory found for $NAME, skipping..."
        continue
    fi

    # Extract the job name and directory for the SLURM script
    JOB_NAME="test_$NAME"
    SLURM_SCRIPT="$OUTPUT_DIR/${JOB_NAME}.slurm"

    # Extract source and target from the name
    SOURCE_TO_TARGET=${NAME,,} # Convert to lowercase for consistency
    TARGET=${SOURCE_TO_TARGET#*to} # Extract everything after "to"

    # Create the SLURM script
    cat <<EOL > $SLURM_SCRIPT
#! /bin/bash
#SBATCH -p publicgpu -A miv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --constraint="gpuv100|gpua40|gpua100"
#SBATCH -o jobs/${JOB_NAME}.out
hostname
source deactivate
module load python/python-3.8.18
source ~/venv/hrda/bin/activate
python --version
echo 'START'

TEST_ROOT=$LATEST_DIR
CONFIG_FILE="\${TEST_ROOT}/*\${TEST_ROOT: -1}.py"
CHECKPOINT_FILE="\${TEST_ROOT}/latest.pth"
SHOW_DIR="\${TEST_ROOT}/preds"
echo 'Config File:' \$CONFIG_FILE
echo 'Checkpoint File:' \$CHECKPOINT_FILE
echo 'Predictions Output Directory:' \$SHOW_DIR
python -m tools.test \${CONFIG_FILE} \${CHECKPOINT_FILE} --eval mIoU --show-dir \${SHOW_DIR} --opacity 1
load-python
python get_results.py --pred_path \$SHOW_DIR --gt_path /home2020/home/miv/astenger/data/segdiff/${TARGET}/test/lbl/labels/
EOL

    # Add this script to the launch script
    echo "sbatch $SLURM_SCRIPT" >> $LAUNCH_SCRIPT

    echo "Generated SLURM script for $NAME at $SLURM_SCRIPT"
done

# Make the launch script executable
chmod +x $LAUNCH_SCRIPT
echo "Launch script created: $LAUNCH_SCRIPT"
