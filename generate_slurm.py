
import os

# Directories
config_dir = "configs/mic"
output_dir = "slurm_scripts"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Base SLURM script template
slurm_template = """#!/bin/bash
#SBATCH -p publicgpu -A miv
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100|gpuv100"
#SBATCH -o jobs/{job_name}.out
hostname
source deactivate
module load python/python-3.8.18
source ~/venv/hrda/bin/activate
python --version
echo 'START'
python run_experiments.py --config {config_path}
echo 'END'
"""

# Iterate over all configuration files in the config directory
for config_file in os.listdir(config_dir):
    if config_file.endswith(".py"):  # Ensure it's a Python config file
        config_path = os.path.join(config_dir, config_file)
        job_name = os.path.splitext(config_file)[0]  # Remove file extension for job name
        
        # Create the SLURM script content
        slurm_script = slurm_template.format(job_name=job_name, config_path=config_path)
        
        # Define the output file path
        slurm_output_path = os.path.join(output_dir, f"{job_name}.slurm")
        
        # Write the SLURM script to the file
        with open(slurm_output_path, "w") as f:
            f.write(slurm_script)

print(f"SLURM scripts generated in '{output_dir}' directory.")

