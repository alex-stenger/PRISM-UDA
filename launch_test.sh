#!/bin/bash

# Launch all generated SLURM test scripts

sbatch slurm_scripts/test_CVCtokvasir.slurm
sbatch slurm_scripts/test_i3tolw4.slurm
sbatch slurm_scripts/test_i3toweih.slurm
sbatch slurm_scripts/test_kvasirtoCVC.slurm
sbatch slurm_scripts/test_lw4toi3.slurm
sbatch slurm_scripts/test_lw4toweih.slurm
sbatch slurm_scripts/test_weihtoi3.slurm
sbatch slurm_scripts/test_weihtolw4.slurm
