#!/bin/bash

# Launch all generated SLURM test scripts

sbatch slurm_scripts/test_i3tolw4.slurm
sbatch slurm_scripts/test_lw4toi3.slurm
