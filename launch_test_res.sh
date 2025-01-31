#!/bin/bash

# Launch all generated SLURM test scripts

sbatch slurm_scripts/test_CVCtokvasir_res.slurm
sbatch slurm_scripts/test_flairtot1_res.slurm
sbatch slurm_scripts/test_flairtot2_res.slurm
sbatch slurm_scripts/test_i3tolw4_res.slurm
sbatch slurm_scripts/test_i3toweih_res.slurm
sbatch slurm_scripts/test_kvasirtoCVC_res.slurm
sbatch slurm_scripts/test_lw4toi3_res.slurm
sbatch slurm_scripts/test_lw4toweih_res.slurm
sbatch slurm_scripts/test_t1toflair_res.slurm
sbatch slurm_scripts/test_t1tot2_res.slurm
sbatch slurm_scripts/test_t2toflair_res.slurm
sbatch slurm_scripts/test_t2tot1_res.slurm
sbatch slurm_scripts/test_weihtoi3_res.slurm
sbatch slurm_scripts/test_weihtolw4_res.slurm
