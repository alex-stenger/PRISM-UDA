#!/bin/bash

sbatch slurm_scripts/I3toLW4_mic_hrda.slurm
sbatch slurm_scripts/I3toWeiH_mic_hrda.slurm
sbatch slurm_scripts/LW4toWeiH_mic_hrda.slurm
sbatch slurm_scripts/LW4toI3_mic_hrda.slurm
sbatch slurm_scripts/WeiHtoLW4_mic_hrda.slurm
sbatch slurm_scripts/WeiHtoI3_mic_hrda.slurm