#!/bin/bash
#SBATCH --output=./_job_history/%j.out
#SBATCH --error=./_job_history/%j.err
#SBATCH -A m4259
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time 3:00:00 


# conda activate deeplearning
export LOSS=FNO

python cal_adjoint.py -model_path /global/homes/y/yixuans/DeepAdjoint/checkpoints/2023-11-01_FNO-REDI/best_model\
        -data REDI\