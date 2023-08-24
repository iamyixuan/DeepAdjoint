#!/bin/bash
#SBATCH --output=./_job_history/%j.out
#SBATCH --error=./_job_history/%j.err
#SBATCH -A m4259
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus 4
#SBATCH --time 12:00:00 


# conda activate deeplearning

python SOMAforward.py -batch_size 32\
                      -epochs 500\
                      -lr 0.0005\
                      -model_name SOMA-U-mask-five-var\
                      -mask True\
                      -net_type UNet\
