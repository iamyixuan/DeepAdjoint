#!/bin/bash
#SBATCH --output=./_job_history/%j.out
#SBATCH --error=./_job_history/%j.err
#SBATCH -A m4259
#SBATCH --qos=regular
#SBATCH --nodes=5
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time 12:00:00 


# conda activate deeplearning
export WORLD_SIZE=20
export MASTER_PORT=12340
export TRAIN=1  

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

srun python SOMAforward.py -batch_size 32\
                      -epochs 1500\
                      -lr 0.0001\
                      -model_name SOMA-U-mask-five-var-multinodes\
                      -mask True\
                      -net_type UNet\
