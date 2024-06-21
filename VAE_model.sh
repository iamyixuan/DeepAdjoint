#!/bin/bash
#SBATCH --output=./_job_history/%j.out
#SBATCH --error=./_job_history/%j.err
#SBATCH -A m4259
#SBATCH --qos=regular
#SBATCH --nodes=3
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time 6:00:00 


# conda activate deeplearning
export WORLD_SIZE=12
export MASTER_PORT=38172
export TRAIN=1  
export LOSS=VAE

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


export MASTER_ADDR=$master_addr
srun python train_vae.py -batch_size 32\
                      -epochs 5000\
                      -lr 1e-3\
                      -model VAE-4-save-lr1e-3\
                      -data GM_D_AVG-temp\
                      -train True
                      -model_path "./experiments/4D-GM_D_AVG-MSE-Adam/best_model_state.pt"
