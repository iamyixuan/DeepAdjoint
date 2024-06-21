#!/bin/bash
#SBATCH --output=./_job_history/%j.out
#SBATCH --error=./_job_history/%j.err
#SBATCH -A m4259
#SBATCH --qos=regular
#SBATCH --nodes=10
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time 12:00:00 


# conda activate deeplearning
export WORLD_SIZE=40
export MASTER_PORT=38172
export TRAIN=1  
export LOSS=FNO

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


export MASTER_ADDR=$master_addr
export LOSS=FNO
srun python train_FNO.py -batch_size 1\
                      -epochs 5000\
                      -lr 0.001\
                      -model 4D\
                      -data GM_D_AVG\
                      -train False
                      -model_path "./experiments/4D-GM_D_AVG-MSE-Adam/best_model_state.pt"
