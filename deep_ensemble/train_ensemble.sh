#!/bin/bash
#SBATCH -A m4259
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time 12:00:00

# Load modules
source ~/.bashrc
conda activate FNO

for i in {0..9}
do
    echo "Training ensemble $i"
    python train_model.py --ensemble_id $i --data_path /pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5 --df_path ./results/results.csv
done

