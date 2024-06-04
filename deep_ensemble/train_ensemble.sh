#!/bin/bash
#SBATCH --output=./_job_history/%j.out
#SBATCH --error=./_job_history/%j.err
#SBATCH -A m4259
#SBATCH --qos=regular
#SBATCH --nodes=10
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time 12:00:00

# Load modules
source ~/.bashrc
conda activate FNO

for i in {0..9}
do
    echo "Training ensemble $i"
    python train_ensemble.py --ensemble_id $i\
                            --data_path "$PSCRATCH/dataset/de_dataset/GM-prog-var-surface.hdf5"\
                            --df_path "./results.csv"
done

