#!/bin/bash


for i in {1..10}
do
    sbatch "train_ensemble_$i.sh"
done

