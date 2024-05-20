#! /bin/bash

data_path="./experiments/FNO-GM_D_AVG-MSE"
cal_score="True"
plot_predictions="False"
rollout="False"
plot_trend="False"

source ~/.bashrc
conda activate FNO

python evaluate.py\
        --cal_score $cal_score\
        --plot_predictions $plot_predictions\
        --rollout $rollout\
        --plot_trend $plot_trend\
        --data_path $data_path\
            
