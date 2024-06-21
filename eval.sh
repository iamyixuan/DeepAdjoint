#! /bin/bash

data_path="./experiments/FNO-GM_D_AVG-MSE"
cal_score="False"
plot_predictions="True"
rollout="False"
plot_trend="False"
var_id=4

source ~/.bashrc
conda activate FNO

python evaluate.py\
        --cal_score $cal_score\
        --plot_predictions $plot_predictions\
        --rollout $rollout\
        --plot_trend $plot_trend\
        --data_path $data_path\
        --var_id $var_id\
            
