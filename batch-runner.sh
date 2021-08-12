#!/bin/bash

# argument $1: relative path to the mlrunner experiment script

for i in {1..20}; do
    tmux new-session -d -s 2dbiped-gp${i}
    tmux send-keys -t 2dbiped-gp${i}:0 "conda activate 2dbiped-gp" Enter
    tmux send-keys -t 2dbiped-gp${i}:0 "python $1 --instance ${i} --end_generation 500 --episodes 5 --environment BipedalWalker-v3 --frames 1600 --processes 2 --trainer_checkpoint 100 --init_team_pop 360 --gap 0.5 --init_max_act_prog_size 128 --inst_del_prob 0.5 --inst_add_prob 0.5 --inst_swp_prob 0.5 --inst_mut_prob 0.5 --elitist --ops robo --rampancy 5,5,5" Enter
done