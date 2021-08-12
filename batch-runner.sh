#!/bin/bash

for i in {1..4}; do
    tmux new-session -d -s 2dbiped-gp${i}
    tmux send-keys -t 2dbiped-gp${i}:0 "conda activate pyton" Enter
    tmux send-keys -t 2dbiped-gp${i}:0 "python -c 'from mlrunner.experiment import run_experiment; run_experiment(instance=${i}, processes=2)'" Enter
done