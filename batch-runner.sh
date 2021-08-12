#!/bin/bash

for i in {1..4}; do
    tmux new-session -d -s 2dbiped-sbb${i}
    tmux send-keys -t 2dbiped-sbb${i}:0 "source activate 2dbiped-sbb" Enter
    tmux send-keys -t 2dbiped-sbb${i}:0 "python -c 'from mlrunner.experiment import run_experiment; run_experiment(instance=${i}, processes=2)'" Enter
done