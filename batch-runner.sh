#!/bin/bash

for i in {1..5}; do
    tmux new-session -d -s 2dbiped-tpgsbb${i}
    tmux send-keys -t 2dbiped-tpgsbb${i}:0 "source activate 2dbiped-tpgsbb" Enter
    tmux send-keys -t 2dbiped-tpgsbb${i}:0 "python -c 'from mlrunner.experiment import run_experiment; run_experiment(instance=${i}, end_generation=5000, environment=\"BipedalWalkerHardcore-v3\", frames=2000, processes=2)'" Enter
done