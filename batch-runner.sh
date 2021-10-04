#!/bin/bash

for i in {1..4}; do
    tmux new-session -d -s pbg-tpgsbb${i}
    tmux send-keys -t pbg-tpgsbb${i}:0 "conda activate pbg-tpgsbb" Enter
    tmux send-keys -t pbg-tpgsbb${i}:0 "python -c 'from mlrunner.experiment import run_experiment; run_experiment(instance=${i}, environment=\"ReacherPyBulletEnv-v0\", processes=2)'" Enter
done