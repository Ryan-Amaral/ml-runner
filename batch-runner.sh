# something like this
tmux new-session -d -s mysesh
tmux send-keys -t mysesh:0 "python script.py" Enter