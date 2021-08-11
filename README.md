# ml-runner

## 2d-Bipedal-Walker-v3 experiments with GP

### What is it?
A general framework/codebase for running machine learning experiments. 
Primarilly evolution based reinforcement learning, but could be used for anything.

### Some Guidelines:

Different branches will be for different experiment sets. 
Some may follow a general template (as in master), others could be completely different. 
A general template will be followed where possible for efficient reuse.

Each branch should follow a common naming convention `{experiment name}-{experiment subtype}`, for example `2dbiped-tpg`.

It is recommended to use a different environment for each experiment, as typically each experiment may have different dependencies, some conflicting.
Dependencies required for the experiment should be placed in `setup.py`.

`utils.py` usually shouldn't need modification, though it doesn't hurt.
`experiments.py` should run the experiment with the given parameters when the script is ran (`if __name__ == "__main__":...`).

### General Features:
- A python script for running an experiment.
- A python requirements file to automatically install dependencies.
- A bash script for running experiment batches.