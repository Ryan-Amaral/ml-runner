# ml-runner

### What is it?
A general framework/codebase for running machine learning experiments. 
Primarilly evolution based reinforcement learning, but could be used for anything.

### Some Rules:

Different branches will be for different experiment sets. 
Some may follow a general template (as in master), others could be completely different. 
A general template will be followed where possible for efficient reuse.

Each branch should follow a common naming convention `{experiment name}-{experiment subtype}`, for example `2dbiped-tpg`.

It is recommended to use a different environment for each experiment, as typically each experiment may have different dependencies, some conflicting.

### General Features:
- A python script for running an experiment.
- A python requirements file to automatically install dependencies.
- A bash script for running experiment batches.