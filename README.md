The official repository for our paper [The Dual Form of Neural Networks Revisited: Connecting Test Time Predictions to Training Patterns via Spotlights of Attention](https://arxiv.org/abs/2202.05798).


## Installation

This project requires Python 3 and PyTorch 1.5.

```bash
pip3 install -r requirements.txt
```

Create a Weights and Biases account and run 
```bash
wandb login
```

More information on setting up Weights and Biases can be found on
https://docs.wandb.com/quickstart.

For plotting, LaTeX is required (to avoid Type 3 fonts and to render symbols). Installation is OS specific.

## Usage

The code makes use of Weights and Biases for experiment tracking. In the "sweeps" directory, we provide sweep configurations for all experiments we have performed. The sweeps are officially meant for hyperparameter optimization, but we use them to run 10 instances of each experiment.

To reproduce our results, start a sweep for each of the YAML files in the "sweeps" directory. Run wandb agent for each of them in the main directory. This will run all the experiments, and they will be displayed on the W&B dashboard.
### Re-creating plots from the paper

Edit config file "paper/ff_as_attention/config.json". Enter your project name in the field "wandb_project" (e.g. "username/ff_as_attention").

Run the script of interest within the "paper/ff_as_attention" directory. For example:

```bash
cd paper/ff_as_attention
python3 cifar10_ff_attention.py
```

The output will be generated in the "paper/ff_as_attention/out/" directory.
