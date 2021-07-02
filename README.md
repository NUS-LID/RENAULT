
# Ensemble and Auxiliary Tasks for Data-Efficient Deep Reinforcement Learning

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

This repository is the official implementation of REN, RENAULT, and their variants for the ATARI games 100K experiments.

Our implementation is based on: <https://github.com/Kaixhin/Rainbow>.

## Setup

### Requirements
* Python 3.7.6
* PyTorch 1.5.1

#### Additional packages
```
atari-py==0.2.6       
joblib==0.15.1
numpy==1.18.5
opencv-python==4.2.0.34
Pillow==7.1.2
plotly==4.8.1
tensorboard==2.2.2
tqdm==4.46.1
```

Although the program should be able to run with different version of dependencies, we highly recommend you to use the exact version specified above.

### Installation
We recommend you to use [pyenv](https://github.com/pyenv/pyenv) to manage Python virtual environments. Please follow this documentation to install it: <https://github.com/pyenv/pyenv#installation>.

#### Creating a new virtualenv

First, install Python 3.7.6 using Pyenv:
```
pyenv install 3.7.6
```
To create a new environment named ``renault`` using the installed Python 3.7.6, execute the following command:
```
pyenv virtualenv 3.7.6 renault
```

Then activate the virtualenv:
```
source activate renault
```

#### Installing dependencies

After you have created and activated your virtualenv, you can install PyTorch by following <https://pytorch.org/get-started/previous-versions/#v151>. Make sure that you install the appropriate PyTorch-related packages according to your CUDA version.

For example, to install PyTorch with CUDA 10.2, you can execute:
```
pip install torch==1.5.1 torchvision==0.6.1
```

Then, you can install all dependencies by executing the following command:
```
pip install -r requirements.txt
```

Now you're ready to run the methods.

## Usage

This instruction assumes the existence of GPU with id 0. Multiple GPUs can also be used by specifying them in the ``--gpus`` argument, for example ``--gpus 0 1 2``. We use a random seed of 123 throughout the instruction. If you want to use other random seeds, then you can simply replace the argument ``--seed <random seed>``. The command line program can accept more than one game. For example, to run ``amidar`` and ``assault``, you set: ``--games amidar assault``. ``alien`` game is used only as an example. To get the list of the available games, you can execute:
```
python -c "import atari_py; print(atari_py.list_games())"
```

To see the result of a run, you can simply append ``--result`` after the original command. For example, after you run REN on ``alien`` with random seed 123, you can execute:
```
python benchmark.py --benchmark_id run1 --gpus 0 --rapid --options="--agent REN --seed 123" --games alien --result
```

### REN
```
python benchmark.py --benchmark_id run1 --gpus 0 --rapid --options="--agent REN --seed 123" --games alien 
```

### REN-J
```
python benchmark.py --benchmark_id run1 --gpus 0 --rapid --options="--agent REN-J --seed 123" --games alien
```

### RENAULT
```
python benchmark.py --benchmark_id run1 --gpus 0 --rapid --options="--agent RENAULT --seed 123 --auxs id cr ci lns mc" --games alien
```

Auxiliary tasks ``--auxs`` abbreviation mapping:
```
id   - Inverse dynamic
cr   - Categorical rewards prediction
lns  - Latent next state prediction
ci   - Total change of intensity prediction
mc   - Change of moment prediction
none - Placeholder auxiliary task that does nothing
```

#### RENAULT-all
```
HIDE_AUX_LOSS=1 python benchmark.py --benchmark_id run1 --gpus 0 --rapid --options="--agent RENAULT --seed 123 --auxs id+cr+ci+lns+mc id+cr+ci+lns+mc id+cr+ci+lns+mc id+cr+ci+lns+mc id+cr+ci+lns+mc --aux-aggregate mean" --games alien
```
``--aux-aggregate`` controls how the auxiliary tasks loss should be aggregated. ``+`` sign inside the ``--auxs`` is used to combine auxiliary tasks. ``HIDE_AUX_LOSS=1`` is used to suppress the auxiliary tasks loss (otherwise it'll be too long).

#### RENAULT-noID
```
python benchmark.py --benchmark_id run1 --gpus 0 --rapid --options="--agent RENAULT --seed 123 --auxs none cr ci lns mc" --games alien
```
