#!/bin/bash
# CONDA_ENV_NAME=prep-python3.10

# conda create -n $CONDA_ENV_NAME python=3.10
# conda activate $CONDA_ENV_NAME
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y scipy pandas scikit-learn pip
conda upgrade -y numpy scipy pandas scikit-learn
conda install -y -c conda-forge scikit-image timm

# BayesOpt Attack dependencies
# For older CUDA version, there might be an error from gpytorch
# If this happens, try pip install gpytorch==1.4 botorch==0.4
conda install botorch -c pytorch -c gpytorch -y

# tensorflow is only used by adversarial-robustness-toolbox
pip install foolbox kornia adversarial-robustness-toolbox[pytorch] tensorflow
pip install git+https://github.com/fra31/auto-attack
# Flag --no-deps is important here to prevent reinstall pytorch on pip
pip install torchjpeg compressai --no-deps