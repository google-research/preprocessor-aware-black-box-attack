# Preprocessors Matter! Realistic Decision-BasedAttacks on Machine Learning Systems

Chawin Sitawarin, Florian Tramèr, Nicholas Carlini

## Abstract

Decision-based adversarial attacks construct inputs that fool a machine-learning model into making targeted mispredictions by making only hard-label queries. For the most part, these attacks have been applied directly to isolated neural network models. However, in practice, machine learning models are just a component of a much larger system. By adding just a single preprocessor in front of a classifier, we find that state-of-the-art query-based attacks are as much as seven times less effective at attacking a prediction pipeline than attacking the machine learning model alone. Hence, attacks that are unaware of this invariance inevitably waste a large number of queries to re-discover or overcome it. We, therefore, develop techniques to first reverse-engineer the preprocessor and then use this extracted information to attack the end-to-end system. Our extraction method requires only a few hundred queries to learn the preprocessors used by most publicly available model pipelines, and our preprocessor-aware attacks recover the same efficacy as just attacking the model alone.

## Setup

To install dependencies,

```[bash]
# conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
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
```

Load ImageNet validation set with blurred face

```[bash]
wget https://image-net.org/data/ILSVRC/blurred/val_blurred.tar.gz
```

## Usage

Coming soon!

## Preprocessor Extraction

Coming soon!

## Disclaimer

This is not an officially supported Google project.
