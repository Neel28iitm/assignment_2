# CNN from Scratch

This repo contains a PyTorch implementation for training a custom CNN on a subset of the iNaturalist/CIFAR-10 dataset.

## Features

- 5 Conv → Activation → MaxPool blocks
- Customizable filters, kernel size, and dense layers
- wandb logging for:
  - Training & validation loss/accuracy
  - Filter visualizations
- Modular code for ease of extension

 ## Training

```bash
pip install -r requirements.txt
python train.py

