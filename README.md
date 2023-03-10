# blob_net

**Keywords:** Python, Rheology, Segmentation of suspensions, PyTorch, Deep Learning

## Table of contents
1. [ Introduction ](#1-introduction)
2. [ Data Organization ](#2-data-organization)
3. [ Scripts ](#3-scripts)  
4. [ How to use the code ](#4-how-to)

## 1. Introduction

This repository contains the Python code used for the implementation of the BlobNet algorithm. The code allows to:

- generate synthetic images in conjonction with a ground truth in order to train the convolutional network without using any annotated image
- train the neural network architecture on the synthetic images
- apply the network to actual images obtained during the experiments

5 experimental images were annotated manually in order to evaluate the performance of the algorithm and are available in the repository

### Requirements

The code relies upon the following Python libraries:
- numpy
- matplotlib
- scikit-image
- scipy
- pytorch

To install the project:
```bash
# clone the project
cd /path/to/project/
git clone https://github.com/bruno-figliuzzi/blob_net.git
cd blob_net/
# install requirements
```

## 2. Data Organization

- `dataset/imgs/`: images of the suspension 
- `dataset/img_truths/`: annotated images
- `dataset/train/`: training set of synthesized images (only 40 images have been kept)
- `dataset/val/`: validation set of synthesized images (only 10 images have been kept)
- `dataset/labels/`: labels for the training/validation images
- `results/`: segmentation results for the suspension images
- `weights/`: weights of the trained neural network architecture

Note:

- all images are PNG files
- the annotations are numpy files.
- the annotations (in dataset/img_truths/) are images-like numpy array where each particle appears with a specific label


## 3. Scripts

1. Training set generation

- `generation.py`: Synthetic images generation

2. Segmentation

- `network.py`: Neural network implementation
- `dataset.py`: Methods used by the neural network to handle the dataset
- `train.py`: Neural network training
- `segmentation.py`: Segmentation of the images of the database
- `metrics.py`: Segmentation metrics

3. Other

- `kmeans.py`: Segmentation with a K-means algorithm


## 4. How to use the code


1. How to generate synthetic images for the training set:

Edit "generation.py" and run:
~$ python generation.py

2. How to train the neural network:

Edit "train.py" and run:
~$ python train.py

3. How to perform the segmentation of an experimental image with the 
neural network architecture

Edit "segmentation.py" and run:
~$ python segmentation.py
computes the segmentation of all images from the folder dataset/imgs/




