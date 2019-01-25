# MVA Object Recognition and Computer Vision Final Project

This repository is a implementation of the paper [Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images (ECCV2018)](https://arxiv.org/pdf/1804.01654.pdf) in Pytorch. It aims at reproducing the results obtained by the paper. The original implementation is in Tensorflow and available [here](https://github.com/nywang16/Pixel2Mesh).

## Required Dependencies

* Python3.0+ with Numpy
* Pytorch (Version 1.0+)
* [PyTorch Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/index.html)

## Optional Dependencies
* Matplotlib (for visualisation)
* CUDA 9.0

## Dataset

The dataset for running `train.py` is available at the following link: https://drive.google.com/file/d/1Z8gt4HdPujBNFABYrthhau9VZW10WWYe/view?usp=sharing
```
tar -xzf ShapeNetTrain.tar
```

The split between train and test for airplanes can be found in ```train_list.txt``` and ```test_list.txt```.

## Scripts

### Training

Two arguments are necessary for running the training script:
```
python3 train.py --data folder/to/data --experiment folder/to/store/models
```

### Evaluating
```
python3 evaluate.py --data folder/to/data --load model_file/to/load/
```

### Demoing
You can run a trained model on a folder of images for generating meshes. `./demo_img` is an example.
```
python3 demo.py --data folder/to/images --load model_file/to/load/ --output folder/to/store/meshes
```
The generated meshes can be visualized using [Meshlab](http://www.meshlab.net/).

### Optional Arguments

Description about optional arguments for each script (```train.py```, ```evaluate.py```, ```demo.py```) can be found by running the command (for example):
```
python3 evaluate.py --help
```
