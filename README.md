# [FKAConv](https://arxiv.org/abs/2004.04462)

![FKAConv products](./doc/predictions.png)

## Paper: FKAConv, Feature-Kernel Alignment for Point Cloud Convolution

Please consider consider citing our ACCV 2020 paper:
```
@inproceedings{boulch2020fka,
  title={{FKAConv: Feature-Kernel Alignment for Point Cloud Convolution}},
  author={Boulch, Alexandre and Puy, Gilles and Marlet, Renaud},
  booktitle={15th Asian Conference on Computer Vision (ACCV 2020)},
  year={2020}
}
```

## Dependencies

FKAConv code is actually included in the [LightConvPoint framework repository](https://github.com/valeoai/LightConvPoint).

It is aimed to change in a near future.

LightConvPoint shall be the base library for point cloud processing while FKAConv shall became the repository dedicated to this specific work.

## Installation

```
pip install -ve /path/to/fkaconv/repository/
```

## Dataset preparation

Dataset are prepared according to the LightConvPoint data preparation.
We use the provided dataloaders.

### Examples

We provide examples classification and segmentation datasets:

The training scripts use [Sacred](https://sacred.readthedocs.io/) to manage options.
The options can be managed by modifying the 'config.yaml' files or with the `with` statement in command line (e.g., `python train.py with training.batchsize=16`).

### ModelNet40 and ShapeNet

In order to train the model, you must modify the `config.yaml` with the right path for the data (dataset-->dir) and the path where to save the results (training-->savedir).
Then, run the folowing commands:
```
cd examples/modelnet40/
python train.py
```

To evalutate the model on the test set of ModelNet40:
```
python test.py -c path_to_save_dir/config.yaml --iter 16 --batchsize 64
```
The `--iter (-i)` (the number of iterations per shape) and `--batchsize (-b)` (the batchsize to be used at test time) are optional. If not provided, their value are the one defined in the `config.yaml` file.

### S3DIS

In order to train the model, you must modify the `config.yaml` with the right path for the data (dataset-->dir) and the path where to save the results (training-->savedir) and validation area.
Then, run the folowing commands:
```
cd examples/s3dis/
python train.py
```

To evaluate the model on the validation set (this is the `val_area` parameter set at the training phase):
```
python test.py -c path_to_save_dir/config.yaml --step 0.2 --savepts
```
The `--step (-s)` (the step of the test sliding window) and `--savepts` (save the points and labels) are optional. If not provided, their value are the one defined in the `config.yaml` file.

### Semantic8

In order to train the model, you must modify the `config.yaml` with the right path for the data (dataset-->dir) and the path where to save the results (training-->savedir).
Then, run the folowing commands:
```
cd examples/semantic8/
python train.py
```

To evaluate the model on the test set of Semantic8, modify the test path parameter in the `config.yaml` file saved in the save directory and run:
```
python test.py -c path_to_save_dir/config.yaml --step 0.8 --savepts
```
The `--step (-s)` (the step of the test sliding window) and `--savepts` (save the points and labels) are optional. If not provided, their value are the one defined in the `config.yaml` file.

### NPM3D

In order to train the model, you must modify the `config.yaml` with the right path for the data (dataset-->dir) and the path where to save the results (training-->savedir).
Then, run the folowing commands:
```
cd examples/npm3d/
python train.py
```

To evaluate the model on the test set of Semantic8, modify the test path parameter in the `config.yaml` file saved in the save directory and run:
```
python test.py -c path_to_save_dir/config.yaml --step 0.8 --savepts
```
The `--step (-s)` (the step of the test sliding window) and `--savepts` (save the points and labels) are optional. If not provided, their value are the one defined in the `config.yaml` file.

## Training/Testing with fusion model

Once two segmentation models have been trained (one with color information and one without), you can train a fusion model.
To do so, we use the ``config_fusion.yaml` file.
The path to fusion model subdirectory must be filled in this file.

Then train the model:
```
python train.py with config_fusion.yaml
```

The test procedure is the same as for single networks.