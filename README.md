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

### ModelNet40
### ShapeNet
### S3DIS
### Semantic8
### NPM3D
