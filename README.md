# Pytorch implementation of [Yolo V3](https://pjreddie.com/media/files/papers/YOLOv3.pdf).

## Introduction
This project is a pytorch implementation of Yolo v3, aimed to replicate the [Darknet](https://github.com/pjreddie/darknet) implementation. Recently, there are a number of implementations:
- [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3), developed based on Keras + Numpy
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3), developed based on Pytorch + Numpy, loss not coverge.
- [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3), inference only, developed based on Pytorch + Numpy.
- [BobLiu20/YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch), Pytorch + Numpy, load pytorch pretrained model, loss does not converge now.

However, for Pytorch implementations, no one can replicate the performance of original darknet implementation. This project's goal is to benchmark the Yolo v3 in pytorch.

## Benchmarking

## Preparation

### prerequisites

### Data Preparation

### Pretrained Model

## Train

## Test

## Demo

## Citation
```
@article{redmon2018yolov3,
  title={Yolov3: An incremental improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1804.02767},
  year={2018}
}
```

