# Pytorch implementation of [Yolo V3](https://pjreddie.com/media/files/papers/YOLOv3.pdf).

## Introduction
This project is a pytorch implementation of Yolo v3, aimed to replicate the [Darknet](https://github.com/pjreddie/darknet) implementation. Recently, there are a number of implementations:
- [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3), developed based on Keras + Numpy
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3), developed based on Pytorch + Numpy, loss does not coverge.
- [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3), inference only, developed based on Pytorch + Numpy.
- [BobLiu20/YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch), Pytorch + Numpy, load pytorch pretrained model, loss does not converge now.

However, for Pytorch implementations, no one can replicate the performance of original darknet implementation. This project's goal is to benchmark the Yolo v3 in pytorch.

## Benchmarking

## Preparation
First of all, clone the code
```
git clone https://github.com/jiasenlu/YOLOv3.pytorch.git
```

Then, create a folder:

```
cd  YOLOv3.pytorch && mkdir data
```
### prerequisites
- Pytorch 3.6
- Pytorch 0.5 (latest)
- TorchVision
- TensorBoard
- CUDA 8.0 or higher

### Data Preparation
* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/.

* **COCO**: Please also follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data.

* **Visual Genome**: Please follow the instructions in [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) to prepare Visual Genome dataset. You need to download the images and object annotation files first, and then perform proprecessing to obtain the vocabulary and cleansed annotations based on the scripts provided in this repository.

### Pretrained Model

We use converted pytorch pretrained model, you can place the pretrained model under `data/weights`.

- Darknet53: [GoogleDrive](https://drive.google.com/open?id=1VYwHUznM3jLD7ftmOSCHnpkVpBJcFIOA)
- Resnet101:
- Resnet50:
- MobelNet:

### Compilation
Compile the cuda dependencies using following simple commands:

```
sh make.sh
```

## Train
```
 python main.py
```
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