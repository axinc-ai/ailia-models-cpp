# ailia-models-cpp

The collection of pre-trained, state-of-the-art models for C++.

[ailia models (Python version)](https://github.com/axinc-ai/ailia-models)

## About ailia SDK

ailia SDK is a cross-platform high speed inference SDK. The ailia SDK provides a consistent C++ API on Windows, Mac, Linux, iOS, Android and Jetson. It supports Unity, Python and JNI for efficient AI implementation. The ailia SDK makes great use of the GPU via Vulkan and Metal to serve accelerated computing.

## Install ailia SDK

### Download ailia SDK

You can download a free evaluation version that allows you to evaluate the ailia SDK for 30 days. Please download from the trial link below.

https://ailia.jp/en/

### Install ailia SDK

Copy the files located in the folder [ailia SDK]/library/ to the folder ./ailia/library/.

### Install dependent libraries

#### Windows

gnumake and Visual Studio 2015 or newer are required.
http://gnuwin32.sourceforge.net/packages/make.htm

#### Mac

Xcode Commandline Tools are required, they can be installed by running the command.

```
xcode-select --install
```

OpenCV is also required, it can be installed by running the command.

```
brew install opencv.
```

#### Linux

OpenCV is also required, it can be installed by running the command.
```
apt install libopencv-dev
```

### Build

```
cd yolox
export AILIA_LIBRARY_PATH=../ailia/library
cmake .
make
```

### Run

```
# model download
./yolox.sh
# run model
./yolox -v 0
```

# Supporting Models

We are now converting to C++. Please wait to complete conversion.

## Image classification

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [resnet50](/resnet50/) | [Deep Residual Learning for Image Recognition]( https://github.com/KaimingHe/deep-residual-networks) | Chainer | 1.2.0 and later |

## Image segmentation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [U-2-Net](/u2net/) | [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net) | Pytorch | 1.2.2 and later |

## Object detection

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [yolov3-tiny](/yolov3-tiny/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later |
| [yolov3-face](/yolov3-face/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later |
| [m2det](/m2det/) | [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://github.com/qijiezhao/M2Det) | Pytorch | 1.2.3 and later |
| [yolox](/yolox/) | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Pytorch | 1.2.6 and later |

## Pose estimation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[lightweight-human-pose-estimation](/lightweight-human-pose-estimation/) | [Fast and accurate human pose estimation in PyTorch. Contains implementation of "Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose" paper.](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) | Pytorch | 1.2.1 and later |

## Face recognition

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[arcface](/arcface/) | [pytorch implement of arcface](https://github.com/ronghuaiyang/arcface-pytorch) | Pytorch | 1.2.1 and later |

