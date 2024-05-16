# ailia-models-cpp

The collection of pre-trained, state-of-the-art models for C++.

[ailia models (Python version)](https://github.com/axinc-ai/ailia-models)

## About ailia SDK

ailia SDK is a cross-platform high speed inference SDK. The ailia SDK provides a consistent C++ API on Windows, Mac, Linux, iOS, Android and Jetson. It supports Unity, Python and JNI for efficient AI implementation. The ailia SDK makes great use of the GPU via Vulkan and Metal to serve accelerated computing.

## Install ailia SDK

### Download ailia SDK

The ailia SDK can be installed from a git submodule.

```
git subomdule init
git submodule update
```

### Download license file

Download the ailia SDK license file with the following command. The license file is valid for one month.

```
cd ailia
python3 download_license.py
```

Alternatively, the license file can be obtained by requesting the evaluation version of the ailia SDK.

[Request trial version](https://axinc.jp/en/trial/)

### Install dependent libraries

#### Windows

gnumake and Visual Studio 2015 or newer are required.
http://gnuwin32.sourceforge.net/packages/make.htm

#### Mac

Xcode Commandline Tools are required, they can be installed by running the command.

```
xcode-select --install
```

OpenCV is required, it can be installed by running the command.

```
brew install opencv.
```

#### Linux

OpenCV is required, it can be installed by running the command.
```
apt install libopencv-dev
```

### Build

```
cmake .
make
```

### Run

```
cd yolox
./yolox.sh -v 0
```

# Supporting Models

We are now converting to C++. Please wait to complete conversion.

## Audio processing

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [silero-vad](/silero-vad/) | [Silero VAD](https://github.com/snakers4/silero-vad) | Pytorch | 1.2.15 and later | |
| [clap](/clap/) | [CLAP](https://github.com/LAION-AI/CLAP) | Pytorch | 1.3.0 and later | |

## Image classification

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [resnet50](/resnet50/) | [Deep Residual Learning for Image Recognition]( https://github.com/KaimingHe/deep-residual-networks) | Chainer | 1.2.0 and later |
| [clip](/clip/) | [CLIP](https://github.com/openai/CLIP) | Pytorch | 1.2.9 and later |

## Image segmentation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [U-2-Net](/u2net/) | [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net) | Pytorch | 1.2.2 and later |

## Object detection

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [yolov3-tiny](/yolov3-tiny/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later |
| [m2det](/m2det/) | [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://github.com/qijiezhao/M2Det) | Pytorch | 1.2.3 and later |
| [yolox](/yolox/) | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Pytorch | 1.2.6 and later |

## Pose estimation

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[lightweight-human-pose-estimation](/lightweight-human-pose-estimation/) | [Fast and accurate human pose estimation in PyTorch. Contains implementation of "Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose" paper.](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) | Pytorch | 1.2.1 and later |

## Face detection

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [yolov3-face](/yolov3-face/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later |

## Face identification

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[arcface](/arcface/) | [pytorch implement of arcface](https://github.com/ronghuaiyang/arcface-pytorch) | Pytorch | 1.2.1 and later |

## Face recognition

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[face_alignment](/face_alignment/)| [2D and 3D Face alignment library build using pytorch](https://github.com/1adrianb/face-alignment) | Pytorch | 1.2.1 and later |
|[mediapipe_iris](/mediapipe_iris/) | [irislandmarks.pytorch](https://github.com/cedriclmenard/irislandmarks.pytorch) | Pytorch | 1.2.2 and later |

## Natural language processing

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
|[fugumt-en-ja](/fugumt-en-ja) | [Fugu-Machine Translator](https://github.com/s-taka/fugumt) | Pytorch | 1.2.9 and later |
|[fugumt-ja-en](/fugumt-ja-en) | [Fugu-Machine Translator](https://github.com/s-taka/fugumt) | Pytorch | 1.2.10 and later |
|[bert_maskedlm](/bert_maskedlm) | [huggingface/transformers](https://github.com/huggingface/transformers) | Pytorch | 1.2.5 and later |
|[sentence_transformers](/sentence_transformers) | [sentence transformers](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) | Pytorch | 1.2.7 and later |
|[t5_whisper_medical](/t5_whisper_medical) | error correction of medical terms using t5 | Pytorch | 1.2.13 and later | |
