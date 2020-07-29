# ailia-models-cpp

The collection of pre-trained, state-of-the-art models for C++.

[ailia models (Python version)](https://github.com/axinc-ai/ailia-models)

## About ailia SDK

ailia SDK is a cross-platform high speed inference SDK. The ailia SDK provides a consistent C++ API on Windows, Mac, Linux, iOS, Android and Jetson. It supports Unity, Python and JNI for efficient AI implementation. The ailia SDK makes great use of the GPU via Vulkan and Metal to serve accelerated computing.

You can download a free evaluation version that allows you to evaluate the ailia SDK for 30 days. Please download from the trial link below.

https://ailia.jp/en/

## Notice

This repository does not include ailia libraries.

So you must get license and import ailia libraries to Plugin folder.

## Develop Environment

- Windows, Mac
- MSVC2019, Xcode11

## Target Environment

- Windows, Mac, iOS, Android, Linux

# Supporting Models

We are now converting to C#. Please wait to complete conversion.

## Object detection

| Name | Detail | Exported From | Supported Ailia Version |
|:-----------|------------:|:------------:|:------------:|
| [yolov3-tiny](/yolov3-tiny/) | [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) | ONNX Runtime | 1.2.1 and later |
