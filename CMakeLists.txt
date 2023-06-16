#*******************************************************************
#
#    DESCRIPTION:
#      AILIA C++ SAMPLE BUILD SCRIPT
#    COPYRIGHT:
#      AXELL CORPORATION, ax Inc.
#
#******************************************************************/

cmake_minimum_required(VERSION 3.10)
project(samples C CXX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(arcface)
add_subdirectory(bert_maskedlm)
add_subdirectory(clip)
add_subdirectory(face_alignment)
add_subdirectory(fugumt)
add_subdirectory(lightweight-human-pose-estimation)
add_subdirectory(m2det)
add_subdirectory(mediapipe_iris)
add_subdirectory(resnet50)
add_subdirectory(sentence_transformers)
add_subdirectory(u2net)
add_subdirectory(yolov3-face)
add_subdirectory(yolov3-tiny)
add_subdirectory(yolox)
