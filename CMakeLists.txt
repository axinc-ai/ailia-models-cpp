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

set(AILIA_LIBRARY_PATH ../ailia/library)
set(AILIA_TOKENIZER_PATH ../ailia_tokenizer/library)
set(AILIA_AUDIO_PATH  ../ailia_audio/library)
set(AILIA_SPEECH_PATH ../ailia_speech/library)

set (INCLUDE_PATH ${AILIA_LIBRARY_PATH}/include ${AILIA_TOKENIZER_PATH}/include ${AILIA_AUDIO_PATH}/include ../util)
if(WIN32)
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/windows/x64 ${AILIA_TOKENIZER_PATH}/windows/x64 ${AILIA_AUDIO_PATH}/windows/x64)
elseif(APPLE)
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/mac ${AILIA_TOKENIZER_PATH}/mac ${AILIA_AUDIO_PATH}/mac)
else(WIN32)
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/linux ${AILIA_TOKENIZER_PATH}/linux ${AILIA_AUDIO_PATH}/linux)
endif(WIN32)

add_subdirectory(arcface)
add_subdirectory(bert_maskedlm)
add_subdirectory(clip)
add_subdirectory(clap)
add_subdirectory(face_alignment)
add_subdirectory(fugumt-ja-en)
add_subdirectory(fugumt-en-ja)
add_subdirectory(lightweight-human-pose-estimation)
add_subdirectory(m2det)
add_subdirectory(mediapipe_iris)
add_subdirectory(resnet50)
add_subdirectory(sentence_transformers)
add_subdirectory(u2net)
add_subdirectory(yolov3-face)
add_subdirectory(yolov3-tiny)
add_subdirectory(yolox)
