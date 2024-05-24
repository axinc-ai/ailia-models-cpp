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

set(AILIA_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/ailia/library)
set(AILIA_TOKENIZER_PATH ${CMAKE_SOURCE_DIR}/ailia_tokenizer/library)
set(AILIA_AUDIO_PATH ${CMAKE_SOURCE_DIR}/ailia_audio/library)
set(AILIA_SPEECH_PATH ${CMAKE_SOURCE_DIR}/ailia_speech/library)

set (INCLUDE_PATH ${AILIA_LIBRARY_PATH}/include ${AILIA_AUDIO_PATH}/include ${AILIA_TOKENIZER_PATH}/include ${AILIA_SPEECH_PATH}/include ../../util)
if(WIN32)
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/windows/x64 ${AILIA_AUDIO_PATH}/windows/x64 ${AILIA_TOKENIZER_PATH}/windows/x64 ${AILIA_SPEECH_PATH}/windows/x64)
elseif(APPLE)
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/mac ${AILIA_AUDIO_PATH}/mac ${AILIA_TOKENIZER_PATH}/mac ${AILIA_SPEECH_PATH}/mac)
elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/jetson ${AILIA_AUDIO_PATH}/jetson ${AILIA_TOKENIZER_PATH}/jetson ${AILIA_SPEECH_PATH}/jetson)
elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/raspberrypi ${AILIA_AUDIO_PATH}/raspberrypi ${AILIA_TOKENIZER_PATH}/raspberrypi ${AILIA_SPEECH_PATH}/raspberrypi)
else(UNIX)
    set (LIBRARY_PATH ${AILIA_LIBRARY_PATH}/linux ${AILIA_AUDIO_PATH}/linux ${AILIA_TOKENIZER_PATH}/linux ${AILIA_SPEECH_PATH}/linux)
endif(WIN32)

add_subdirectory(audio_processing/clap)
add_subdirectory(audio_processing/silero-vad)
add_subdirectory(audio_processing/whisper)

add_subdirectory(background_removal/u2net)

add_subdirectory(image_classification/resnet50)
add_subdirectory(image_classification/clip)

add_subdirectory(face_detection/yolov3-face)

add_subdirectory(face_identification/arcface)

add_subdirectory(face_recognition/face_alignment)
add_subdirectory(face_recognition/mediapipe_iris)

add_subdirectory(natural_language_processing/bert_maskedlm)
add_subdirectory(natural_language_processing/fugumt-ja-en)
add_subdirectory(natural_language_processing/fugumt-en-ja)
add_subdirectory(natural_language_processing/sentence_transformers)
add_subdirectory(natural_language_processing/t5_whisper_medical)

add_subdirectory(object_detection/yolov3-tiny)
add_subdirectory(object_detection/yolox)
add_subdirectory(object_detection/m2det)

add_subdirectory(pose_estimation/lightweight-human-pose-estimation)
