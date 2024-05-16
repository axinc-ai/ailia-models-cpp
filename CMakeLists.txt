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

set(AILIA_LIBRARY_PATH "Path for ailia SDK library" ../ailia/library)
set(AILIA_TOKENIZER_PATH "Path for ailia Tokenizer library" ../ailia_tokenizer/library)
set(AILIA_AUDIO_PATH "Path for ailia Audio library" ../ailia_audio/library)
set(AILIA_SPEECH_PATH "Path for ailia Speech library" ../ailia_speech/library)

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
