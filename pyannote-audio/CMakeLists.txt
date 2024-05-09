cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME pyannote-audio)
set(SRC_FILES 
    ${PROJECT_NAME}.cpp
    ./pyannote_audio_utils/audio/pipeline/speaker_diarization.cpp
    ./pyannote_audio_utils/audio/pipeline/utils/diarization.cpp
    ./pyannote_audio_utils/core/annotation.cpp
    ./pyannote_audio_utils/core/utils/types.cpp
    ./pyannote_audio_utils/core/segment.cpp
    ./pyannote_audio_utils/core/timeline.cpp
)

# AILIA ライブラリのパスを環境変数から取得
set(AILIA_LIBRARY_PATH $ENV{AILIA_LIBRARY_PATH})

# インクルードディレクトリを追加
include_directories(
    ${AILIA_LIBRARY_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/pyannote_audio_utils/audio/pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/pyannote_audio_utils/audio/pipeline/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/pyannote_audio_utils/core
    ${CMAKE_CURRENT_SOURCE_DIR}/pyannote_audio_utils/core/utils
)

# ライブラリのパスを設定
if(WIN32)
    set(LIBRARY_PATH ${AILIA_LIBRARY_PATH}/windows/x64)
elseif(APPLE)
    set(LIBRARY_PATH ${AILIA_LIBRARY_PATH}/mac)
else()
    set(LIBRARY_PATH ${AILIA_LIBRARY_PATH}/linux)
endif()

message("Include directories: ${INCLUDE_PATH}")
message("Library path: ${LIBRARY_PATH}")

project(${PROJECT_NAME} CXX)

# ライブラリディレクトリをリンク
link_directories(${LIBRARY_PATH})

add_executable(${PROJECT_NAME} ${SRC_FILES})
# target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)
