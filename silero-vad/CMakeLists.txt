﻿cmake_minimum_required(VERSION 3.1)

set (PROJECT_NAME silero-vad)
set (SRC_FILES ${PROJECT_NAME}.cpp ../util/wave_reader.cpp)
set (INCLUDE_PATH $ENV{AILIA_LIBRARY_PATH}/include ../util)
if(WIN32)
    set (LIBRARY_PATH $ENV{AILIA_LIBRARY_PATH}/windows/x64)
elseif(APPLE)
    set (LIBRARY_PATH $ENV{AILIA_LIBRARY_PATH}/mac)
else(WIN32)
    set (LIBRARY_PATH $ENV{AILIA_LIBRARY_PATH}/linux)
endif(WIN32)
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message(${INCLUDE_PATH})
message(${LIBRARY_PATH})

project(${PROJECT_NAME} CXX)

include_directories(${INCLUDE_PATH})
link_directories(${LIBRARY_PATH})

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_link_libraries(${PROJECT_NAME} ailia)
set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)
