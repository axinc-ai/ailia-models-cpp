cmake_minimum_required(VERSION 3.1)

set (PROJECT_NAME clap)
set (SRC_FILES ${PROJECT_NAME}.cpp)
set (SRC_FILES ${SRC_FILES} ../util/utils.cpp)
set (INCLUDE_PATH $ENV{AILIA_LIBRARY_PATH}/include $ENV{AILIA_TOKENIZER_PATH}/include ../util)
if(WIN32)
    set (LIBRARY_PATH $ENV{AILIA_LIBRARY_PATH}/windows/x64 $ENV{AILIA_TOKENIZER_PATH}/windows/x64)
elseif(APPLE)
    set (LIBRARY_PATH $ENV{AILIA_LIBRARY_PATH}/mac $ENV{AILIA_TOKENIZER_PATH}/mac)
else(WIN32)
    set (LIBRARY_PATH $ENV{AILIA_LIBRARY_PATH}/linux $ENV{AILIA_TOKENIZER_PATH}/linux)
endif(WIN32)
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

project(${PROJECT_NAME} CXX)

include_directories(${INCLUDE_PATH})
link_directories(${LIBRARY_PATH})

# erase later
include_directories(../include)
link_directories(../lib)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_link_libraries(${PROJECT_NAME} ailia ailia_tokenizer)
set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)
