#!/bin/bash

MODEL="retinaface"
FILE1="${MODEL}_resnet50.onnx"
FILE2="${MODEL}_resnet50.onnx.prototxt"
FILE3="${MODEL}_mobile0.25.onnx"
FILE4="${MODEL}_mobile0.25.onnx.prototxt"

#download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading onnx file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1} -o ${FILE1}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE2} ]; then
        echo "Downloading onnx file... save path: ${FILE2}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2} -o ${FILE2}
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE3} ]; then
        echo "Downloading onnx file... save path: ${FILE3}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE3} -o ${FILE3}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE4} ]; then
        echo "Downloading onnx file... save path: ${FILE4}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE4} -o ${FILE4}
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi
#execute
./${MODEL} $*
