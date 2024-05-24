#!/bin/bash

MODEL="m2det"
FILE1="${MODEL}.onnx"
FILE2="${MODEL}.onnx.prototxt"

#download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading onnx file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1} -o ${FILE1}
#        wget https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE2} ]; then
        echo "Downloading onnx file... save path: ${FILE2}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2} -o ${FILE2}
#        wget https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2}
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi

#execute
./${MODEL} $*
