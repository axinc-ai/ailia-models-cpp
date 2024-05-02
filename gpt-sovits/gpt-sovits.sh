#!/bin/bash

MODEL="gpt-sovits"
FILE1="cnhubert.onnx"
FILE2="t2s_encoder.onnx"
FILE3="t2s_fsdec.onnx"
FILE4="t2s_sdec.onnx"
FILE5="vits.onnx"

#download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading onnx file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1} -o ${FILE1}
    fi
    if [ ! -e ${FILE2} ]; then
        echo "Downloading onnx file... save path: ${FILE2}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2} -o ${FILE2}
    fi
    if [ ! -e ${FILE3} ]; then
        echo "Downloading onnx file... save path: ${FILE3}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE3} -o ${FILE3}
    fi
    if [ ! -e ${FILE4} ]; then
        echo "Downloading onnx file... save path: ${FILE4}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE4} -o ${FILE4}
    fi
    if [ ! -e ${FILE5} ]; then
        echo "Downloading onnx file... save path: ${FILE5}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE5} -o ${FILE5}
    fi
    echo "ONNX file are prepared!"
fi
#execute
./${MODEL} $*
