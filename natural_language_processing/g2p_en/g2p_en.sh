#!/bin/bash

MODEL="g2p_en"
FILE1="g2p_encoder.onnx"
FILE2="g2p_encoder.decoder"
FILE3="cmudict"
FILE4="homographs.en"

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
    echo "ONNX file are prepared!"
fi
#execute
./${MODEL} $*
