#!/bin/bash

MODEL="g2p_en"
FILE1="g2p_encoder.onnx"
FILE2="g2p_decoder.onnx"
FILE3="cmudict"
FILE4="homographs.en"
FILE5="averaged_perceptron_tagger_classes.txt"
FILE6="averaged_perceptron_tagger_tagdict.txt"
FILE7="averaged_perceptron_tagger_weights.txt"

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
    if [ ! -e ${FILE6} ]; then
        echo "Downloading onnx file... save path: ${FILE6}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE6} -o ${FILE6}
    fi
    if [ ! -e ${FILE7} ]; then
        echo "Downloading onnx file... save path: ${FILE7}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE7} -o ${FILE7}
    fi
    echo "ONNX file are prepared!"
fi
#execute
./${MODEL} $*
