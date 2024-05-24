#!/bin/bash

MODEL="bert_maskedlm"
FILE1="bert-base-japanese-whole-word-masking.onnx"
FILE2="bert-base-japanese-whole-word-masking.onnx.prototxt"
FILE3="vocab_wordpiece.txt"
FILE4="ipadic.zip"

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
        echo "Downloading vocab file... save path: ${FILE3}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE3} -o ${FILE3}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE4} ]; then
        echo "Downloading dic file... save path: ${FILE4}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE4} -o ${FILE4}
    fi
    unzip ${FILE4}
    echo "Dic files are prepared!"
fi
#execute
./${MODEL} $*
