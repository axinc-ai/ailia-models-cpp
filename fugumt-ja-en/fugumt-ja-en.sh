#!/bin/bash

MODEL="fugumt-ja-en"
FILE1='encoder_model.onnx'
FILE2='encoder_model.onnx.prototxt'
FILE3='decoder_model.onnx'
FILE4='decoder_model.onnx.prototxt'
FILE5="source.spm"
FILE6="target.spm"

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
        echo "Downloading spm file... save path: ${FILE3}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE3} -o ${FILE3}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE4} ]; then
        echo "Downloading spm file... save path: ${FILE4}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE4} -o ${FILE4}
    fi
    echo "SPM files are prepared!"
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE5} ]; then
        echo "Downloading spm file... save path: ${FILE5}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE5} -o ${FILE5} #日本語と英語を入れ替えたいから、ファイル名（source.spmとtarget.spm）を入れ替える
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE6} ]; then
        echo "Downloading spm file... save path: ${FILE6}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE6} -o ${FILE6} #日本語と英語を入れ替えたいから、ファイル名（source.spmとtarget.spm）を入れ替える
    fi
    echo "SPM files are prepared!"
fi
#execute
./${MODEL} $*
