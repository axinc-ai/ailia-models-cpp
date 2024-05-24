#!/bin/bash

MODEL="clap"
FILE1="CLAP_audio_LAION-Audio-630K_with_fusion.onnx"
FILE2="CLAP_audio_LAION-Audio-630K_with_fusion.onnx.prototxt"
FILE3="CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx"
FILE4="CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx.prototxt"
FILE5="CLAP_text_text_branch_RobertaModel_roberta-base.onnx"
FILE6="CLAP_text_text_branch_RobertaModel_roberta-base.onnx.prototxt"

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
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE5} ]; then
        echo "Downloading onnx file... save path: ${FILE5}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE5} -o ${FILE5}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    if [ ! -e ${FILE6} ]; then
        echo "Downloading onnx file... save path: ${FILE6}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE6} -o ${FILE6}
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi
#execute
./${MODEL} $*
