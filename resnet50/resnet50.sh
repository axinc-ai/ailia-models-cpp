#!/bin/bash

MODEL="resnet50"
FILE1="${MODEL}.opt.onnx"
FILE2="${MODEL}.opt.onnx.prototxt"

status=0
for n in "$@"
do
    if [ "$n" = "-a" ] || [ "$n" = "--arch" ]; then
        status=1
    elif [ $status -eq 1 ]; then
        if [ "$n" = "${MODEL}" ]; then
            FILE1="${MODEL}.onnx"
            FILE2="${MODEL}.onnx.prototxt"
        elif [ "$n" = "${MODEL}.opt" ]; then
            FILE1="${MODEL}.opt.onnx"
            FILE2="${MODEL}.opt.onnx.prototxt"
        elif [ "$n" = "${MODEL}_pytorch" ]; then
            FILE1="${MODEL}_pytorch.onnx"
            FILE2="${MODEL}_pytorch.onnx.prototxt"
        else
            break;
        fi
        status=0
    else
        status=0
    fi
done

#download
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ] && [ $status -eq 0 ]; then
    if [ ! -e ${FILE1} ]; then
        echo "Downloading onnx file... save path: ${FILE1}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1} -o ${FILE1}
#        wget https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1}
    fi
fi
if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ] && [ $status -eq 0 ]; then
    if [ ! -e ${FILE2} ]; then
        echo "Downloading onnx file... save path: ${FILE2}"
        curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2} -o ${FILE2}
#        wget https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2}
    fi
    echo "ONNX file and Prototxt file are prepared!"
fi

#execute
./${MODEL} $*
