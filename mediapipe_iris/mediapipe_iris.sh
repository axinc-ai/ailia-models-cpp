#!/bin/bash

if [ ! "$1" = "-h" ] && [ ! "$1" = "--help" ]; then
    for MODEL in "blazeface" "facemesh" "iris"
    do
        FILE1="${MODEL}_s.opt.onnx"
        FILE2="${MODEL}_s.opt.onnx.prototxt"

        if [ ! -e ${FILE1} ]; then
            echo "Downloading onnx file... save path: ${FILE1}"
            curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE1} -o ${FILE1}
        fi

        if [ ! -e ${FILE2} ]; then
            echo "Downloading onnx file... save path: ${FILE2}"
            curl https://storage.googleapis.com/ailia-models/${MODEL}/${FILE2} -o ${FILE2}
        fi
    done

    echo "ONNX file and Prototxt file are prepared!"
fi

./mediapipe_iris $*
