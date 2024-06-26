@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=g2p_en
set FILE1=g2p_encoder.onnx
set FILE2=g2p_decoder.onnx
set FILE3=cmudict
set FILE4=homographs.en
set FILE5=averaged_perceptron_tagger_classes.txt
set FILE6=averaged_perceptron_tagger_tagdict.txt
set FILE7=averaged_perceptron_tagger_weights.txt

rem download
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE1% (
        echo Downloading onnx file... ^(save path: %FILE1%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE1% -o %FILE1%
    )
    if not exist %FILE2% (
        echo Downloading onnx file... ^(save path: %FILE2%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE2% -o %FILE2%
    )
    if not exist %FILE3% (
        echo Downloading onnx file... ^(save path: %FILE3%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE3% -o %FILE3%
    )
    if not exist %FILE4% (
        echo Downloading onnx file... ^(save path: %FILE4%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE4% -o %FILE4%
    )
    if not exist %FILE5% (
        echo Downloading onnx file... ^(save path: %FILE5%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE5% -o %FILE5%
    )
    if not exist %FILE6% (
        echo Downloading onnx file... ^(save path: %FILE6%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE6% -o %FILE6%
    )
    if not exist %FILE7% (
        echo Downloading onnx file... ^(save path: %FILE7%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE7% -o %FILE7%
    )
    echo ONNX file are prepared^^!
)
rem execute
.\%MODEL%.exe %*
