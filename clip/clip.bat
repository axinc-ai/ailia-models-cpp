@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=clip
set FILE1=ViT-B32-encode_image.onnx
set FILE2=ViT-B32-encode_image.onnx.prototxt
set FILE3=ViT-B32-encode_text.onnx
set FILE4=ViT-B32-encode_text.onnx.prototxt

rem download
:download
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE1% (
        echo Downloading onnx file... ^(save path: %FILE1%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE1% -o %FILE1%
    )
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE2% (
        echo Downloading onnx file... ^(save path: %FILE2%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE2% -o %FILE2%
    )
    echo ONNX file and Prototxt file are prepared^^!
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE3% (
        echo Downloading onnx file... ^(save path: %FILE3%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE3% -o %FILE3%
    )
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE4% (
        echo Downloading onnx file... ^(save path: %FILE4%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE4% -o %FILE4%
    )
    echo ONNX file and Prototxt file are prepared^^!
)
rem execute
:execute
.\%MODEL%.exe %*
