@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=t5_whisper_medical
set FILE1=t5_whisper_medical-decoder-with-lm-head.obf.onnx
set FILE2=t5_whisper_medical-decoder-with-lm-head.onnx.prototxt
set FILE3=t5_whisper_medical-encoder.obf.onnx
set FILE4=t5_whisper_medical-encoder.onnx.prototxt
set FILE5=spiece.model

rem download
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
    echo SPM files are prepared^^!
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE5% (
        echo Downloading spm file... ^(save path: %FILE5%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE5% -o %FILE5%
    )
    echo SPM files are prepared^^!
)
rem execute
.\%MODEL%.exe %*
