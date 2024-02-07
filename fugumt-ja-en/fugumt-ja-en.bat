@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=fugumt
set EXE_FILE=fugumt-ja-en
set FILE1=encoder_model.onnx
set FILE2=encoder_model.onnx.prototxt
set FILE3=decoder_model.onnx
set FILE4=decoder_model.onnx.prototxt
set FILE5=source.spm
set FILE6=target.spm

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
        echo Downloading spm file... ^(save path: %FILE3%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE3% -o %FILE3%
    )
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE4% (
        echo Downloading spm file... ^(save path: %FILE4%^)
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
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE6% (
        echo Downloading spm file... ^(save path: %FILE6%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE6% -o %FILE6%
    )
    echo SPM files are prepared^^!
)
rem execute
.\%EXE_FILE%.exe %*
