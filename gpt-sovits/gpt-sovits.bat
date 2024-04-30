@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=gpt-sovits
set FILE1=nahida_cnhubert.onnx
set FILE2=nahida_t2s_encoder.onnx
set FILE3=nahida_t2s_fsdec.onnx
set FILE4=nahida_t2s_sdec.onnx
set FILE5=nahida_vits.onnx

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
    echo ONNX file are prepared^^!
)
rem execute
.\%MODEL%.exe %*
