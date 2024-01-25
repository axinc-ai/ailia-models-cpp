@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=clap
set FILE1=CLAP_audio_LAION-Audio-630K_with_fusion.onnx
set FILE2=CLAP_audio_LAION-Audio-630K_with_fusion.onnx.prototxt
set FILE3=CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx
set FILE4=CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx.prototxt
set FILE5=CLAP_text_text_branch_RobertaModel_roberta-base.onnx
set FILE6=CLAP_text_text_branch_RobertaModel_roberta-base.onnx.prototxt

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
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE5% (
        echo Downloading onnx file... ^(save path: %FILE5%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE5% -o %FILE5%
    )
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE6% (
        echo Downloading onnx file... ^(save path: %FILE6%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE6% -o %FILE6%
    )
    echo ONNX file and Prototxt file are prepared^^!
)
rem execute
:execute
.\%MODEL%.exe %*
