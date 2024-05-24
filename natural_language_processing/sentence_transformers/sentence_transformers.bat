@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=sentence-transformers-japanese
set EXE=sentence-transformers
set FILE1=paraphrase-multilingual-mpnet-base-v2.onnx
set FILE2=paraphrase-multilingual-mpnet-base-v2.onnx.prototxt
set FILE3=sentencepiece.bpe.model

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
        echo Downloading vocab file... ^(save path: %FILE3%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE3% -o %FILE3%
    )
)
rem execute
.\%EXE%.exe %*
