@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=bert_maskedlm
set FILE1=bert-base-japanese-whole-word-masking.onnx
set FILE2=bert-base-japanese-whole-word-masking.onnx.prototxt
set FILE3=vocab_wordpiece.txt
set FILE4=ipadic.zip

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
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE4% (
        echo Downloading dict file... ^(save path: %FILE4%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE4% -o %FILE4%
    )
    call powershell -command "Expand-Archive %FILE4%"
    echo Dict files are prepared^^!
)
rem execute
.\%MODEL%.exe %*
