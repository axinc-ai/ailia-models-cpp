@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=m2det
set FILE1=%MODEL%.onnx
set FILE2=%MODEL%.onnx.prototxt

rem download
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE1% (
        echo Downloading onnx file... ^(save path: %FILE1%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE1% -o %FILE1%
        rem bitsadmin /RawReturn /TRANSFER getfile https://storage.googleapis.com/ailia-models/%MODEL%/%FILE1% %CD%/%FILE1%
    )
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE2% (
        echo Downloading onnx file... ^(save path: %FILE2%^)
        curl https://storage.googleapis.com/ailia-models/%MODEL%/%FILE2% -o %FILE2%
        rem bitsadmin /RawReturn /TRANSFER getfile https://storage.googleapis.com/ailia-models/%MODEL%/%FILE2% %CD%/%FILE2%
    )
    echo ONNX file and Prototxt file are prepared^^!
)

rem execute
.\%MODEL%.exe %*
