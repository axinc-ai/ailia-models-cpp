@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=resnet50
set FILE1=%MODEL%.opt.onnx
set FILE2=%MODEL%.opt.onnx.prototxt

if "%~1" == "-h"     goto execute
if "%~1" == "--help" goto execute

rem arch loop
:arch_loop
if "%~1" == ""       goto download
if "%~1" == "-a"     goto set_arch
if "%~1" == "--arch" goto set_arch
shift
goto arch_loop

:set_arch
shift
if "%~1" == "" goto execute
echo %~1
if "%~1" == "%MODEL%.opt" (
    set FILE1=%MODEL%.opt.onnx
    set FILE2=%MODEL%.opt.onnx.prototxt
) else if "%~1" == "%MODEL%" (
    set FILE1=%MODEL%.onnx
    set FILE2=%MODEL%.onnx.prototxt
) else if "%~1" == "%MODEL%_pytorch" (
    set FILE1=%MODEL%_pytorch.onnx
    set FILE2=%MODEL%_pytorch.onnx.prototxt
) else (
    goto execute
)
shift
goto arch_loop

rem download
:download
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
:execute
.\%MODEL%.exe %*
