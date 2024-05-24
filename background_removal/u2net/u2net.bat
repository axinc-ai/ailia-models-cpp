@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=u2net
set FILE1=%MODEL%.onnx
set FILE2=%MODEL%.onnx.prototxt
set ARCH=large
set OPSET=10

if "%~1" == "-h"     goto execute
if "%~1" == "--help" goto execute

rem arch opset loop
:arch_opset_loop
if "%~1" == ""        goto download
if "%~1" == "-a"      goto set_arch
if "%~1" == "--arch"  goto set_arch
if "%~1" == "-o"      goto set_opset
if "%~1" == "--opset" goto set_opset
shift
goto arch_opset_loop

:set_arch
shift
if "%~1" == "" goto execute
if "%~1" == "small" (
    set ARCH=small
) else if "%~1" == "large" (
    set ARCH=large
) else (
    goto execute
)
shift
goto arch_opset_loop

:set_opset
shift
if "%~1" == "" goto execute
if "%~1" == "10" (
    set OPSET=10
) else if "%~1" == "11" (
    set OPSET=11
) else (
    goto execute
)
shift
goto arch_opset_loop

rem download
:download
if %ARCH% == small (
    if %OPSET% == 10 (
        set FILE1=%MODEL%p.onnx
        set FILE2=%MODEL%p.onnx.prototxt
    ) else (
        set FILE1=%MODEL%p_opset11.onnx
        set FILE2=%MODEL%p_opset11.onnx.prototxt
    )
) else (
    if %OPSET% == 10 (
        set FILE1=%MODEL%.onnx
        set FILE2=%MODEL%.onnx.prototxt
    ) else (
        set FILE1=%MODEL%_opset11.onnx
        set FILE2=%MODEL%_opset11.onnx.prototxt
    )
)

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
