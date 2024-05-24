@echo off
setlocal enabledelayedexpansion
cd %~dp0

set MODEL=arcface
set URL_ROOT=https://storage.googleapis.com/ailia-models
set ARCH=%MODEL%
set FILE1=%ARCH%.onnx
set FILE2=%ARCH%.onnx.prototxt

set FACE_ARCH=yolov3-face
set FACE_OPT=.opt
set FACE_FILE1=%FACE_ARCH%%FACE_OPT%.onnx
set FACE_FILE2=%FACE_ARCH%%FACE_OPT%.onnx.prototxt

set VIDEO=0

if "%~1" == "-h"     goto execute
if "%~1" == "--help" goto execute

rem arch face loop
:arch_face_loop
if "%~1" == ""       goto download
if "%~1" == "-a"     goto set_arch
if "%~1" == "--arch" goto set_arch
if "%~1" == "-f"     goto set_face_arch
if "%~1" == "--face" goto set_face_arch
if "%~1" == "-v" (
    set VIDEO=1
)
if "%~1" == "--video" (
    set VIDEO=1
)
shift
goto arch_face_loop

:set_arch
shift
if "%~1" == "" goto execute
if "%~1" == "arcface" (
    set ARCH=arcface
) else if "%~1" == "arcface_mixed_90_82" (
    set ARCH=arcface_mixed_90_82
) else if "%~1" == "arcface_mixed_90_99" (
    set ARCH=arcface_mixed_90_99
) else if "%~1" == "arcface_mixed_eq_90_89" (
    set ARCH=arcface_mixed_eq_90_89
) else (
    goto execute
)
shift
goto arch_face_loop

:set_face_arch
shift
if "%~1" == "" goto execute
if "%~1" == "yolov3" (
    set FACE_ARCH=yolov3-face
    set FACE_OPT=.opt
) else if "%~1" == "blazeface" (
    set FACE_ARCH=blazeface
    set FACE_OPT=
) else (
    goto execute
)
shift
goto arch_face_loop

rem download
:download
set FILE1=%ARCH%.onnx
set FILE2=%ARCH%.onnx.prototxt
set FACE_FILE1=%FACE_ARCH%%FACE_OPT%.onnx
set FACE_FILE2=%FACE_ARCH%%FACE_OPT%.onnx.prototxt

if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE1% (
        echo Downloading onnx file... ^(save path: %FILE1%^)
        curl %URL_ROOT%/%MODEL%/%FILE1% -o %FILE1%
        rem bitsadmin /RawReturn /TRANSFER getfile %URL_ROOT%/%MODEL%/%FILE1% %CD%/%FILE1%
    )
)
if not "%1" == "-h" if not "%1" == "--help" (
    if not exist %FILE2% (
        echo Downloading onnx file... ^(save path: %FILE2%^)
        curl %URL_ROOT%/%MODEL%/%FILE2% -o %FILE2%
        rem bitsadmin /RawReturn /TRANSFER getfile %URL_ROOT%/%MODEL%/%FILE2% %CD%/%FILE2%
    )
    echo ONNX file and Prototxt file are prepared^^!
)
if not "%1" == "-h" if not "%1" == "--help" if %VIDEO% == 1 (
    if not exist %FACE_FILE1% (
        echo Downloading onnx file... ^(save path: %FACE_FILE1%^)
        curl %URL_ROOT%/%FACE_ARCH%/%FACE_FILE1% -o %FACE_FILE1%
        rem bitsadmin /RawReturn /TRANSFER getfile %URL_ROOT%/%FCAE_ARCH%/%FACE_FILE1% %CD%/%FACE_FILE1%
    )
)
if not "%1" == "-h" if not "%1" == "--help" if %VIDEO% == 1  (
    if not exist %FACE_FILE2% (
        echo Downloading onnx file... ^(save path: %FACE_FILE2%^)
        curl %URL_ROOT%/%FACE_ARCH%/%FACE_FILE2% -o %FACE_FILE2%
        rem bitsadmin /RawReturn /TRANSFER getfile %URL_ROOT%/%FCAE_ARCH%/%FACE_FILE2% %CD%/%FACE_FILE2%
    )
    echo ONNX file and Prototxt file are prepared^^!
)

rem execute
:execute
.\%MODEL%.exe %*
