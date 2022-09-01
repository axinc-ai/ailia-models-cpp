@echo off
setlocal enabledelayedexpansion
cd %~dp0

if "%~1" == "-h"     goto execute
if "%~1" == "--help" goto execute

:download
for %%m in (blazeface facemesh iris) do (
    set FILE1=%%m.opt.onnx
    set FILE2=%%m.opt.onnx.prototxt

    if "%%m" == "iris" (
        set MODEL=mediapipe_iris
    ) else (
        set MODEL=%%m
    )

    if not exist !FILE1! (
        echo Downloading onnx file... ^(save path: !FILE1!^)
        curl https://storage.googleapis.com/ailia-models/!MODEL!/!FILE1! -o !FILE1!
    )
)

echo ONNX files and Prototxt files are prepared^^!

:execute
.\%MODEL%.exe %*
