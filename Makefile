ifndef OSTYPE
OSTYPE=Windows
#OSTYPE=Mac
endif

ifeq ($(OSTYPE),Windows)
#require msvc
COMPILER=cl
OPTION=/EHsc /I ./include /I ./util
LIBS=ailia.lib
LIBS_POSE=ailia_pose_estimate.lib
EXT=.exe
else
ifeq ($(OSTYPE),Linux)
#require opencv (apt install libopencv-dev)
COMPILER=g++
OPTION=-I ./util -I ./include -I /usr/include/opencv -std=c++11 -Wl,-rpath,./
LIBS=libailia.so -lopencv_imgproc -lopencv_imgcodecs -lopencv_core
LIBS_POSE=libailia_pose_estimate.so
EXT=
else
#require opencv (brew install opencv)
COMPILER=clang++
OPTION=-I ./util -I ./include -I /usr/local/include/opencv4/ -stdlib=libc++ -std=c++11 -Wl,-rpath,./
LIBS=libailia.dylib /usr/local/lib/libopencv_core.dylib /usr/local/lib/libopencv_imgproc.dylib /usr/local/lib/libopencv_imgcodecs.dylib
LIBS_POSE=libailia_pose_estimate.dylib
EXT=
endif
endif

all:
	$(COMPILER) yolov3-tiny/yolov3-tiny.cpp $(LIBS) $(OPTION) -o yolov3-tiny-app$(EXT)
