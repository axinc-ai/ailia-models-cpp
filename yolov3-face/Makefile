TARGET = yolov3-face
SRCS   = yolov3-face.cpp

ifeq ($(OS),Windows_NT)

## Windows ##
AILIA_PATH  = ..\ailia\library
OPENCV_PATH = C:\tools\opencv
SRCS += ..\util\utils.cpp \
        ..\util\detector_utils.cpp \
        ..\util\webcamera_utils.cpp
CXX    = cl
CFLAGS = /O2 \
         /I$(AILIA_PATH)\include \
         /I..\util \
         /I$(OPENCV_PATH)\sources\include \
         /I$(OPENCV_PATH)\build\include
LIBS   = $(AILIA_PATH)\windows\x64\ailia.lib Wininet.lib \
         $(OPENCV_PATH)\build\x64\vc15\lib\opencv_world3410.lib
.PHONY: $(TARGET)
$(TARGET): $(SRCS)
	$(CXX) $(SRCS) $(CFLAGS) /link $(LIBS) /OUT:$@.exe
.PHONY: clean
clean:
	@if exist $(TARGET).exe del $(TARGET).exe *.obj
## end of Windows ##

else

## Linux MacOS ##
AILIA_PATH = ../ailia/library
SRCS  += ../util/utils.cpp \
         ../util/detector_utils.cpp \
         ../util/webcamera_utils.cpp
CXX    = clang++
CFLAGS = -O2 -Wall -std=c++11 -I$(AILIA_PATH)/include -I../util
LIBS   = -lcurl
ifeq ($(shell uname),Darwin)
# MacOS
OPENCV_PATH = /usr/local/Cellar/opencv/4.4.0
CFLAGS += -I$(OPENCV_PATH)/include/opencv4
LIBS   += $(AILIA_PATH)/mac/libailia.dylib \
          $(OPENCV_PATH)/lib/libopencv_core.dylib \
          $(OPENCV_PATH)/lib/libopencv_highgui.dylib \
          $(OPENCV_PATH)/lib/libopencv_imgproc.dylib \
          $(OPENCV_PATH)/lib/libopencv_imgcodecs.dylib \
          $(OPENCV_PATH)/lib/libopencv_videoio.dylib
else
# Linux
LIBS   += $(AILIA_PATH)/linux/libailia.so \
          -lopencv_core \
          -lopencv_highgui \
          -lopencv_imgproc \
          -lopencv_imgcodecs \
          -lopencv_videoio
endif
.PHONY: $(TARGET)
$(TARGET): $(SRCS)
	$(CXX) $(SRCS) $(CFLAGS) $(LIBS) -o $@
.PHONY: clean
clean:
	@if [ -e $(TARGET) ]; then rm $(TARGET) *.o ; fi
## end of Linux MacOS ##

endif
