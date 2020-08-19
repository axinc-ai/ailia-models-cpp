#ifndef _WEBCAMERA_UTILS_H_
#define _WEBCAMERA_UTILS_H_

#include <opencv2/opencv.hpp>

#ifndef __cplusplus
extern "C" {
#endif

int adjust_frame_size(cv::Mat& sframe, cv::Mat& dframe, int width, int height);

#ifndef __cplusplus
}
#endif

#endif
