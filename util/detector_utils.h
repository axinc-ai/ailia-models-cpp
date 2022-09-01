#ifndef _DETECTOR_UTILS_H_
#define _DETECTOR_UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include "ailia_detector.h"

#ifndef __cplusplus
extern "C" {
#endif

int load_image(cv::Mat& img, const char* path);
cv::Scalar hsv_to_rgb(int h, int s, int v);
int plot_result(AILIADetector* detector, cv::Mat& img, const std::vector<const char*> category, bool logging = true);

#ifndef __cplusplus
}
#endif

#endif
