#ifndef _BLAZEFACE_UTILS_H_
#define _BLAZEFACE_UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>

#ifndef __cplusplus
extern "C" {
#endif

int blazeface_postprocess(const cv::Mat& raw_box, const cv::Mat& raw_score, std::vector<cv::Mat>& detections);

#ifndef __cplusplus
}
#endif

#endif
