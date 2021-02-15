#ifndef _WEBCAMERA_UTILS_H_
#define _WEBCAMERA_UTILS_H_

#include <opencv2/opencv.hpp>

#ifndef __cplusplus
extern "C" {
#endif

int adjust_frame_size(const cv::Mat& sframe, cv::Mat& dframe, int width, int height);
int adjust_frame_size(const cv::Mat& sframe, cv::Mat& dframe0, cv::Mat& dframe, int d_width, int d_height);
int preprocess_frame(const cv::Mat& sframe, cv::Mat& dframe0, cv::Mat& dframe, int d_width, int d_height,
                     bool rgb = true, std::string normalize_type = {"255"});
int get_writer(cv::VideoWriter& writer, const char* path, cv::Size size, bool rgb = true);

#ifndef __cplusplus
}
#endif

#endif
