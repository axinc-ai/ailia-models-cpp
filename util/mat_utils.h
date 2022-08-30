#ifndef _MAT_UTILS_H_
#define _MAT_UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>

#ifndef __cplusplus
extern "C" {
#endif

void transpose(const cv::Mat& simg, cv::Mat& dimg, std::vector<int> swap = {2, 0, 1});
void concatenate(const cv::Mat& simg0, const cv::Mat& simg1, cv::Mat& dimg, int axis);
void expand_dims(const cv::Mat& simg, cv::Mat& dimg, int axis);
void print_shape(const cv::Mat& img, const char* prefix = nullptr, const char* suffix = "\n");
void reshape_channels_as_dimensions(const cv::Mat& simg, cv::Mat& dimg);
void reshape_dimensions_as_channels(const cv::Mat& simg, cv::Mat& dimg);

#ifndef __cplusplus
}
#endif

#endif
