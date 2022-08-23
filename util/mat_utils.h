#ifndef _MAT_UTILS_H_
#define _MAT_UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>

#ifndef __cplusplus
extern "C" {
#endif

void transpose(const cv::Mat& simg, cv::Mat& dimg, std::vector<int> swap = {2, 0, 1});
void concatenate(const cv::Mat& simg0, const cv::Mat& simg1, cv::Mat& dimg, int axis);
void reshape_channels_as_dimension(const cv::Mat& simg, cv::Mat& dimg);
void expand_dims(const cv::Mat& simg, cv::Mat& dimg, int axis);
void print_shape(const cv::Mat& img, const char* prefix = nullptr, const char* suffix = "\n");

#ifndef __cplusplus
}
#endif

#endif
