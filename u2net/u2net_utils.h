#ifndef _U2NET_UTILS_H_
#define _U2NET_UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>

#ifndef __cplusplus
extern "C" {
#endif

void transform(cv::Mat simg, cv::Mat& dimg, cv::Size scaled_size);
int  load_image(cv::Mat& image, cv::Size& src_size, const char* path, cv::Size scaled_size);
int  save_result(cv::Mat pred, const char* path, cv::Size src_size);

#ifndef __cplusplus
}
#endif

#endif
