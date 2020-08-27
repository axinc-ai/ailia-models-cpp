#ifndef _IMAGE_UTILS_H_
#define _IMAGE_UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>

#ifndef __cplusplus
extern "C" {
#endif


int load_image(cv::Mat& img, const char* path, cv::Size shape,
               bool rgb = true, std::string normalize_type = {"255"}, bool gen_input_ailia = false);

#ifndef __cplusplus
}
#endif

#endif
