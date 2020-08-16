#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>


int adjust_frame_size(cv::Mat& sframe, cv::Mat& dframe, int d_width, int d_height)
{
    int s_width  = sframe.cols;
    int s_height = sframe.rows;

    float scale = std::max<float>((float)s_width/(float)d_width, (float)s_height/(float)d_height);

    // padding base
    cv::Mat img(std::round(scale*(float)d_width), std::round(scale*(float)d_height), CV_8UC3, cv::Scalar(0, 0, 0));

    int start_x = (img.cols - s_width)  / 2;
    int start_y = (img.rows - s_height) / 2;
    cv::Rect roi(start_x, start_y, s_width, s_height);

    cv::Mat img_roi = img(roi);
    sframe.copyTo(img_roi);

    cv::resize(img, dframe, cv::Size(d_width, d_height), 0, 0);

    return 0;
}
