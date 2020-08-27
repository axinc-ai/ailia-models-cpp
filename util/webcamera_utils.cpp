#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>


int adjust_frame_size(cv::Mat& sframe, cv::Mat& dframe, int d_width, int d_height)
{
    int s_width  = sframe.cols;
    int s_height = sframe.rows;

    float scale = std::max<float>((float)s_width/(float)d_width, (float)s_height/(float)d_height);

    // padding base
    cv::Mat dframe0(std::round(scale*(float)d_height), std::round(scale*(float)d_width), CV_8UC3, cv::Scalar(0, 0, 0));

    int start_x = (dframe0.cols - s_width)  / 2;
    int start_y = (dframe0.rows - s_height) / 2;
    cv::Rect roi(start_x, start_y, s_width, s_height);

    cv::Mat dframe0_roi = dframe0(roi);
    sframe.copyTo(dframe0_roi);

    cv::resize(dframe0, dframe, cv::Size(d_width, d_height), 0, 0);

    return 0;
}


int adjust_frame_size(cv::Mat& sframe, cv::Mat& dframe0, cv::Mat& dframe, int d_width, int d_height)
{
    int s_width  = sframe.cols;
    int s_height = sframe.rows;

    float scale = std::max<float>((float)s_width/(float)d_width, (float)s_height/(float)d_height);

    // padding base
    dframe0 = cv::Mat(std::round(scale*(float)d_height), std::round(scale*(float)d_width), CV_8UC3, cv::Scalar(0, 0, 0));

    int start_x = (dframe0.cols - s_width)  / 2;
    int start_y = (dframe0.rows - s_height) / 2;
    cv::Rect roi(start_x, start_y, s_width, s_height);

    cv::Mat dframe0_roi = dframe0(roi);
    sframe.copyTo(dframe0_roi);

    cv::resize(dframe0, dframe, cv::Size(d_width, d_height), 0, 0);

    return 0;
}
