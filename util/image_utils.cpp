#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "mat_utils.h"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif


int normalize_image(const cv::Mat& simg, cv::Mat& dimg, std::string normalize_type)
{
    if (normalize_type == "255" || normalize_type == "127.5" || normalize_type == "ImageNet") {
        if (normalize_type == "ImageNet") {
            if (simg.rows > 0 && simg.channels() != 3) {
                return -1;
            }
            else if (simg.rows < 0 && simg.channels() != 3) {
                if (simg.channels() != 1 || simg.size[simg.dims-1] != 3) {
                    return -1;
                }
            }
        }
        int size = 1, chan = 1;
        if (simg.rows > 0) {
            dimg = cv::Mat(simg.rows, simg.cols, CV_MAKETYPE(CV_32F, simg.channels()));
            if (normalize_type == "ImageNet") {
                size = simg.rows*simg.cols;
                chan = simg.channels();
            }
            else {
                size = simg.rows*simg.cols*simg.channels();
                chan = 1;
            }
        }
        else {
            dimg = cv::Mat(simg.dims, simg.size, CV_MAKETYPE(CV_32F, simg.channels()));
            if (normalize_type == "ImageNet") {
                for (int i = 0; i < simg.dims-1; i++) {
                    size *= simg.size[i];
                }
                if (simg.channels() == 3) {
                    size *= simg.size[simg.dims-1];
                }
                chan = 3;
            }
            else {
                size = simg.channels();
                for (int i = 0; i < simg.dims; i++) {
                    size *= simg.size[i];
                }
                chan = 1;
            }
        }

        unsigned char* sdata = (unsigned char*)simg.data;
        float*         ddata = (float*)dimg.data;

        if (normalize_type == "255") {
            for (int i = 0; i < size; i++) {
                float col = sdata[i];
                ddata[i] = col / 255.0f;
            }
        }
        else if (normalize_type == "127.5") {
            for (int i = 0; i < size; i++) {
                float col = sdata[i];
                ddata[i] = col / 127.5f - 1.0f;
            }
        }
        else if (normalize_type == "ImageNet") {
            float mean[] = {0.485f, 0.456f, 0.406f};
            float std[]  = {0.229f, 0.224f, 0.225f};
            for (int i = 0; i < size; i++) {
                for (int c = 0; c < chan; c++) {
                    float col = sdata[i*chan+c];
                    ddata[i*chan+c] = (col/255.0f-mean[c])/std[c];
                }
            }
        }
    }
    else {
//        dimg = simg.clone();
        simg.copyTo(dimg);
    }

    return 0;
}


int load_image(cv::Mat& img, const char* path, cv::Size shape,
               bool rgb, std::string normalize_type, bool gen_input_ailia)
{
    cv::Mat oimg = cv::imread(path, (int)rgb);
    if (oimg.empty()) {
        PRINT_ERR("\'%s\' not found\n", path);
        return -1;
    }

    cv::Mat mimg0, mimg1, mimg2;
    if (rgb) {
        cv::cvtColor(oimg, mimg0, cv::COLOR_BGR2RGB);
    }
    else {
//        mimg0 = oimg.clone();
        oimg.copyTo(mimg0);
    }
    int status = normalize_image(mimg0, mimg1, normalize_type);
    if (status < 0) {
        return -1;
    }
    cv::resize(mimg1, mimg2, shape);

    if (gen_input_ailia) {
        if (rgb) {
            transpose(mimg2, img, {2, 0, 1});
        }
        else {
            mimg2.copyTo(img);
        }
    }
    else {
        mimg2.copyTo(img);
    }

    return 0;
}
