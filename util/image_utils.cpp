#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "image_utils.h"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif


static void transpose(cv::Mat& simg, cv::Mat& dimg,
                      std::vector<int> swap = {2, 0, 1})
{
    std::vector<int> size0 = {simg.rows, simg.cols, simg.channels()};
    std::vector<int> size1 = {size0[swap[0]], size0[swap[1]], size0[swap[2]]};
    dimg = cv::Mat_<unsigned char>(size1.size(), &size1[0]);

    unsigned char* sdata = (unsigned char*)simg.data;
    unsigned char* ddata = (unsigned char*)dimg.data;
    int sd[3] = {0, 0, 0};
    for (int d0 = 0; d0 < size1[0]; d0++) {
        sd[swap[0]] = d0;
        for (int d1 = 0; d1 < size1[1] ; d1++) {
            sd[swap[1]] = d1;
            for (int d2 = 0; d2 < size1[2]; d2++) {
                sd[swap[2]] = d2;
                ddata[d0*size1[1]*size1[2]+d1*size1[2]+d2] = sdata[sd[0]*size0[1]*size0[2]+sd[1]*size0[2]+sd[2]];
            }
        }
    }

    return;
}


static void normalize_image(cv::Mat& simg, cv::Mat& dimg, std::string normalize_type)
{
    if (normalize_type == "255" || normalize_type == "127.5" ||
        normalize_type == "ImageNet") {
        dimg = cv::Mat_<float>(simg.rows, simg.cols, CV_MAKETYPE(CV_32F, simg.channels()));
        unsigned char* sdata = (unsigned char*)simg.data;
        float*         ddata = (float*)dimg.data;

        if (normalize_type == "255") {
            for (int i = 0; i < simg.rows*simg.cols; i++) {
                for (int c = 0; c < simg.channels(); c++) {
                    float col = sdata[i*simg.channels()+c];
                    ddata[i*simg.channels()+c] = col / 255.0f;
                }
            }
        }
        else if (normalize_type == "127.5") {
            for (int i = 0; i < simg.rows*simg.cols; i++) {
                for (int c = 0; c < simg.channels(); c++) {
                    float col = sdata[i*simg.channels()+c];
                    ddata[i*simg.channels()+c] = col / 127.5f - 1.0f;
                }
            }
        }
        else if (normalize_type == "ImageNet") {
            float mean[] = {0.485f, 0.456f, 0.406f};
            float std[]  = {0.229f, 0.224f, 0.225f};
            for (int i = 0; i < simg.rows*simg.cols; i++) {
                for (int c = 0; c < simg.channels(); c++) {
                    float col = sdata[i*simg.channels()+c];
                    ddata[i*simg.channels()+c] = (col/255.0f-mean[c])/std[c];
                }
            }
        }
    }
    else {
//        dimg = simg.clone();
        simg.copyTo(dimg);
    }
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
    normalize_image(mimg0, mimg1, normalize_type);
    cv::resize(mimg1, mimg2, shape);

    if (gen_input_ailia) {
        if (rgb) {
            transpose(mimg2, img, {2, 0, 1});
            int newsize[] = {1, mimg2.size[0], mimg2.size[1], mimg2.size[2]};
            img.reshape(1, 4, newsize);
        }
        else {
            mimg2.copyTo(img);
            int newsize[] = {1, 1, mimg2.size[0], mimg2.size[1]};
            img.reshape(1, 4, newsize);
        }
    }
    else {
        mimg2.copyTo(img);
    }

    return 0;
}
