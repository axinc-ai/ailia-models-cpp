#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif


void transpose(const cv::Mat& simg, cv::Mat& dimg, std::vector<int> swap = {2, 0, 1})
{
    std::vector<int> size0;
    if (simg.rows > 0) {
         size0 = {simg.rows, simg.cols, simg.channels()};
    }
    else {
        size0 = {simg.size[0], simg.size[1], simg.size[2]};
    }
    std::vector<int> size1 = {size0[swap[0]], size0[swap[1]], size0[swap[2]]};
    dimg = cv::Mat(size1.size(), &size1[0], simg.type());

    if (simg.elemSize1() == sizeof(char)) {
        char* sdata = (char*)simg.data;
        char* ddata = (char*)dimg.data;
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
    }
    else if (simg.elemSize1() == sizeof(short)) {
        short* sdata = (short*)simg.data;
        short* ddata = (short*)dimg.data;
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
    }
    else {
        int* sdata = (int*)simg.data;
        int* ddata = (int*)dimg.data;
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
    }

    return;
}


int concatenate(const cv::Mat& simg0, const cv::Mat& simg1, cv::Mat& dimg, int axis)
{
    if (simg0.dims != simg1.dims || simg0.rows*simg1.rows < 0 ||
        simg0.type() != simg0.type() || simg0.elemSize1() != simg1.elemSize1() ||
        axis > simg0.dims) {
        PRINT_ERR("concatenate failed\n");
        return -1;
    }

    bool ndarray = false;
    if (simg0.rows < 0) {
        ndarray = true;
    }

    int dims = simg0.dims + 1;
    int* size0 = new int[dims];
    int* size1 = new int[dims];
    if (ndarray) {
        for (int i = 0; i < dims-1; i++) {
            size0[i] = simg0.size[i];
            size1[i] = simg1.size[i];
        }
        size0[dims-1] = simg0.channels();
        size1[dims-1] = simg1.channels();
    }
    else {
        size0[0] = simg0.rows;
        size1[0] = simg1.rows;
        size0[1] = simg0.cols;
        size1[1] = simg1.cols;
        size0[2] = simg0.channels();
        size1[2] = simg1.channels();
    }

    int outer = 1, inner = 1;
    std::vector<int> new_shape;
    int d;
    for (d = 0; d < axis; d++) {
        if (size0[d] != size1[d]) {
            PRINT_ERR("concatenate failed\n");
            return -1;
        }
        new_shape.push_back(size0[d]);
        outer *= size0[d];
    }
    new_shape.push_back(size0[axis] + size1[axis]);
    int med = size0[axis] + size1[axis];
    int med0 = size0[axis];
    int med1 = size1[axis];
    for (d = d + 1; d < dims; d++) {
        if (size0[d] != size1[d]) {
            PRINT_ERR("concatenate failed\n");
            return -1;
        }
        new_shape.push_back(size0[d]);
        inner *= size0[d];
    }

    if (ndarray) {
        dimg = cv::Mat(new_shape.size(), &new_shape[0], simg0.type());
    }
    else {
        dimg = cv::Mat(new_shape[0], new_shape[1], simg0.type());
    }
    if (simg0.elemSize1() == sizeof(char)) {
        for (int o = 0; o < outer; o++) {
            int m = 0;
            for (int m0 = 0; m0 < size0[axis]; m0++, m++) {
                 char* sdata = (char*)simg0.data+(med0*o+m0)*inner;
                 char* ddata = (char*)dimg.data +(med*o+m)*inner;
                 for (int i = 0; i < inner; i++) {
                     ddata[i] = sdata[i];
                 }
            }
            for (int m1 = 0; m1 < size1[axis]; m1++, m++) {
                 char* sdata = (char*)simg1.data+(med1*o+m1)*inner;
                 char* ddata = (char*)dimg.data +(med*o+m)*inner;
                 for (int i = 0; i < inner; i++) {
                     ddata[i] = sdata[i];
                 }
            }
        }
    }
    else if (simg0.elemSize1() == sizeof(short)) {
        for (int o = 0; o < outer; o++) {
            int m = 0;
            for (int m0 = 0; m0 < size0[axis]; m0++, m++) {
                 short* sdata = (short*)simg0.data+(med0*o+m0)*inner;
                 short* ddata = (short*)dimg.data +(med*o+m)*inner;
                 for (int i = 0; i < inner; i++) {
                     ddata[i] = sdata[i];
                 }
            }
            for (int m1 = 0; m1 < size1[axis]; m1++, m++) {
                 short* sdata = (short*)simg1.data+(med1*o+m1)*inner;
                 short* ddata = (short*)dimg.data +(med*o+m)*inner;
                 for (int i = 0; i < inner; i++) {
                     ddata[i] = sdata[i];
                 }
            }
        }
    }
    else {
        for (int o = 0; o < outer; o++) {
            int m = 0;
            for (int m0 = 0; m0 < size0[axis]; m0++, m++) {
                 int* sdata = (int*)simg0.data+(med0*o+m0)*inner;
                 int* ddata = (int*)dimg.data +(med*o+m)*inner;
                 for (int i = 0; i < inner; i++) {
                     ddata[i] = sdata[i];
                 }
            }
            for (int m1 = 0; m1 < size1[axis]; m1++, m++) {
                 int* sdata = (int*)simg1.data+(med1*o+m1)*inner;
                 int* ddata = (int*)dimg.data +(med*o+m)*inner;
                 for (int i = 0; i < inner; i++) {
                     ddata[i] = sdata[i];
                 }
            }
        }
    }

    return 0;
}
