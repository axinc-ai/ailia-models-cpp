/*******************************************************************
*
*    DESCRIPTION:
*      GDI+ image loader sample
*    AUTHOR:
*      AXELL Corporation
*    DATE:2018/08/14
*
*******************************************************************/

#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include "opencv2/opencv.hpp"
#include "opencv2/core/types_c.h"
#else
#include <windows.h>
#include <gdiplus.h>
#pragma comment (lib, "gdiplus.lib")
#endif

struct GdiLoaderFileInfo{
	int Width;
	int Height;
};

int load_image(std::vector<unsigned char> &image,struct GdiLoaderFileInfo &info,const char *path){
#if defined(__APPLE__) || defined(__linux__)
	cv::Mat m = cv::imread(path, -1);
	if (m.empty()){
		printf("image not found\n");
		return -1;
	}
	int h = m.rows;
	int w = m.cols;
	int c = m.channels();
	info.Width  = w;
	info.Height = h;
	image.resize(w*h*4);
	unsigned char *data = (unsigned char *)m.data;
	int step = m.step;
	int i, j, k;

	for(i = 0; i < h; ++i){
		for(k = 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				image[i*w*4+j*4+k] = data[i*step+j*c+k];
			}
		}
	}

	return 0;
#else
	wchar_t     path_w[ MAX_PATH ];
	size_t      pathLength = 0;

	if( mbstowcs_s( &pathLength,&path_w[0],MAX_PATH,path,_TRUNCATE ) != 0 ) {
		return -1;
	}

	Gdiplus::GdiplusStartupInput    gdiplusStartupInput;
	ULONG_PTR                       gdiplusToken;

	if( Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL) !=  Gdiplus::Ok ) {

		return -1;
	}

	Gdiplus::Bitmap* pBitmap = Gdiplus::Bitmap::FromFile(path_w);
	bool file_open_failed=false;
	if( pBitmap && pBitmap->GetLastStatus() == Gdiplus::Ok ) {
		info.Width  = pBitmap->GetWidth();
		info.Height = pBitmap->GetHeight();

		image.resize(info.Width*info.Height*4);

		for( int y=0; y<info.Height; y++ ) {
			for( int x=0; x<info.Width; x++ ) {
				Gdiplus::Color  srcColor;
				pBitmap->GetPixel(x, y, &srcColor);

				image[y*info.Width*4+x*4+0] = srcColor.GetB();
				image[y*info.Width*4+x*4+1] = srcColor.GetG();
				image[y*info.Width*4+x*4+2] = srcColor.GetR();
				image[y*info.Width*4+x*4+3] = srcColor.GetA();
			}
		}
	}else{
		file_open_failed=true;
	}

	delete pBitmap;

	Gdiplus::GdiplusShutdown(gdiplusToken);

 	if(file_open_failed){
		printf("file not found %s",path);
 		return -1;
 	}
	return 0;
#endif
}
