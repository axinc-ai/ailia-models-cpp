/*******************************************************************
*
*    DESCRIPTION:
*      AILIA detector sample (for YOLOv3 onnx)
*    AUTHOR:
*      AXELL Corporation
*    DATE:2019/11/22
*
*******************************************************************/

#include <stdio.h>
#include <vector>
#include <string>

#undef UNICODE

#include "time.h"

#include "gdi_loader.cpp"

#include "ailia_detector_category.h"

#include "ailia.h"
#include "ailia_detector.h"

#define PROTOTXT "../models/yolo-v3-tiny/yolov3-tiny.opt.onnx.prototxt"
#define ONNXMODEL "../models/yolo-v3-tiny/yolov3-tiny.opt.onnx"
#define CATEGORY COCO_CATEGORY
#define CATEGORY_N 80

int detector(AILIADetector *det,const unsigned char *image,int width,int height){
	int status;

	const float threshold=0.2f;
	const float iou=0.45f;
	status=ailiaDetectorCompute(det,image,width*4,width,height,AILIA_IMAGE_FORMAT_BGRA,threshold,iou);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaDetectorCompute Failed %d",status);
		return -1;
	}

	unsigned int obj_count;
	status=ailiaDetectorGetObjectCount(det,&obj_count);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaDetectorGetObjectCount Failed %d",status);
		return -1;
	}

	printf("detected count %d\n",obj_count);

	if(obj_count>5){
		obj_count=5;
	}

	for(int i=0;i<obj_count;i++){
		AILIADetectorObject obj;
		status=ailiaDetectorGetObject(det,&obj,i,AILIA_DETECTOR_OBJECT_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			printf("ailiaDetectorGetObjectCount Failed %d",status);
			return -1;
		}
		printf("category %s prob %f (%f,%f)-(%f,%f)\n",CATEGORY[obj.category],obj.prob,obj.x,obj.y,obj.x+obj.w,obj.y+obj.h);
	}

	return 0;
}

int main(int argc, char **argv){
	if((argc<2) || (5<argc)){
		printf("Usage  : ailia_detector_v3 ../images/person.jpg [model env_id input_size]\n");
		return -1;
	}

	//画像を読み込み
	struct GdiLoaderFileInfo info;
	std::vector<unsigned char> src_image;
	int status=load_image(src_image,info,argv[1]);
	if(status!=AILIA_STATUS_SUCCESS){
		return -1;
	}

	//推論を行う
	AILIANetwork *ailia;

	int env_id = AILIA_ENVIRONMENT_ID_AUTO;
	if (argc>=4) {
		env_id = atoi(argv[3]);
	}

	status=ailiaCreate(&ailia,env_id,AILIA_MULTITHREAD_AUTO);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaCreate Failed %d",status);
		return -1;
	}

	AILIAEnvironment *env_ptr = nullptr;
	status=ailiaGetSelectedEnvironment(ailia, &env_ptr, AILIA_ENVIRONMENT_VERSION);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaGetSelectedEnvironment Failed %d",status);
		return -1;
	}

	printf("env_id : %d\n", env_ptr->id);
	printf("env_name : %s\n", env_ptr->name);

	std::string prototxt(PROTOTXT);
	std::string onnxmodel(ONNXMODEL);
	if(argc>=3){
		prototxt=std::string(argv[2])+std::string(".onnx.prototxt");
		onnxmodel=std::string(argv[2])+std::string(".onnx");
	}

	const int category=CATEGORY_N;
	const unsigned int flags=AILIA_DETECTOR_FLAG_NORMAL;

	AILIADetector *det;
	status=ailiaCreateDetector(&det,ailia,AILIA_NETWORK_IMAGE_FORMAT_RGB,AILIA_NETWORK_IMAGE_CHANNEL_FIRST,AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_FP32,AILIA_DETECTOR_ALGORITHM_YOLOV3,category,flags);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaCreateDetector Failed %d",status);
		return -1;
	}

	printf("Prototxt : %s\n",prototxt.c_str());
	printf("ONNX : %s\n",onnxmodel.c_str());

	status=ailiaOpenStreamFile(ailia,prototxt.c_str());
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaOpenStreamFile Failed %d\n",status);
		printf("ailiaGetErrorDetail %s",ailiaGetErrorDetail(ailia));
		return -1;
	}

	status=ailiaOpenWeightFile(ailia,onnxmodel.c_str());
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaOpenWeightFile Failed %d",status);
		return -1;
	}

	unsigned int model_input_width = 416;
	unsigned int model_input_height = 416;
	if (argc>=5) {
		model_input_width = atoi(argv[4]);
		model_input_height = model_input_width;
	}

	status=ailiaDetectorSetInputShape(det,model_input_width,model_input_height);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaDetectorSetInputShape(w=%u,h=%u) Failed %d", model_input_width, model_input_height, status);
		return -1;
	}

	printf("model_input_width: %u\n", model_input_width);
	printf("model_input_height: %u\n", model_input_height);

#if 0 // for benchmark
	for (int i=0;i<10;++i) {
		clock_t start=clock();
		detector(det,&src_image[0],info.Width,info.Height);
		clock_t end=clock();
		printf("detection time [%2d]: %f sec\n", i,(float)(end-start)/CLOCKS_PER_SEC);
	}
#else
	clock_t start=clock();
	detector(det,&src_image[0],info.Width,info.Height);
	clock_t end=clock();
	printf("detection time : %f sec\n", (float)(end-start)/CLOCKS_PER_SEC);
#endif

	ailiaDestroyDetector(det);
	ailiaDestroy(ailia);

	return 0;
}
