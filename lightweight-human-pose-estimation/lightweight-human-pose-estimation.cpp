/*******************************************************************
*
*    DESCRIPTION:
*      AILIA pose estimator sample
*    AUTHOR:
*      AXELL Corporation
*    DATE:2019/04/28
*
*******************************************************************/

#include <stdio.h>
#include <vector>
#include <string>

#include "time.h"

#include "gdi_loader.cpp"

#include "ailia.h"
#include "ailia_pose_estimator.h"

#define PROTOTXT "../models/lightweight-human-pose-estimation/lightweight-human-pose-estimation.opt.onnx.prototxt"
#define CAFFEMODEL "../models/lightweight-human-pose-estimation/lightweight-human-pose-estimation.opt.onnx"

int pose_estimator(AILIAPoseEstimator *pose,const unsigned char *image,int width,int height){
	int status;

	status=ailiaPoseEstimatorCompute(pose,image,width*4,width,height,AILIA_IMAGE_FORMAT_BGRA);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaPoseEstimatorCompute Failed %d",status);
		return -1;
	}

	unsigned int obj_count;
	status=ailiaPoseEstimatorGetObjectCount(pose,&obj_count);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaDetectorGetObjectCount Failed %d",status);
		return -1;
	}

	printf("detected person %d\n",obj_count);

	for(int i=0;i<obj_count;i++){
		AILIAPoseEstimatorObjectPose obj;
		status=ailiaPoseEstimatorGetObjectPose(pose,&obj,i,AILIA_POSE_ESTIMATOR_OBJECT_POSE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			printf("ailiaPoseEstimatorGetObjectPose Failed %d",status);
			return -1;
		}
		printf("person %d\n",i);
		for(int j=0;j<AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_CNT;j++){
			printf("keypoint %d (%f,%f)\n",j,obj.points[j].x,obj.points[j].y);
		}
	}

	return 0;
}

int main(int argc, char **argv){
	if(argc!=2 && argc!=3){
		printf("Usage  : ailia_pose_estimator ../images/person.jpg\n");
		return -1;
	}

	//画像を読み込み
	struct GdiLoaderFileInfo info;
	std::vector<unsigned char> src_image;
	int status=load_image(src_image,info,argv[1]);
	if(status!=AILIA_STATUS_SUCCESS){
		return -1;
	}

	//環境選択
	unsigned int env_count;
	status=ailiaGetEnvironmentCount(&env_count);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaGetEnvironmentCount Failed %d",status);
		return -1;
	}
	int env_id=AILIA_ENVIRONMENT_ID_AUTO;
	for(unsigned int i=0;i<env_count;i++){
		AILIAEnvironment* env;
		status=ailiaGetEnvironment(&env,i,AILIA_ENVIRONMENT_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			printf("ailiaGetEnvironment Failed %d",status);
			return -1;
		}
		printf("Idx:%d Name:%s Type:%d\n",i,env->name,env->type);
		if(env->type==AILIA_ENVIRONMENT_TYPE_GPU){
			env_id=env->id;
		}
	}
	printf("Selected %d\n",env_id);

	//推論を行う
	AILIANetwork *ailia;

	status=ailiaCreate(&ailia,env_id,AILIA_MULTITHREAD_AUTO);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaCreate Failed %d",status);
		return -1;
	}

	std::string prototxt(PROTOTXT);
	std::string caffemodel(CAFFEMODEL);
	if(argc==3){
		prototxt=std::string(argv[2])+std::string(".prototxt");
		caffemodel=std::string(argv[2])+std::string(".caffemodel");
	}

	printf("Prototxt : %s\n",prototxt.c_str());
	printf("Caffemodel : %s\n",caffemodel.c_str());

	status=ailiaOpenStreamFile(ailia,prototxt.c_str());
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaOpenStreamFile Failed %d\n",status);
		printf("ailiaGetErrorDetail %s\n",ailiaGetErrorDetail(ailia));
		return -1;
	}

	status=ailiaOpenWeightFile(ailia,caffemodel.c_str());
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaOpenWeightFile Failed %d\n",status);
		return -1;
	}

	AILIAPoseEstimator *pose;
	status=ailiaCreatePoseEstimator(&pose,ailia,AILIA_POSE_ESTIMATOR_ALGORITHM_LW_HUMAN_POSE);
	if(status!=AILIA_STATUS_SUCCESS){
		printf("ailiaCreatePoseEstimator Failed %d\n",status);
		printf("ailiaGetErrorDetail %s\n",ailiaGetErrorDetail(ailia));
		return -1;
	}

	clock_t start=clock();
	pose_estimator(pose,&src_image[0],info.Width,info.Height);
	clock_t end=clock();
	printf("estimate time : %f sec\n",(float)(end-start)/CLOCKS_PER_SEC);

	ailiaDestroyPoseEstimator(pose);
	ailiaDestroy(ailia);

	return 0;
}
