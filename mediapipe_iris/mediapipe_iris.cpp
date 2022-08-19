/*******************************************************************
*
*    DESCRIPTION:
*      AILIA mediapipe_iris sample
*    AUTHOR:
*
*    DATE:2022/07/20
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#undef UNICODE

#include "ailia.h"
#include "ailia_detector.h"
#include "utils.h"
#include "detector_utils.h"
#include "webcamera_utils.h"


// ======================
// Parameters
// ======================

#define DETECTION_WEIGHT_PATH "blazeface.opt.onnx"
#define DETECTION_MODEL_PATH  "blazeface.opt.onnx.prototxt"
#define LANDMARK_WEIGHT_PATH  "facemesh.opt.onnx"
#define LANDMARK_MODEL_PATH   "facemesh.opt.onnx.prototxt"
#define LANDMARK2_WEIGHT_PATH "iris.opt.onnx"
#define LANDMARK2_MODEL_PATH  "iris.opt.onnx.prototxt"

#define IMAGE_PATH      "man.jpg"
#define SAVE_IMAGE_PATH "output.png"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

static std::string detection_weight(DETECTION_WEIGHT_PATH);
static std::string detection_model(DETECTION_MODEL_PATH);
static std::string landmark_weight(LANDMARK_WEIGHT_PATH);
static std::string landmark_model(LANDMARK_MODEL_PATH);
static std::string landmark2_weight(LANDMARK2_WEIGHT_PATH);
static std::string landmark2_model(LANDMARK2_MODEL_PATH);

static std::string image_path(IMAGE_PATH);
static std::string video_path("0");
static std::string save_image_path(SAVE_IMAGE_PATH);

static bool benchmark  = false;
static bool video_mode = false;
static int args_env_id = -1;


// ======================
// Argument Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: mediapipe_iris [-h] [-i IMAGE] [-v VIDEO] [-s SAVE_IMAGE_PATH] [-b] [-e ENV_ID]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("mediapipe_iris model\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i IMAGE, --input IMAGE\n");
    PRINT_OUT("                        The input image path.\n");
    PRINT_OUT("  -v VIDEO, --video VIDEO\n");
    PRINT_OUT("                        The input video path. If the VIDEO argument is set to\n");
    PRINT_OUT("                        0, the webcam input will be used.\n");
    PRINT_OUT("  -s SAVE_IMAGE_PATH, --savepath SAVE_IMAGE_PATH\n");
    PRINT_OUT("                        Save path for the output image.\n");
    PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
    PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
    PRINT_OUT("                        video mode)\n");
    PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
    PRINT_OUT("                        The backend environment id.\n");
    return;
}


static void print_error(std::string arg)
{
    PRINT_ERR("mediapipe_iris: error: unrecognized arguments: %s\n", arg.c_str());
    return;
}


static int argument_parser(int argc, char **argv)
{
    int status = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (status == 0) {
            if (arg == "-i" || arg == "--input") {
                status = 1;
            }
            else if (arg == "-v" || arg == "--video") {
                video_mode = true;
                status = 2;
            }
            else if (arg == "-s" || arg == "--savepath") {
                status = 3;
            }
            else if (arg == "-b" || arg == "--benchmark") {
                benchmark = true;
            }
            else if (arg == "-h" || arg == "--help") {
                print_usage();
                print_help();
                return -1;
            }
            else if (arg == "-e" || arg == "--env_id") {
                status = 4;
            }
            else {
                print_usage();
                print_error(arg);
                return -1;
            }
        }
        else if (arg[0] != '-') {
            switch (status) {
            case 1:
                image_path = arg;
                break;
            case 2:
                video_path = arg;
                break;
            case 3:
                save_image_path = arg;
                break;
            case 4:
                args_env_id = atoi(arg.c_str());
                break;
            default:
                print_usage();
                print_error(arg);
                return -1;
            }
            status = 0;
        }
        else {
            print_usage();
            print_error(arg);
            return -1;
        }
    }

    return AILIA_STATUS_SUCCESS;
}


// ======================
// Main functions
// ======================

static int recognize_from_image(AILIANetwork* detection_ailia, AILIANetwork* landmark_ailia, AILIANetwork* landmark2_ailia)
{
    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_video(AILIANetwork* detection_ailia, AILIANetwork* landmark_ailia, AILIANetwork* landmark2_ailia)
{
    return AILIA_STATUS_SUCCESS;
}


int main(int argc, char **argv)
{
    int status = argument_parser(argc, argv);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    // env list
    unsigned int env_count;
    status = ailiaGetEnvironmentCount(&env_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetEnvironmentCount Failed %d", status);
        return -1;
    }

    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    for (unsigned int i = 0; i < env_count; i++) {
        AILIAEnvironment* env;
        status = ailiaGetEnvironment(&env, i, AILIA_ENVIRONMENT_VERSION);
        PRINT_OUT("env_id : %d type : %d name : %s\n", env->id, env->type, env->name);
        if (args_env_id == env->id) {
            env_id = env->id;
        }
        if (args_env_id == -1 && env_id == AILIA_ENVIRONMENT_ID_AUTO){
            if (env->type == AILIA_ENVIRONMENT_TYPE_GPU) {
                env_id = env->id;
            }
        }
    }
    if (args_env_id == -1){
        PRINT_OUT("you can select environment using -e option\n");
    }

    // initialize detection net
    AILIANetwork *detection_ailia;
    {
        status = ailiaCreate(&detection_ailia, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(detection_ailia, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(detection_ailia);
            return -1;
        }

        PRINT_OUT("selected env name : %s\n", env_ptr->name);

        status = ailiaOpenStreamFile(detection_ailia, detection_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(detection_ailia));
            ailiaDestroy(detection_ailia);
            return -1;
        }

        status = ailiaOpenWeightFile(detection_ailia, detection_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(detection_ailia);
            return -1;
        }
    }

    // initialize landmark net
    AILIANetwork *landmark_ailia;
    {
        status = ailiaCreate(&landmark_ailia, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(landmark_ailia, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(landmark_ailia);
            return -1;
        }

        status = ailiaOpenStreamFile(landmark_ailia, landmark_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(landmark_ailia));
            ailiaDestroy(landmark_ailia);
            return -1;
        }

        status = ailiaOpenWeightFile(landmark_ailia, landmark_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(landmark_ailia);
            return -1;
        }
    }

    // initialize landmark2 net
    AILIANetwork *landmark2_ailia;
    {
        status = ailiaCreate(&landmark2_ailia, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(landmark2_ailia, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(landmark2_ailia);
            return -1;
        }

        status = ailiaOpenStreamFile(landmark2_ailia, landmark2_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(landmark2_ailia));
            ailiaDestroy(landmark2_ailia);
            return -1;
        }

        status = ailiaOpenWeightFile(landmark2_ailia, landmark2_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(landmark2_ailia);
            return -1;
        }
    }

    if (video_mode) {
        status = recognize_from_video(detection_ailia, landmark_ailia, landmark2_ailia);
    }
    else {
        status = recognize_from_image(detection_ailia, landmark_ailia, landmark2_ailia);
    }

    ailiaDestroy(detection_ailia);
    ailiaDestroy(landmark_ailia);
    ailiaDestroy(landmark2_ailia);

    return status;
}
