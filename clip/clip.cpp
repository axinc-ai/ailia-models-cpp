/*******************************************************************
*
*    DESCRIPTION:
*      AILIA clip sample
*    AUTHOR:
*
*    DATE:2023/06/16
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#define sleep(n) Sleep(n*1000)
#else
#include <unistd.h>
#endif

#undef UNICODE

#include "ailia.h"
#include "ailia_tokenizer.h"

#include "utils.h"
#include "webcamera_utils.h"


// ======================
// Parameters
// ======================

#define WEIGHT_PATH_IMAGE "ViT-B32-encode_image.onnx"
#define MODEL_PATH_IMAGE  "ViT-B32-encode_image.onnx.prototxt"

#define WEIGHT_PATH_TEXT "ViT-B32-encode_text.onnx"
#define MODEL_PATH_TEXT  "ViT-B32-encode_text.onnx.prototxt"

#define IMAGE_PATH  "chelsea.png"

#define IMAGE_WIDTH  224 // for video mode
#define IMAGE_HEIGHT 224 // for video mode

#define MAX_CLASS_COUNT 5
#define SLEEP_TIME      3

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

static std::string weight_image(WEIGHT_PATH_IMAGE);
static std::string model_image(MODEL_PATH_IMAGE);

static std::string weight_text(WEIGHT_PATH_TEXT);
static std::string model_text(MODEL_PATH_TEXT);

static std::string image_path(IMAGE_PATH);
static std::vector<std::string> texts;

static bool benchmark  = false;


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: clip [-h] [-i IMAGE] [-b]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("clip classification model\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i IMAGE, --input IMAGE\n");
    PRINT_OUT("                        The input image path.\n");
    PRINT_OUT("  -t TEXT, --text TEXT\n");
    PRINT_OUT("                        The input text.\n");
    PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
    PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
    PRINT_OUT("                        video mode)\n");
    return;
}


static void print_error(std::string arg)
{
    PRINT_ERR("clip: error: unrecognized arguments: %s\n", arg.c_str());
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
            else if (arg == "-b" || arg == "--benchmark") {
                benchmark = true;
            }
            else if (arg == "-h" || arg == "--help") {
                print_usage();
                print_help();
                return -1;
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
// Utils
// ======================

static void preprocess_image(const cv::Mat& simg, cv::Mat& dimg)
{
    if (simg.channels() == 3) {
        cv::cvtColor(simg, dimg, cv::COLOR_BGR2BGRA);
    }
    else if (simg.channels() == 1) {
        cv::cvtColor(simg, dimg, cv::COLOR_GRAY2BGRA);
    }
    else {
//        dimg = simg.clone();
        simg.copyTo(dimg);
    }

    return;
}


// ======================
// Main functions
// ======================

static int recognize_from_image(AILIANetwork *image_enc)
{
    // prepare input data
    cv::Mat simg = cv::imread(image_path.c_str(), cv::IMREAD_UNCHANGED);
    if (simg.empty()) {
        PRINT_ERR("\'%s\' image not found\n", image_path.c_str());
        return -1;
    }
    cv::Mat img;
    preprocess_image(simg, img);
//    PRINT_OUT("input image shape: (%d, %d, %d)\n",
//              img.cols, img.rows, img.channels());

    // inference
    PRINT_OUT("Start inference...\n");
    int status;
    /*
    status = ailiaPredict(classifier, img.data,
                                    img.cols*4, img.cols, img.rows,
                                    AILIA_IMAGE_FORMAT_BGRA, MAX_CLASS_COUNT);
    */
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaPredict failed %d\n", status);
        return -1;
    }

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}

std::vector<int> tokenize(AILIATokenizer * tokenizer, std::string text){
    int context_length = 77;
	printf("Input Text : %s\n", text.c_str());
	ailiaTokenizerEncode(tokenizer, text.c_str());
	unsigned int count;
	ailiaTokenizerGetTokenCount(tokenizer, &count);
	std::vector<int> tokens(count);
	ailiaTokenizerGetTokens(tokenizer, &tokens[0], count);
	printf("Tokens : ");
	for (int i = 0; i < count; i++){
		printf("%d ", tokens[i]);
	}
	printf("\n");
    return tokens;
}

int main(int argc, char **argv)
{
    int status = argument_parser(argc, argv);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    texts.push_back(std::string("a dog"));
    texts.push_back(std::string("a cat"));


	AILIATokenizer *tokenizer;
	status = ailiaTokenizerCreate(&tokenizer, AILIA_TOKENIZER_TYPE_CLIP, AILIA_TOKENIZER_FLAG_NONE);
	if (status != 0){
		printf("ailiaTokenizerCreate error %d\n", status);
		return -1;
	}
    std::vector<int>  tokens = tokenize(tokenizer, texts[0]);
	ailiaTokenizerDestroy(tokenizer);



    // net initialize
    AILIANetwork *ailia;
    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    status = ailiaCreate(&ailia, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }

    AILIAEnvironment *env_ptr = nullptr;
    status = ailiaGetSelectedEnvironment(ailia, &env_ptr, AILIA_ENVIRONMENT_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
        ailiaDestroy(ailia);
        return -1;
    }

//    PRINT_OUT("env_id: %d\n", env_ptr->id);
    PRINT_OUT("env_name: %s\n", env_ptr->name);

    status = ailiaOpenStreamFile(ailia, model_image.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia));
        ailiaDestroy(ailia);
        return -1;
    }

    status = ailiaOpenWeightFile(ailia, weight_image.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
        ailiaDestroy(ailia);
        return -1;
    }

    status = recognize_from_image(ailia);

    ailiaDestroy(ailia);

    return status;
}
