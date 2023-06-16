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

void softmax(float *data, int n){
	float sum=0;
	for(int i=0;i<n;i++){
		sum+=exp(data[i]);
	}
	for(int i=0;i<n;i++){
		data[i]=exp(data[i])/sum;
	}
}

float norm(std::vector<float> & vec1){
	float norm1 = 0;
	for (int i = 0; i < vec1.size(); i++){
		norm1 += vec1[i] * vec1[i];
	}
	norm1 = sqrt(norm1);
	return norm1;
}

float cos_similarity(std::vector<float> & vec1, std::vector<float> & vec2){
	float sum = 0;
	float norm1 = norm(vec1);
	float norm2 = norm(vec2);
	for (int i = 0; i < vec1.size(); i++){
		sum += (vec1[i] / norm1) * (vec2[i] / norm2);
	}
	return sum;
}

// ======================
// Main functions
// ======================

/*
    image_feature = image_feature / np.linalg.norm(image_feature, ord=2, axis=-1, keepdims=True)
    logit_scale = 100
    logits_per_image = (image_feature * logit_scale).dot(text_feature.T)
    pred = softmax(logits_per_image, axis=1)
*/

static std::vector<float> image_embedding(AILIANetwork *image_enc)
{
    std::vector<float> features(512);

    // prepare input data
    cv::Mat simg = cv::imread(image_path.c_str(), cv::IMREAD_UNCHANGED);
    if (simg.empty()) {
        PRINT_ERR("\'%s\' image not found\n", image_path.c_str());
        return features;
    }
    cv::Mat img;
    preprocess_image(simg, img);

    // rgb order, (/255 - mean )/std
    std::vector<float> input_img(IMAGE_WIDTH * IMAGE_WIDTH * 3);
    float mean[3] = {0.48145466, 0.4578275, 0.40821073};
    float stdf[3] = {0.26862954, 0.26130258, 0.27577711};
    for (int y = 0; y < IMAGE_WIDTH; y++){
        for (int x = 0; x < IMAGE_WIDTH; x++){
            int y2 = img.rows * y / IMAGE_WIDTH;
            int x2 = img.cols * x / IMAGE_WIDTH;
            input_img[0 * IMAGE_WIDTH * IMAGE_WIDTH + y * IMAGE_WIDTH + x] = (img.data[(img.cols * y2 + x2)*4 + 0] / 255.0f - mean[0])/stdf[0];
            input_img[1 * IMAGE_WIDTH * IMAGE_WIDTH + y * IMAGE_WIDTH + x] = (img.data[(img.cols * y2 + x2)*4 + 1] / 255.0f - mean[1])/stdf[1];
            input_img[2 * IMAGE_WIDTH * IMAGE_WIDTH + y * IMAGE_WIDTH + x] = (img.data[(img.cols * y2 + x2)*4 + 2] / 255.0f - mean[2])/stdf[2];
        }
    }

    // inference
    int status;

    unsigned int input_blob_idx = 0;
    status = ailiaGetBlobIndexByInputIndex(image_enc, &input_blob_idx, 0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ImageEmbedding ailiaGetBlobIndexByInputIndex %d", status);
        return features;
    }

    AILIAShape sequence_shape;
    sequence_shape.x=IMAGE_WIDTH;
    sequence_shape.y=IMAGE_WIDTH;
    sequence_shape.z=3;
    sequence_shape.w=1;
    sequence_shape.dim=4;

    status = ailiaSetInputBlobShape(image_enc,&sequence_shape,input_blob_idx,AILIA_SHAPE_VERSION);
    if(status!=AILIA_STATUS_SUCCESS){
        PRINT_ERR("ImageEmbedding ailiaSetInputBlobShape failed %d\n", status);
        return features;
    }

    status = ailiaPredict(image_enc, &features[0], features.size() * sizeof(float), &input_img[0], input_img.size() * sizeof(float));
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ImageEmbedding ailiaPredict failed %d\n", status);
        return features;
    }

    return features;
}

std::vector<int> tokenize(AILIATokenizer * tokenizer, std::string text){
    int context_length = 77;
	printf("Input Text : %s\n", text.c_str());
	ailiaTokenizerEncode(tokenizer, text.c_str());
	unsigned int count;
	ailiaTokenizerGetTokenCount(tokenizer, &count);
	std::vector<int> tokens(count);
	ailiaTokenizerGetTokens(tokenizer, &tokens[0], count);
    std::vector<int> pad_tokens(context_length);
    for (int i = 0; i < context_length; i++){
        if (i < tokens.size()){
            pad_tokens[i] = tokens[i];
            if (i == context_length - 1){
                pad_tokens[i] = tokens[tokens.size() - 1]; // SOT
            }
        }else{
            pad_tokens[i] = 0;
        }
    }
	printf("Tokens : ");
	for (int i = 0; i < pad_tokens.size(); i++){
		printf("%d ", pad_tokens[i]);
	}
	printf("\n");
    return tokens;
}

std::vector<float> text_embedding(AILIANetwork *ailia_text, std::vector<int> &tokens){
    std::vector<float> features(512);

    unsigned int input_blob_idx = 0;
    int status = ailiaGetBlobIndexByInputIndex(ailia_text, &input_blob_idx, 0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("TextEmbedding ailiaGetBlobIndexByInputIndex %d", status);
        return features;
    }

    AILIAShape sequence_shape;
    sequence_shape.x=77;
    sequence_shape.y=1;
    sequence_shape.z=1;
    sequence_shape.w=1;
    sequence_shape.dim=2;

    status = ailiaSetInputBlobShape(ailia_text,&sequence_shape,input_blob_idx,AILIA_SHAPE_VERSION);
    if(status!=AILIA_STATUS_SUCCESS){
        PRINT_ERR("TextEmbedding ailiaSetInputBlobShape failed %d\n", status);
        return features;
    }

    std::vector<float> tokens_float(77);
    for (int i = 0; i < 77; i++){
        tokens_float[i] = (float)tokens[i];
    }
    status = ailiaPredict(ailia_text, &features[0], features.size() * sizeof(float), &tokens_float[0], tokens_float.size() * sizeof(int));
    if (status != AILIA_STATUS_SUCCESS){
        PRINT_ERR("TextEmbedding ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_text));
        return features;
    }
    return features;
}

int main(int argc, char **argv)
{
    int status = argument_parser(argc, argv);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    // net initialize
    AILIANetwork *ailia_image;
    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    status = ailiaCreate(&ailia_image, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }
    AILIAEnvironment *env_ptr = nullptr;
    status = ailiaGetSelectedEnvironment(ailia_image, &env_ptr, AILIA_ENVIRONMENT_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
        ailiaDestroy(ailia_image);
        return -1;
    }
    PRINT_OUT("env_name: %s\n", env_ptr->name);
    status = ailiaOpenStreamFile(ailia_image, model_image.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_image));
        ailiaDestroy(ailia_image);
        return -1;
    }
    status = ailiaOpenWeightFile(ailia_image, weight_image.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
        ailiaDestroy(ailia_image);
        return -1;
    }

    AILIANetwork *ailia_text;
    status = ailiaCreate(&ailia_text, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }
    status = ailiaGetSelectedEnvironment(ailia_text, &env_ptr, AILIA_ENVIRONMENT_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
        ailiaDestroy(ailia_text);
        return -1;
    }
    PRINT_OUT("env_name: %s\n", env_ptr->name);
    status = ailiaOpenStreamFile(ailia_text, model_text.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_text));
        ailiaDestroy(ailia_text);
        return -1;
    }
    status = ailiaOpenWeightFile(ailia_text, weight_text.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
        ailiaDestroy(ailia_text);
        return -1;
    }

    // Tokenize
    texts.push_back(std::string("a dog"));
    texts.push_back(std::string("a cat"));

    std::vector<std::vector<int>> tokens;

	AILIATokenizer *tokenizer;
	status = ailiaTokenizerCreate(&tokenizer, AILIA_TOKENIZER_TYPE_CLIP, AILIA_TOKENIZER_FLAG_NONE);
	if (status != 0){
		printf("ailiaTokenizerCreate error %d\n", status);
		return -1;
	}
    for (int i = 0; i < texts.size(); i++){
        std::vector<int>  token = tokenize(tokenizer, texts[i]);
        tokens.push_back(token);
    }
	ailiaTokenizerDestroy(tokenizer);

    // text embedding
    std::vector< std::vector<float> > text_features;
    for (int i = 0; i < texts.size(); i++){
        std::vector<float> features = text_embedding(ailia_text, tokens[i]);
        text_features.push_back(features);
    }

    // image embedding
    std::vector<float> image_features = image_embedding(ailia_image);

    // distance


    ailiaDestroy(ailia_image);
    ailiaDestroy(ailia_text);

    PRINT_OUT("Program finished successfully.\n");

    return status;
}
