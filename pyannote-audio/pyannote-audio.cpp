/*******************************************************************
*
*    DESCRIPTION:
*      AILIA pyannote-audio sample
*    AUTHOR:
*
*    DATE:2024/04/30
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <math.h>

#include <iostream>

#undef UNICODE

// #include "ailia.h"
// #include "ailia_tokenizer.h"



// ======================
// Parameters
// ======================

#define WEIGHT_SEGMENTATION_PATH "segmentation.onnx"
#define MODEL_SEGMENTATION_PATH "segmentation.onnx.prototxt"
#define WEIGHT_EMBEDDING_PATH "speaker-embedding.onnx"
#define MODEL_EMBEDDING_PATH "speaker-embedding.onnx.prototxt"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif


static std::string segmentation_weight(WEIGHT_SEGMENTATION_PATH);
static std::string segmentation_model(MODEL_SEGMENTATION_PATH);
static std::string embedding_weight(WEIGHT_EMBEDDING_PATH);
static std::string embedding_model(MODEL_EMBEDDING_PATH);





// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
	PRINT_OUT("usage: fugumt [-h] [-i TEXT] [-b] [-e ENV_ID]\n");
	return;
}


static void print_help()
{
	PRINT_OUT("\n");
	PRINT_OUT("fugumt model\n");
	PRINT_OUT("\n");
	PRINT_OUT("optional arguments:\n");
	PRINT_OUT("  -h, --help            show this help message and exit\n");
	PRINT_OUT("  -i TEXT, --input TEXT\n");
	PRINT_OUT("                        The input text.\n");
	PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
	PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
	PRINT_OUT("                        video mode)\n");
	PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
	PRINT_OUT("                        The backend environment id.\n");
	return;
}


static void print_error(std::string arg)
{
	PRINT_ERR("fugumt: error: unrecognized arguments: %s\n", arg.c_str());
	return;
}





// ======================
// Main functions
// ======================

int main(int argc, char **argv)
{
    PRINT_OUT("デバック\n");
}
