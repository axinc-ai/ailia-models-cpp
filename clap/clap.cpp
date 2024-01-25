/*******************************************************************
*
*    DESCRIPTION:
*      AILIA clap sample
*    AUTHOR:
*
*    DATE:2024/01/25
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <algorithm>

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


// ======================
// Parameters
// ======================

#define CLAP_AUDIO_WEIGHT_PATH	"CLAP_audio_LAION-Audio-630K_with_fusion.onnx"
#define CLAP_AUDIO_MODEL_PATH	"CLAP_audio_LAION-Audio-630K_with_fusion.onnx.prototxt"
#define CLAP_TEXT_ROBERTAMODEL_WEIGHT_PATH	"CLAP_text_text_branch_RobertaModel_roberta-base.onnx"
#define CLAP_TEXT_ROBERTAMODEL_MODEL_PATH	"CLAP_text_text_branch_RobertaModel_roberta-base.onnx.prototxt"
#define CLAP_TEXT_PROJECTION_WEIGHT_PATH	"CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx"
#define CLAP_TEXT_PROJECTION_MODEL_PATH		"CLAP_text_projection_LAION-Audio-630K_with_fusion.onnx.prototxt"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

static std::string weight_audio(CLAP_AUDIO_WEIGHT_PATH);
static std::string model_audio(CLAP_AUDIO_MODEL_PATH);
static std::string weight_text_robertamodel(CLAP_TEXT_ROBERTAMODEL_WEIGHT_PATH);
static std::string model_text_robertamodel(CLAP_TEXT_ROBERTAMODEL_MODEL_PATH);
static std::string weight_text_projection(CLAP_TEXT_PROJECTION_WEIGHT_PATH);
static std::string model_text_projection(CLAP_TEXT_PROJECTION_MODEL_PATH);

static std::string input_wav_path("input.wav");

static bool benchmark  = false;
static int args_env_id = -1;
static bool debug = false;


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: clap [-h] [-i WAV_FILE] [-b] [-e ENV_ID]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("clap classification model\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i WAV_FILE, --input WAV_FILE\n");
    PRINT_OUT("                        The input wav file path.\n");
    PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
    PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
    PRINT_OUT("                        video mode)\n");
	PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
	PRINT_OUT("                        The backend environment id.\n");
    return;
}


static void print_error(std::string arg)
{
    PRINT_ERR("clap: error: unrecognized arguments: %s\n", arg.c_str());
    return;
}


static int argument_parser(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
		if (arg == "-i" || arg == "--input") {
			input_wav_path = argv[++i];
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
			args_env_id = atoi(argv[++i]);
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

int get_env_id(void)
{
    unsigned int env_count;
	int status = ailiaGetEnvironmentCount(&env_count);
	if (status != AILIA_STATUS_SUCCESS) {
		PRINT_ERR("ailiaGetEnvironmentCount Failed %d", status);
		return -1;
	}

	int env_id = AILIA_ENVIRONMENT_ID_AUTO;
	for (unsigned int i = 0; i < env_count; i++) {
		AILIAEnvironment* env;
		status = ailiaGetEnvironment(&env, i, AILIA_ENVIRONMENT_VERSION);
		bool is_fp16 = (env->props & AILIA_ENVIRONMENT_PROPERTY_FP16) != 0;
		PRINT_OUT("env_id : %d type : %d name : %s", env->id, env->type, env->name);
		PRINT_OUT("\n");
		if (args_env_id == env->id){
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

    return env_id;
}

int main(int argc, char **argv)
{
    int status = argument_parser(argc, argv);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

	// env list
    int env_id = get_env_id();

    PRINT_OUT("Program finished successfully.\n");

    return status;
}
