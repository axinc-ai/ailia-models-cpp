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

static std::string vocab_file("roberta-base/vocab.json");
static std::string merge_file("roberta-base/merges.txt");

static std::string input_wav_path("input.wav");
static std::vector<std::string> texts;

static bool benchmark  = false;
static int args_env_id = -1;
static bool debug = true;


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
    PRINT_OUT("  -t TEXT, --text TEXT\n");
    PRINT_OUT("                        The input text.\n");
	PRINT_OUT("  -v VOCAB_FILE, --vocab_file VOCAB_FILE\n");
    PRINT_OUT("                        The vocab file in roberta tokenizer.\n");
	PRINT_OUT("  -m MERGE_FILE, --merge_file MERGE_FILE\n");
    PRINT_OUT("                        The merge file in roberta tokenizer.\n");
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
		else if (arg == "-t" || arg == "--text") {
			texts.push_back(argv[++i]);
		}
		else if (arg == "-v" || arg == "--vocab_file") {
			vocab_file = argv[++i];
		}
		else if (arg == "-m" || arg == "--merge_file") {
			merge_file = argv[++i];
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
// Audio embeddings
// ======================


// ======================
// Text embeddings
// ======================

void tokenize(AILIATokenizer* tokenizer, std::string text, 
	std::vector<int>& input_ids, std::vector<int>& attention_mask,
	unsigned int max_length=77)
{
    if (debug){
        PRINT_OUT("Input Text : %s\n", text.c_str());
    }
	ailiaTokenizerEncode(tokenizer, text.c_str());
	unsigned int count;
	ailiaTokenizerGetTokenCount(tokenizer, &count);
	std::vector<int> tokens(count);
	ailiaTokenizerGetTokens(tokenizer, &tokens[0], count);

	input_ids = std::vector<int>(max_length);
	attention_mask = std::vector<int>(max_length);
    for (int i = 0; i < max_length; i++){
        if (i < tokens.size()){
            input_ids[i] = tokens[i];
			attention_mask[i] = 1;
        }else{
            input_ids[i] = 1;
			attention_mask[i] = 0;
        }
    }
    if (debug){
        PRINT_OUT("Tokens : ");
        for (int i = 0; i < input_ids.size(); i++){
            PRINT_OUT("%d ", input_ids[i]);
        }
        PRINT_OUT("\n");
    }
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

int initialize_ailia(AILIANetwork **ailia, int env_id, std::string model_file, std::string weight_file){
    int status = ailiaCreate(ailia, env_id, AILIA_MULTITHREAD_AUTO);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaCreate failed %d\n", status);
        return -1;
    }
    status = ailiaOpenStreamFile(*ailia, model_file.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
        PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(*ailia));
        ailiaDestroy(*ailia);
        return -1;
    }
    status = ailiaOpenWeightFile(*ailia, weight_file.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
        ailiaDestroy(*ailia);
        return -1;
    }
    return AILIA_STATUS_SUCCESS;
}

int main(int argc, char **argv)
{
    int status = argument_parser(argc, argv);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

	// env list
    int env_id = get_env_id();

    // net initialize
    AILIANetwork *ailia_audio;
    status = initialize_ailia(&ailia_audio, env_id, model_audio, weight_audio);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }
    AILIANetwork *ailia_text_robertamodel;
    status = initialize_ailia(&ailia_text_robertamodel, env_id, model_text_robertamodel, weight_text_robertamodel);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }
    AILIANetwork *ailia_text_projection;
    status = initialize_ailia(&ailia_text_projection, env_id, model_text_projection, weight_text_projection);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    // Tokenize
    std::vector<std::vector<int>> ary_input_ids;
    std::vector<std::vector<int>> ary_attention_mask;
    if (texts.size() == 0){
		texts = {
			"applause applaud clap", 
			"The crowd is clapping.",
			"I love the contrastive learning", 
			"bell", 
			"soccer", 
			"open the door.",
			"applause",
			"dog",
			"dog barking"
		};
    }
	AILIATokenizer *tokenizer;
	status = ailiaTokenizerCreate(&tokenizer, AILIA_TOKENIZER_TYPE_ROBERTA, AILIA_TOKENIZER_FLAG_NONE);
	if (status != 0){
		PRINT_ERR("ailiaTokenizerCreate error %d\n", status);
		return -1;
	}
	status = ailiaTokenizerOpenVocabFile(tokenizer, vocab_file.c_str());
	if (status != 0){
		printf("ailiaTokenizerOpenVocabFile error %d\n", status);
		return -1;
	}
	status = ailiaTokenizerOpenMergeFile(tokenizer, merge_file.c_str());
	if (status != 0){
		printf("ailiaTokenizerOpenMergeFile error %d\n", status);
		return -1;
	}
    PRINT_OUT("Tokenize...\n");
    for (int i = 0; i < texts.size(); i++){
		std::vector<int> input_ids;
		std::vector<int> attention_mask;
        tokenize(tokenizer, texts[i], input_ids, attention_mask);
        ary_input_ids.push_back(input_ids);
		ary_attention_mask.push_back(attention_mask);
    }
	ailiaTokenizerDestroy(tokenizer);

    // release instance
    ailiaDestroy(ailia_audio);
    ailiaDestroy(ailia_text_robertamodel);
	ailiaDestroy(ailia_text_projection);

    PRINT_OUT("Program finished successfully.\n");

    return status;
}
