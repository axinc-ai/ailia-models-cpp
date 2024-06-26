/*******************************************************************
*
*    DESCRIPTION:
*      AILIA G2P EN sample
*    AUTHOR:
*
*    DATE:2024/06/26
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <math.h>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <iostream>
#include <map>
#include <fstream>
#include <sstream>

#undef UNICODE

#include "ailia.h"
#include "averaged_perceptron.h"

bool debug = false;
bool debug_token = false;


// ======================
// Parameters
// ======================

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

#define MODEL_N 2

#define MODEL_ENCODER 0
#define MODEL_DECODER 1

const char *MODEL_NAME[2] = {"g2p_encoder.onnx", "g2p_decoder.onnx"};

static bool benchmark  = false;
static int args_env_id = -1;

std::string reference_text = "To be or not to be, that is the questionary";

// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
	PRINT_OUT("usage: g2p_en [-h] [-i TEXT] [-b] [-e ENV_ID]\n");
	return;
}


static void print_help()
{
	PRINT_OUT("\n");
	PRINT_OUT("g2p_en model\n");
	PRINT_OUT("\n");
	PRINT_OUT("optional arguments:\n");
	PRINT_OUT("  -h, --help            show this help message and exit\n");
	PRINT_OUT("  -i FILE, --input FILE\n");
	PRINT_OUT("                        The input file.\n");
	PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
	PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
	PRINT_OUT("                        video mode)\n");
	PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
	PRINT_OUT("                        The backend environment id.\n");
	return;
}


static void print_error(std::string arg)
{
	PRINT_ERR("gpt-sovits: error: unrecognized arguments: %s\n", arg.c_str());
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
				reference_text = std::string(arg);
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
// ailia functions
// ======================

void setErrorDetail(const char *func, const char *detail){
	PRINT_ERR("Error %s Detail %s\n", func, detail);
	throw(func);
}

struct AILIATensor{
	std::vector<float> data;
	AILIAShape shape;
};

void forward(AILIANetwork *ailia, std::vector<AILIATensor*> &inputs, std::vector<AILIATensor> &outputs){
	int status;

	unsigned int input_blob_cnt;
	status = ailiaGetInputBlobCount(ailia, &input_blob_cnt);
	if (status != AILIA_STATUS_SUCCESS) {
		setErrorDetail("ailiaGetInputBlobCount",ailiaGetErrorDetail(ailia));
	}

	if (input_blob_cnt != inputs.size()){
		setErrorDetail("input blob cnt and input tensor size must be same", "");
	}

	for (int i = 0; i < inputs.size(); i++){
		unsigned int input_blob_idx = 0;
		status = ailiaGetBlobIndexByInputIndex(ailia, &input_blob_idx, i);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobIndexByInputIndex", ailiaGetErrorDetail(ailia));
		}

		if (debug){
			PRINT_OUT("input blob shape %d %d %d %d dims %d\n",inputs[i]->shape.x,inputs[i]->shape.y,inputs[i]->shape.z,inputs[i]->shape.w,inputs[i]->shape.dim);
		}

		status = ailiaSetInputBlobShape(ailia,&inputs[i]->shape,input_blob_idx,AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaSetInputBlobShape",ailiaGetErrorDetail(ailia));
		}

		status = ailiaSetInputBlobData(ailia, &(inputs[i]->data)[0], inputs[i]->data.size() * sizeof(float), input_blob_idx);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaSetInputBlobData",ailiaGetErrorDetail(ailia));
		}
	}

	status = ailiaUpdate(ailia);
	if (status != AILIA_STATUS_SUCCESS) {
		setErrorDetail("ailiaUpdate",ailiaGetErrorDetail(ailia));
	}

	unsigned int output_blob_cnt;
	status = ailiaGetOutputBlobCount(ailia, &output_blob_cnt);
	if (status != AILIA_STATUS_SUCCESS) {
		setErrorDetail("ailiaGetOutputBlobCount",ailiaGetErrorDetail(ailia));
	}

	for (int i = 0; i < output_blob_cnt; i++){
		unsigned int output_blob_idx = 0;
		status = ailiaGetBlobIndexByOutputIndex(ailia, &output_blob_idx, i);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobIndexByInputIndex",ailiaGetErrorDetail(ailia));
		}

		AILIAShape output_blob_shape;
		status=ailiaGetBlobShape(ailia,&output_blob_shape,output_blob_idx,AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaGetBlobShape", ailiaGetErrorDetail(ailia));
		}

		if (debug){
			PRINT_OUT("output_blob_shape %d %d %d %d dims %d\n",output_blob_shape.x,output_blob_shape.y,output_blob_shape.z,output_blob_shape.w,output_blob_shape.dim);
		}

		if (outputs.size() <= i){
			AILIATensor tensor;
			outputs.push_back(tensor);
		}
		
		AILIATensor &ref_tensor = outputs[i];
		int new_shape = output_blob_shape.x*output_blob_shape.y*output_blob_shape.z*output_blob_shape.w;
		if (new_shape != ref_tensor.data.size()){
			ref_tensor.data.resize(new_shape);
		}
		ref_tensor.shape = output_blob_shape;

		status = ailiaGetBlobData(ailia, &ref_tensor.data[0], ref_tensor.data.size() * sizeof(float), output_blob_idx);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobData",ailiaGetErrorDetail(ailia));
		}
	}
}

// ======================
// dictionary functions
// ======================

std::unordered_map<std::string, std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>> construct_homograph_dictionary(const std::string& dirname) {
	std::unordered_map<std::string, std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>> homograph2features;
	std::ifstream file(dirname + "/homographs.en");

	std::string line;
	while (std::getline(file, line)) {
		if (line[0] == '#') continue;

		std::vector<std::string> parts;
		std::stringstream ss(line);
		std::string part;
		while (std::getline(ss, part, '|')) {
			parts.push_back(part);
		}

		std::string headword = parts[0];
		std::vector<std::string> pron1;
		std::vector<std::string> pron2;
		std::string pron1_str = parts[1];
		std::string pron2_str = parts[2];
		std::string pos1 = parts[3];

		std::stringstream pron1_ss(pron1_str);
		std::stringstream pron2_ss(pron2_str);
		std::string token;
		while (pron1_ss >> token) {
			pron1.push_back(token);
		}
		while (pron2_ss >> token) {
			pron2.push_back(token);
		}

		homograph2features[headword] = std::make_tuple(pron1, pron2, pos1);
	}
	return homograph2features;
}

std::unordered_map<std::string, std::vector<std::string>> construct_cmu_dictionary(const std::string& dirname) {
	std::unordered_map<std::string, std::vector<std::string>> cmudict;
	std::ifstream file(dirname + "/cmudict");

	std::string line;
	while (std::getline(file, line)) {
		if (line[0] == '#') continue;

		std::vector<std::string> lists;
		std::stringstream ss(line);
		std::string part;
		while (ss >> part) {
			lists.push_back(part);
		}

		std::string headword = lists[0];
		std::vector<std::string> pron(lists.begin() + 2, lists.end());
		cmudict[headword] = pron;
	}
	return cmudict;
}

// ======================
// normalize functions
// ======================

std::string toLowerCase(const std::string &text) {
	std::string result = text;
	std::transform(result.begin(), result.end(), result.begin(), ::tolower);
	return result;
}

std::string regexReplace(const std::string &text, const std::regex &pattern, const std::string &replacement) {
	return std::regex_replace(text, pattern, replacement);
}

std::vector<std::string> split(const std::string &text) {
	std::vector<std::string> words;
	std::string word;
	for (char ch : text) {
		if (isspace(ch)) {
			if (!word.empty()) {
				words.push_back(word);
				word.clear();
			}
		} else {
			word += ch;
		}
	}
	if (!word.empty()) {
		words.push_back(word);
	}
	return words;
}

std::vector<int> tokenize(const std::string& word) {
	std::vector<std::string> graphemes;
	graphemes = {"<pad>", "<unk>", "</s>",
					"a", "b", "c", "d", "e", "f", "g",
					"h", "i", "j", "k", "l", "m",
					"n", "o", "p", "q", "r", "s",
					"t", "u", "v", "w", "x", "y", "z"};
	
	std::map<std::string, int> g2idx;
	for(size_t i = 0; i < graphemes.size(); ++i) {
		g2idx[graphemes[i]] = i;
	}

	std::vector<int> x;
	for (const auto& c : word) {
		std::string s(1, c);
		if (g2idx.find(s) != g2idx.end()) {
			x.push_back(g2idx[s]);
		} else {
			x.push_back(g2idx["<unk>"]);
		}
	}
	x.push_back(g2idx["</s>"]);
	return x;
}

// ======================
// predict functions
// ======================

std::vector<std::string> get_phonemes() {
	return {"<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1",
			"AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2",
			"EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG",
			"OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", 
			"UW1", "UW2", "V", "W", "Y", "Z", "ZH"};
}

std::unordered_map<int, std::string> get_idx2p(const std::vector<std::string>& phonemes) {
	std::unordered_map<int, std::string> idx2p;
	for (size_t i = 0; i < phonemes.size(); ++i) {
		idx2p[i] = phonemes[i];
	}
	return idx2p;
}

std::vector<std::string> preds_to_phonemes(const std::vector<int>& preds, const std::unordered_map<int, std::string>& idx2p) {
	std::vector<std::string> phoneme_preds;
	for (int idx : preds) {
		if (idx2p.find(idx) != idx2p.end()) {
			phoneme_preds.push_back(idx2p.at(idx));
		} else {
			phoneme_preds.push_back("<unk>");
		}
	}
	return phoneme_preds;
}


std::vector<std::string> predict(AILIANetwork* net[MODEL_N], const std::string &word){
	std::vector<int> x = tokenize(word);
	if (debug){
		printf("tokens : ");
		for (int i = 0; i < x.size(); i++){
			printf("%d ", x[i]);
		}
		printf("\n");
	}

	std::vector<float> h_data(256);

	AILIATensor h_tensor;
	h_tensor.data = h_data;
	h_tensor.shape.x = 256;
	h_tensor.shape.y = 1;
	h_tensor.shape.z = 1;
	h_tensor.shape.w = 1;
	h_tensor.shape.dim = 2;

	for (int i = 0; i < x.size(); i++){
		std::vector<float> x_data(1);
		x_data[0] = x[i];

		AILIATensor x_tensor;
		x_tensor.data = x_data;
		x_tensor.shape.x = 1;
		x_tensor.shape.y = 1;
		x_tensor.shape.z = 1;
		x_tensor.shape.w = 1;
		x_tensor.shape.dim = 1;

		std::vector<AILIATensor*> encoder_inputs;
		encoder_inputs.push_back(&x_tensor);
		encoder_inputs.push_back(&h_tensor);

		std::vector<AILIATensor> encoder_outputs;
		forward(net[MODEL_ENCODER], encoder_inputs, encoder_outputs);

		h_tensor = encoder_outputs[0];
	}

	std::vector<int> preds;
	int pred = 2;

	for (int i = 0; i < 20; i++){
		std::vector<float> pred_data(1);
		pred_data[0] = pred;

		AILIATensor pred_tensor;
		pred_tensor.data = pred_data;
		pred_tensor.shape.x = pred_data.size();
		pred_tensor.shape.y = 1;
		pred_tensor.shape.z = 1;
		pred_tensor.shape.w = 1;
		pred_tensor.shape.dim = 1;

		std::vector<AILIATensor*> decoder_inputs;
		decoder_inputs.push_back(&pred_tensor);
		decoder_inputs.push_back(&h_tensor);

		std::vector<AILIATensor> decoder_outputs;
		forward(net[MODEL_DECODER], decoder_inputs, decoder_outputs);

		AILIATensor logits_tensor = decoder_outputs[0];
		h_tensor = decoder_outputs[1];

		float max_logits = -1;
		for (int i = 0; i < logits_tensor.shape.x; i++){
			if (max_logits < logits_tensor.data[i]){
				max_logits = logits_tensor.data[i];
				pred = i;
			}
		}

		if (pred == 3){
			break;
		}

		preds.push_back(pred);
	}

	if (debug){
		printf("output\n");
		for (int i = 0; i < preds.size(); i++){
			printf("%d ", preds[i]);
		}
		printf("\n");
	}

	std::vector<std::string> phonemes = get_phonemes();
	std::unordered_map<int, std::string> idx2p = get_idx2p(phonemes);
	std::vector<std::string> phoneme_preds = preds_to_phonemes(preds, idx2p);
	return phoneme_preds;
}

// ======================
// main functions
// ======================

static int compute(AILIANetwork* net[MODEL_N], std::string text)
{
	int status = AILIA_STATUS_SUCCESS;

	printf("Input : \n");
	printf("%s\n", text.c_str());

	text = toLowerCase(text);
	text = regexReplace(text, std::regex("[^ a-z'.,?!\\-]"), "");
	text = regexReplace(text, std::regex("i\\.e\\."), "that is");
	text = regexReplace(text, std::regex("e\\.g\\."), "for example");
	std::string text2 = text;
	text2 = regexReplace(text2, std::regex("\\."), " . ");
	text2 = regexReplace(text2, std::regex(","), " , ");
	text2 = regexReplace(text2, std::regex("!"), " ! ");
	text2 = regexReplace(text2, std::regex("\\?"), " ? ");

	std::vector<std::string> words = split(text2);

	std::vector<std::pair<std::string, std::string>> tokens; // Example tokens
	std::unordered_map<std::string, std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>> homograph2features = construct_homograph_dictionary("./");
	std::unordered_map<std::string, std::vector<std::string>> cmudict = construct_cmu_dictionary("./");

	for (const auto& word : words) {
		tokens.push_back(std::pair<std::string, std::string>(word, word));
	}

	std::vector<std::string> prons;
	for (const auto& token : tokens) {
		std::string word = token.first;
		std::string pos = token.second;
		std::vector<std::string> pron;

		if (!std::regex_search(word, std::regex("[a-z]"))) {
			pron.push_back(word);
		} else if (homograph2features.find(word) != homograph2features.end()) {
			auto [pron1, pron2, pos1] = homograph2features[word];
			if (pos.find(pos1) == 0) {
				pron = pron1;
			} else {
				pron = pron2;
			}
		} else if (cmudict.find(word) != cmudict.end()) {
			pron = cmudict[word];
		} else {
			pron = predict(net, word);
		}

		for (int i = 0; i < pron.size(); i++) {
			prons.push_back(pron[i]);
		}
	}

	PRINT_OUT("Output :\n");
	for (int i = 0; i < prons.size(); i++){
		PRINT_OUT("%s ", prons[i].c_str());
	}
	PRINT_OUT("\n");

	PRINT_OUT("Program finished successfully.\n");

	return AILIA_STATUS_SUCCESS;
}

int main(int argc, char **argv)
{
	test();
	return 0;

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
		//bool is_fp16 = (env->props & AILIA_ENVIRONMENT_PROPERTY_FP16) != 0;
		PRINT_OUT("env_id : %d type : %d name : %s", env->id, env->type, env->name);
		//if (is_fp16){
		//	PRINT_OUT(" (Warning : FP16 backend is not worked this model)\n");
		//	continue;
		//}
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

	// net initialize
	AILIANetwork *ailia[MODEL_N];
	for (int i = 0; i < MODEL_N; i++){
		status = ailiaCreate(&ailia[i], env_id, AILIA_MULTITHREAD_AUTO);
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaCreate failed %d\n", status);
			if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
				PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
			}
			return -1;
		}

		status = ailiaSetMemoryMode(ailia[i], AILIA_MEMORY_OPTIMAIZE_DEFAULT | AILIA_MEMORY_REUSE_INTERSTAGE);
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaSetMemoryMode failed %d\n", status);
			ailiaDestroy(ailia[i]);
			return -1;
		}

		AILIAEnvironment *env_ptr = nullptr;
		status = ailiaGetSelectedEnvironment(ailia[i], &env_ptr, AILIA_ENVIRONMENT_VERSION);
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
			ailiaDestroy(ailia[i]);
			return -1;
		}

		PRINT_OUT("selected env name : %s\n", env_ptr->name);

		status = ailiaOpenWeightFile(ailia[i], MODEL_NAME[i]);
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
			ailiaDestroy(ailia[i]);
			return -1;
		}
	}

	auto start2 = std::chrono::high_resolution_clock::now();
	status = compute(ailia, reference_text);
	auto end2 = std::chrono::high_resolution_clock::now();
	if (benchmark){
		PRINT_OUT("total processing time %lld ms\n",  std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count());
	}

	for (int i = 0; i < MODEL_N; i++){
		ailiaDestroy(ailia[i]);
	}

	return status;
}
