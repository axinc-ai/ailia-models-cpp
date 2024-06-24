/*******************************************************************
*
*    DESCRIPTION:
*      AILIA SoundChoice G2P sample
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
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <regex>

#undef UNICODE

#include "ailia.h"

bool debug = true;
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

#define MODEL_N 3

#define MODEL_BERT 0
#define MODEL_ENCODER 1
#define MODEL_DECODER 2

const char *MODEL_NAME[3] = {"soundchoice-g2p_emb.onnx", "soundchoice-g2p_atn.onnx", "rnn_beam_searcher.onnx"};

static bool benchmark  = false;
static int args_env_id = -1;

//const int  REF_TOKEN_SIZE = 13;
//std::string reference_text = "To be or not to be, that is the question";
//const int reference_token[REF_TOKEN_SIZE] = {101, 2000, 2022, 2030, 2025, 2000, 2022, 1010, 2008, 2003, 1996, 3160, 102};

const int  REF_TOKEN_SIZE = 14;
std::string reference_text = "To be or not to be, that is the questionary";
const int reference_token[REF_TOKEN_SIZE] = {101, 2000, 2022, 2030, 2025, 2000, 2022, 1010, 2008, 2003, 1996, 3160, 5649, 102};

const int BERT_EMBEDDING_SIZE = 768;
const int BERT_HIDDEN_LAYER_N = 4;

// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
	PRINT_OUT("usage: soundchoice-g2p [-h] [-i TEXT] [-b] [-e ENV_ID]\n");
	return;
}


static void print_help()
{
	PRINT_OUT("\n");
	PRINT_OUT("soundchoice-g2p model\n");
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
// Main functions
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

std::string clean_pipeline(const std::string& txt, const std::unordered_set<char>& graphemes) {
	std::regex RE_MULTI_SPACE(R"(\s{2,})");
	std::string result = txt;
	
	// Convert to uppercase
	std::transform(result.begin(), result.end(), result.begin(), ::toupper);
	
	// Remove characters not in graphemes
	result.erase(
		std::remove_if(result.begin(), result.end(), [&](char c) {
			return graphemes.find(c) == graphemes.end();
		}),
		result.end()
	);
	
	// Replace multiple spaces with a single space
	result = std::regex_replace(result, RE_MULTI_SPACE, " ");
	
	return result;
}

std::unordered_map<std::string, int> lab2ind = {
   {"<bos>", 0}, {"<eos>", 1}, {"<unk>", 2}, {"A", 3}, {"B", 4}, {"C", 5}, {"D", 6}, {"E", 7},
	{"F", 8}, {"G", 9}, {"H", 10}, {"I", 11}, {"J", 12}, {"K", 13}, {"L", 14}, {"M", 15}, {"N", 16},
	{"O", 17}, {"P", 18}, {"Q", 19}, {"R", 20}, {"S", 21}, {"T", 22}, {"U", 23}, {"V", 24},
	{"W", 25}, {"X", 26}, {"Y", 27}, {"Z", 28}, {"'", 29}, {" ", 30}
};

std::vector<int> grapheme_pipeline(const std::string& char_seq, bool uppercase = true) {
	std::string char_seq_upper = char_seq;

	if (uppercase) {
		std::transform(char_seq_upper.begin(), char_seq_upper.end(), char_seq_upper.begin(), ::toupper);
	}

	std::vector<std::string> grapheme_list;
	for (const char& c : char_seq_upper) {
		std::string grapheme(1, c);  // convert char to string
		if (lab2ind.find(grapheme) != lab2ind.end()) {
			grapheme_list.push_back(grapheme);
		}
	}

	auto encode_label = [](const std::string& label) -> int {
		try {
			return lab2ind.at(label);
		} catch (const std::out_of_range&) {
			std::string unk_label = "<unk>";
			return lab2ind.at(unk_label);
		}
	};

	std::vector<int> grapheme_encoded_list;
	for (const auto& grapheme : grapheme_list) {
		grapheme_encoded_list.push_back(encode_label(grapheme));
	}

	std::string bos_label = "<bos>";
	std::vector<int> grapheme_encoded = { lab2ind[bos_label] };
	grapheme_encoded.insert(grapheme_encoded.end(), grapheme_encoded_list.begin(), grapheme_encoded_list.end());

	int grapheme_len = grapheme_encoded.size();

	// Convert grapheme_list of strings to list of single characters
	std::vector<int> grapheme_char_list;
	for (const std::string& grapheme : grapheme_list) {
		grapheme_char_list.push_back(grapheme[0]);
	}

	//return {grapheme_char_list, grapheme_encoded_list, grapheme_encoded, grapheme_len};
	return grapheme_encoded;
}

static int is_special_token_or_continue(int token, std::vector<int> &continue_tokens){
	const int TOKEN_ID_CLS = 101;
	const int TOKEN_ID_SEP = 102;
	if (token == TOKEN_ID_CLS || token == TOKEN_ID_SEP || std::count(continue_tokens.begin(), continue_tokens.end(), token) != 0){
		return 1;
	}
	return 0;
}

static std::vector<float> expand_to_chars(std::vector<int> grapheme_encoded, std::vector<float> &word_emb){
	std::vector<float> char_word_emb(grapheme_encoded.size() * BERT_EMBEDDING_SIZE);
	int word_separator = 30; // space
	int word_cnt = 0;
	for (int i = 0; i < grapheme_encoded.size(); i++){
		printf("%d ", grapheme_encoded[i]);

		if (word_emb.size() < BERT_EMBEDDING_SIZE * word_cnt){
			throw("Word emb overflow");
		}

		for (int j = 0; j < BERT_EMBEDDING_SIZE; j++){
			char_word_emb[BERT_EMBEDDING_SIZE * i + j] = word_emb[BERT_EMBEDDING_SIZE * word_cnt + j];
		}

		if (grapheme_encoded[i] == word_separator){
			word_cnt++;
		}
	}
	printf("word_cnt %d %d\n", word_cnt, word_emb.size() / BERT_EMBEDDING_SIZE);
	return char_word_emb;
}


std::vector<AILIATensor> encode_input(AILIANetwork *bert, const std::string& input_text, std::vector<int> &continue_tokens) {
	std::unordered_set<char> graphemes = {
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
		'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
		'\'', ' '
	};

	std::string txt_cleaned = clean_pipeline(input_text, graphemes);
	if (debug){
		printf("input %s\n", input_text.c_str());
		printf("clean %s\n", txt_cleaned.c_str());
	}

	std::vector<int> grapheme_encoded = grapheme_pipeline(txt_cleaned);
	if (debug){
		printf("grapheme_encoded ");
		for (int i = 0; i < grapheme_encoded.size(); i++){
			printf("%d ", grapheme_encoded[i]);
		}
		printf("\n");
	}

	std::vector<float> input_ids_data(REF_TOKEN_SIZE);
	std::vector<float> attention_mask_data(REF_TOKEN_SIZE);
	std::vector<float> token_type_ids_data(REF_TOKEN_SIZE);

	for (int i = 0; i < REF_TOKEN_SIZE; i++){
		input_ids_data[i] = reference_token[i];
		attention_mask_data[i] = 1;
		token_type_ids_data[i] = 0;
	}

	AILIATensor input_ids;
	input_ids.data = input_ids_data;
	input_ids.shape.x = input_ids_data.size();
	input_ids.shape.y = 1;
	input_ids.shape.z = 1;
	input_ids.shape.w = 1;
	input_ids.shape.dim = 2;

	AILIATensor attention_mask;
	attention_mask.data = attention_mask_data;
	attention_mask.shape.x = attention_mask_data.size();
	attention_mask.shape.y = 1;
	attention_mask.shape.z = 1;
	attention_mask.shape.w = 1;
	attention_mask.shape.dim = 2;

	AILIATensor token_type_ids;
	token_type_ids.data = token_type_ids_data;
	token_type_ids.shape.x = token_type_ids_data.size();
	token_type_ids.shape.y = 1;
	token_type_ids.shape.z = 1;
	token_type_ids.shape.w = 1;
	token_type_ids.shape.dim = 2;

	std::vector<AILIATensor*> bert_inputs;
	bert_inputs.push_back(&input_ids);
	bert_inputs.push_back(&attention_mask);
	bert_inputs.push_back(&token_type_ids);
	std::vector<AILIATensor> bert_outputs;
	forward(bert, bert_inputs, bert_outputs);

	if (debug){
		printf("hidden_states shape %d %d %d %d\n", bert_outputs[0].shape.x, bert_outputs[0].shape.y, bert_outputs[0].shape.z, bert_outputs[0].shape.w);
		printf("hidden_states ");
		for (int i = 0; i < 10; i++){
			printf("%f ", bert_outputs[0].data[i]);
		}
		printf("\n");
	}

	// hidden layerの末尾4レイヤーの結果をマージする
	std::vector<float> word_emb_with_special_token(BERT_EMBEDDING_SIZE * bert_outputs[0].shape.y);
	for (int i = 0; i < bert_outputs[0].shape.y; i++){
		for (int j = 0; j < BERT_EMBEDDING_SIZE; j++){
			for (int k = 0; k < BERT_HIDDEN_LAYER_N; k++){
				word_emb_with_special_token[i * BERT_EMBEDDING_SIZE + j] += bert_outputs[0].data[(bert_outputs[0].shape.w - 1 - k) * BERT_EMBEDDING_SIZE * bert_outputs[0].shape.y + i * BERT_EMBEDDING_SIZE + j];
			}
		}
	}

	// wordに対応するトークン位置を取得する
	std::vector<int> token_ids_word;
	for (int i = 0; i < input_ids_data.size(); i++){
		if (!is_special_token_or_continue(input_ids_data[i], continue_tokens)){
			token_ids_word.push_back(i);
		}
	}

	// wordに対応するトークン位置のembeddingを取得する
	std::vector<float> word_emb(BERT_EMBEDDING_SIZE * token_ids_word.size());
	for (int i = 0; i < token_ids_word.size(); i++){
		int id = token_ids_word[i];
		for (int j = 0; j < BERT_EMBEDDING_SIZE; j++){
			word_emb[i * BERT_EMBEDDING_SIZE + j] = word_emb_with_special_token[id * BERT_EMBEDDING_SIZE + j];
		}
	}

	if (debug){
		printf("input_ids_data ");
		for (int i = 0; i < input_ids_data.size(); i++){
			printf("%d ", (int)input_ids_data[i]);
		}
		printf("\n");
		printf("is_special_token_or_continue ");
		for (int i = 0; i < input_ids_data.size(); i++){
			printf("%d ", is_special_token_or_continue(input_ids_data[i], continue_tokens));
		}
		printf("\n");
		printf("token_ids_word ");
		for (int i = 0; i < token_ids_word.size(); i++){
			printf("%d ", token_ids_word[i]);
		}
		printf("\n");
		printf("word_emb ");
		for (int i = 0; i < 10; i++){
			printf("%f ", word_emb[i]);
		}
		printf("\n");
	}

	// character単位のembeddingに変換する
	std::vector<float> char_emb = expand_to_chars(grapheme_encoded, word_emb);

	if (debug){
		printf("\n");
		printf("char_emb ");
		for (int i = 0; i < 10; i++){
			printf("%f ", char_emb[i]);
		}
		printf("\n");
	}

	std::vector<float> grapheme_encoded_data(grapheme_encoded.size());
	for (int i = 0; i < grapheme_encoded.size(); i++){
		grapheme_encoded_data[i] = grapheme_encoded[i];
	}

	AILIATensor grapheme_encoded_tensor;
	grapheme_encoded_tensor.data = grapheme_encoded_data;
	grapheme_encoded_tensor.shape.x = grapheme_encoded.size();
	grapheme_encoded_tensor.shape.y = 1;
	grapheme_encoded_tensor.shape.z = 1;
	grapheme_encoded_tensor.shape.w = 1;
	grapheme_encoded_tensor.shape.dim = 2;

	AILIATensor char_emb_tensor;
	char_emb_tensor.data = char_emb;
	char_emb_tensor.shape.x = BERT_EMBEDDING_SIZE;
	char_emb_tensor.shape.y = grapheme_encoded.size();
	char_emb_tensor.shape.z = 1;
	char_emb_tensor.shape.w = 1;
	char_emb_tensor.shape.dim = 3;

	std::vector<AILIATensor> outputs;
	outputs.push_back(grapheme_encoded_tensor);
	outputs.push_back(char_emb_tensor);

	return outputs;
}

static int compute(AILIANetwork* net[MODEL_N], std::vector<int> &continue_tokens)
{
	int status = AILIA_STATUS_SUCCESS;

	std::vector<AILIATensor> encode_outpus = encode_input(net[MODEL_BERT], reference_text, continue_tokens);

	AILIATensor grapheme_encoded = encode_outpus[0];
	AILIATensor word_emb = encode_outpus[1];

	std::vector<AILIATensor*> atten_inputs;
	atten_inputs.push_back(&grapheme_encoded);
	atten_inputs.push_back(&word_emb);
	std::vector<AILIATensor> atten_outputs;
	forward(net[MODEL_ENCODER], atten_inputs, atten_outputs);

	AILIATensor p_seq = atten_outputs[0];
	AILIATensor encoder_outputs = atten_outputs[1];

	if (debug){
		printf("p_seq.shape %d %d %d %d\n", p_seq.shape.x, p_seq.shape.y, p_seq.shape.z, p_seq.shape.w);
		printf("p_seq ");
		for (int i = 0; i < 10; i++){
			printf("%f ", p_seq.data[i]);
		}
		printf("\n");
		printf("encoder_outputs.shape %d %d %d %d\n", encoder_outputs.shape.x, encoder_outputs.shape.y, encoder_outputs.shape.z, encoder_outputs.shape.w);
		printf("encoder_outputs ");
		for (int i = 0; i < 10; i++){
			printf("%f ", encoder_outputs.data[i]);
		}
		printf("\n");
	}

	PRINT_OUT("Program finished successfully.\n");

	return AILIA_STATUS_SUCCESS;
}

static std::vector<int> load_vocab(const char *path_a)
{
	FILE *fp = NULL;
	fp = fopen(path_a, "r");
	if (fp == NULL){
		throw("vocab file not found");
	}
	std::vector<char> line;
	std::vector<int> continue_tokens;
	int id = 0;
	while(!feof(fp)){
		char c = fgetc(fp);
		line.push_back(c);
		if (c == '\n'){
			line[line.size() - 1] = '\0';
			if (line.size() >= 2){
				//printf("%s\n", &line[0]);
				if (line[0] == '#' && line[1] == '#'){
					continue_tokens.push_back(id);
					//printf("%d ", id);
				}
			}
			line.clear();
			id++;
		}
	}
	fclose(fp);
	return continue_tokens;
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

	std::vector<int> continue_tokens = load_vocab("vocab.txt");

	auto start2 = std::chrono::high_resolution_clock::now();
	status = compute(ailia, continue_tokens);
	auto end2 = std::chrono::high_resolution_clock::now();
	if (benchmark){
		PRINT_OUT("total processing time %lld ms\n",  std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count());
	}

	for (int i = 0; i < MODEL_N; i++){
		ailiaDestroy(ailia[i]);
	}

	return status;
}
