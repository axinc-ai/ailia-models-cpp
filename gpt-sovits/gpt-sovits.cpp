/*******************************************************************
*
*    DESCRIPTION:
*      AILIA GPT-SoVits sample
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

#undef UNICODE

#include "ailia.h"
#include "ailia_audio.h"
#include "wave_reader.h"

bool debug = false;


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

#define NUM_INPUTS 4
#define NUM_OUTPUTS 3

#define MODEL_N 5

const char *MODEL_NAME[5] = {"nahida_cnhubert.onnx", "nahida_t2s_encoder.onnx", "nahida_t2s_fsdec.onnx", "nahida_t2s_sdec.onnx", "nahida_vits.onnx"};

static bool benchmark  = false;
static int args_env_id = -1;

std::string input_text = "reference_audio_captured_by_ax.wav";

const char * REF_PHONES[37] = {"m", "i", "z", "u", "o", "m", "a", "r", "e", "e", "sh", "i", "a", "k", "a", "r", "a", "k", "a", "w", "a", "n", "a", "k", "U", "t", "e", "w", "a", "n", "a", "r", "a", "n", "a", "i", "."};
const char *TEXT_PHONES[72] = {"e", "i", "e", "cl", "k", "U", "s", "u", "k", "a", "b", "u", "sh", "I", "k", "i", "g", "a", "i", "sh", "a", "d", "e", "w", "a", "e", "e", "a", "i", "n", "o", "j", "i", "ts", "u", "y", "o", "o", "k", "a", "n", "o", "t", "a", "m", "e", "n", "o", "g", "i", "j", "u", "ts", "u", "o", "k", "a", "i", "h", "a", "ts", "u", "sh", "I", "t", "e", "i", "m", "a", "s", "U", "."};

/*
[[225 160 319 254 229 225  96 248 129 129 251 160  96 222  96 248  96 222
   96 316  96 227  96 222  82 252 129 316  96 227  96 248  96 227  96 160
    3]]
*/

/*
[[129 160 129 126 222  82 250 254 222  96 122 254 251  52 222 160 156  96
  160 251  96 127 129 316  96 129 129  96 160 227 229 221 160 253 254 318
  229 229 222  96 227 229 252  96 225 129 227 229 156 160 221 254 253 254
  229 222  96 160 158  96 253 254 251  52 252 129 160 225  96 250  82   3]]
*/

#define SYMBOLS_N 322
const char *SYMBOLS[SYMBOLS_N] = {"!", ",", "-", ".", "?", "AA", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "E1", "E2", "E3", "E4", "E5", "EE", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "En1", "En2", "En3", "En4", "En5", "F", "G", "HH", "I", "IH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OO", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "SP", "SP2", "SP3", "T", "TH", "U", "UH0", "UH1", "UH2", "UNK", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH", "_", "a", "a1", "a2", "a3", "a4", "a5", "ai1", "ai2", "ai3", "ai4", "ai5", "an1", "an2", "an3", "an4", "an5", "ang1", "ang2", "ang3", "ang4", "ang5", "ao1", "ao2", "ao3", "ao4", "ao5", "b", "by", "c", "ch", "cl", "d", "dy", "e", "e1", "e2", "e3", "e4", "e5", "ei1", "ei2", "ei3", "ei4", "ei5", "en1", "en2", "en3", "en4", "en5", "eng1", "eng2", "eng3", "eng4", "eng5", "er1", "er2", "er3", "er4", "er5", "f", "g", "gy", "h", "hy", "i", "i01", "i02", "i03", "i04", "i05", "i1", "i2", "i3", "i4", "i5", "ia1", "ia2", "ia3", "ia4", "ia5", "ian1", "ian2", "ian3", "ian4", "ian5", "iang1", "iang2", "iang3", "iang4", "iang5", "iao1", "iao2", "iao3", "iao4", "iao5", "ie1", "ie2", "ie3", "ie4", "ie5", "in1", "in2", "in3", "in4", "in5", "ing1", "ing2", "ing3", "ing4", "ing5", "iong1", "iong2", "iong3", "iong4", "iong5", "ir1", "ir2", "ir3", "ir4", "ir5", "iu1", "iu2", "iu3", "iu4", "iu5", "j", "k", "ky", "l", "m", "my", "n", "ny", "o", "o1", "o2", "o3", "o4", "o5", "ong1", "ong2", "ong3", "ong4", "ong5", "ou1", "ou2", "ou3", "ou4", "ou5", "p", "py", "q", "r", "ry", "s", "sh", "t", "ts", "u", "u1", "u2", "u3", "u4", "u5", "ua1", "ua2", "ua3", "ua4", "ua5", "uai1", "uai2", "uai3", "uai4", "uai5", "uan1", "uan2", "uan3", "uan4", "uan5", "uang1", "uang2", "uang3", "uang4", "uang5", "ui1", "ui2", "ui3", "ui4", "ui5", "un1", "un2", "un3", "un4", "un5", "uo1", "uo2", "uo3", "uo4", "uo5", "v", "v1", "v2", "v3", "v4", "v5", "van1", "van2", "van3", "van4", "van5", "ve1", "ve2", "ve3", "ve4", "ve5", "vn1", "vn2", "vn3", "vn4", "vn5", "w", "x", "y", "z", "zh", "…"};


// ======================
// Arguemnt Parser
// ======================

static void print_usage()
{
	PRINT_OUT("usage: gpt-sovits [-h] [-i TEXT] [-b] [-e ENV_ID]\n");
	return;
}


static void print_help()
{
	PRINT_OUT("\n");
	PRINT_OUT("gpt-sovits model\n");
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
	PRINT_ERR("fugumt: error: unrecognized arguments: %s\n", arg.c_str());
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
				input_text = arg;
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
}

int forward(AILIANetwork *ailia, std::vector<float> *inputs[NUM_INPUTS], std::vector<float> *outputs[NUM_OUTPUTS]){
	int status;

	for (int i = 0; i < NUM_INPUTS; i++){
		unsigned int input_blob_idx = 0;
		status = ailiaGetBlobIndexByInputIndex(ailia, &input_blob_idx, i);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobIndexByInputIndex", ailiaGetErrorDetail(ailia));
			return status;
		}

		AILIAShape sequence_shape;
		int batch_size = 1;
		if ( i == 0 ){
			sequence_shape.x=inputs[i]->size() / batch_size;
			sequence_shape.y=batch_size;
			sequence_shape.z=1;
			sequence_shape.w=1;
			sequence_shape.dim=2;
		}
		if ( i == 1 ){
			sequence_shape.x=inputs[i]->size();
			sequence_shape.y=1;
			sequence_shape.z=1;
			sequence_shape.w=1;
			sequence_shape.dim=1;
		}
		if ( i == 2 || i == 3){
			sequence_shape.x=inputs[i]->size() / batch_size / 2;
			sequence_shape.y=batch_size;
			sequence_shape.z=2;
			sequence_shape.w=1;
			sequence_shape.dim=3;
		}
		if (debug){
			printf("input blob shape %d %d %d %d dims %d\n",sequence_shape.x,sequence_shape.y,sequence_shape.z,sequence_shape.w,sequence_shape.dim);
		}

		status = ailiaSetInputBlobShape(ailia,&sequence_shape,input_blob_idx,AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaSetInputBlobShape",ailiaGetErrorDetail(ailia));
			return status;
		}

		if (inputs[i]->size() > 0){
			status = ailiaSetInputBlobData(ailia, &(*inputs[i])[0], inputs[i]->size() * sizeof(float), input_blob_idx);
			if (status != AILIA_STATUS_SUCCESS) {
				setErrorDetail("ailiaSetInputBlobData",ailiaGetErrorDetail(ailia));
				return status;
			}
		}
	}

	status = ailiaUpdate(ailia);
	if (status != AILIA_STATUS_SUCCESS) {
		setErrorDetail("ailiaUpdate",ailiaGetErrorDetail(ailia));
		return status;
	}

	for (int i = 0; i < NUM_OUTPUTS; i++){
		unsigned int output_blob_idx = 0;
		status = ailiaGetBlobIndexByOutputIndex(ailia, &output_blob_idx, i);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobIndexByInputIndex",ailiaGetErrorDetail(ailia));
			return status;
		}

		AILIAShape output_blob_shape;
		status=ailiaGetBlobShape(ailia,&output_blob_shape,output_blob_idx,AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaGetBlobShape", ailiaGetErrorDetail(ailia));
			return status;
		}

		if (debug){
			printf("output_blob_shape %d %d %d %d dims %d\n",output_blob_shape.x,output_blob_shape.y,output_blob_shape.z,output_blob_shape.w,output_blob_shape.dim);
		}

		(*outputs[i]).resize(output_blob_shape.x*output_blob_shape.y*output_blob_shape.z*output_blob_shape.w);

		status =ailiaGetBlobData(ailia, &(*outputs[i])[0], outputs[i]->size() * sizeof(float), output_blob_idx);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobData",ailiaGetErrorDetail(ailia));
			return status;
		}
	}

	return AILIA_STATUS_SUCCESS;
}

std::vector<float> calc_vad(AILIANetwork* net, std::vector<float> wave, int sampleRate, int nChannels, int nSamples)
{
	int batch = 1;
	int sequence = 1536;

	std::vector<float> input(batch * sequence);
	std::vector<float> sr(1);
	std::vector<float> h(2 * batch * 64);
	std::vector<float> c(2 * batch * 64);

	std::vector<float> conf;

	for (int s = 0; s < nSamples; s+=sequence){
		for (int i = 0; i < input.size(); i++){
			if (s + i < nSamples){
				input[i] = wave[s + i];
			}else{
				input[i] = 0;
			}
		}
		sr[0] = sampleRate;
		if (debug){
			PRINT_OUT("\n");
		}

		std::vector<float> *inputs[NUM_INPUTS];
		inputs[0] = &input;
		inputs[1] = &sr;
		inputs[2] = &h;
		inputs[3] = &c;

		std::vector<float> output(batch);
		
		std::vector<float> *outputs[NUM_OUTPUTS];
		outputs[0] = &output;
		outputs[1] = &h;
		outputs[2] = &c;

		forward(net, inputs, outputs);

		conf.push_back(output[0]);
	}

	return conf;
}

static int recognize_from_audio(AILIANetwork* net[MODEL_N])
{
	int status = AILIA_STATUS_SUCCESS;

	int sampleRate, nChannels, nSamples;
	std::vector<float> wave = read_wave_file(input_text.c_str(), &sampleRate, &nChannels, &nSamples);
	if (wave.size() == 0){
		PRINT_ERR("Input file not found (%s)\n", input_text.c_str());
		return AILIA_STATUS_ERROR_FILE_API;
	}

	// resmaple to 16k and 32k
	PRINT_OUT("sampleRate %d\n", sampleRate);

	/*
	gpt = T2SModel(t2s_encoder, t2s_first_decoder, t2s_stage_decoder,)
    gpt_sovits = GptSoVits(gpt, vits)
    ssl = SSLModel(ssl)

    input_audio = args.audio

    ref_phones = g2p(args.transcript)
    ref_seq = np.array([cleaned_text_to_sequence(ref_phones)], dtype=np.int64)

    text_phones = g2p(args.input)
    text_seq = np.array([cleaned_text_to_sequence(text_phones)], dtype=np.int64)

    # empty for ja or en
    ref_bert = np.zeros((ref_seq.shape[1], 1024), dtype=np.float32)
    text_bert = np.zeros((text_seq.shape[1], 1024), dtype=np.float32)
    
    vits_hps_data_sampling_rate = 32000

    zero_wav = np.zeros(
        int(vits_hps_data_sampling_rate * 0.3),
        dtype=np.float32,
    )
    wav16k, sr = librosa.load(input_audio, sr=16000)
    wav16k = np.concatenate([wav16k, zero_wav], axis=0)
    wav16k = wav16k[np.newaxis, :]
    ref_audio_16k = wav16k # hubertの入力のみpaddingする

    wav32k, sr = librosa.load(input_audio, sr=vits_hps_data_sampling_rate)
    wav32k = wav32k[np.newaxis, :]

    ssl_content = ssl.forward(ref_audio_16k)

    a = gpt_sovits.forward(ref_seq, text_seq, ref_bert, text_bert, wav32k, ssl_content)

    savepath = args.savepath
    logger.info(f'saved at : {savepath}')

    soundfile.write(savepath, a, vits_hps_data_sampling_rate)

    logger.info('Script finished successfully.')
	*/

	/*
	if (sampleRate != 16000){
		PRINT_OUT("sampleRate must be 16000 (actual %d)\n", sampleRate);
		return AILIA_STATUS_INVALID_ARGUMENT;
	}

	if (nChannels != 1){
		PRINT_OUT("nChannels must be 1 (actual %d)\n", nChannels);
		return AILIA_STATUS_INVALID_ARGUMENT;
	}

	std::vector<float> conf = calc_vad(net, wave, sampleRate, nChannels, nSamples);

	PRINT_OUT("Confidence :\n");
	for (int i = 0; i < conf.size(); i++){
		if (i < 10){
			PRINT_OUT("%f sec %f\n", i * sampleRate / 1536.0f, conf[i]);
		}
	}
	PRINT_OUT("\n");
	*/

	PRINT_OUT("Program finished successfully.\n");

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
		bool is_fp16 = (env->props & AILIA_ENVIRONMENT_PROPERTY_FP16) != 0;
		PRINT_OUT("env_id : %d type : %d name : %s", env->id, env->type, env->name);
		if (is_fp16){
			PRINT_OUT(" (Warning : FP16 backend is not worked this model)\n");
			continue;
		}
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

	status = recognize_from_audio(ailia);

	for (int i = 0; i < MODEL_N; i++){
		ailiaDestroy(ailia[i]);
	}

	return status;
}
