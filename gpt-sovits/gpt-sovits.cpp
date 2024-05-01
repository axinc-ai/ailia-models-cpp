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

bool debug = true;


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

#define MODEL_SSL 0

const char *MODEL_NAME[5] = {"nahida_cnhubert.onnx", "nahida_t2s_encoder.onnx", "nahida_t2s_fsdec.onnx", "nahida_t2s_sdec.onnx", "nahida_vits.onnx"};

static bool benchmark  = false;
static int args_env_id = -1;

std::string input_text = "reference_audio_captured_by_ax.wav";

#define REF_PHONES_SIZE 37
#define TEXT_PHONES_SIZE 72

const char * REF_PHONES[REF_PHONES_SIZE] = {"m", "i", "z", "u", "o", "m", "a", "r", "e", "e", "sh", "i", "a", "k", "a", "r", "a", "k", "a", "w", "a", "n", "a", "k", "U", "t", "e", "w", "a", "n", "a", "r", "a", "n", "a", "i", "."};
const char *TEXT_PHONES[TEXT_PHONES_SIZE] = {"e", "i", "e", "cl", "k", "U", "s", "u", "k", "a", "b", "u", "sh", "I", "k", "i", "g", "a", "i", "sh", "a", "d", "e", "w", "a", "e", "e", "a", "i", "n", "o", "j", "i", "ts", "u", "y", "o", "o", "k", "a", "n", "o", "t", "a", "m", "e", "n", "o", "g", "i", "j", "u", "ts", "u", "o", "k", "a", "i", "h", "a", "ts", "u", "sh", "I", "t", "e", "i", "m", "a", "s", "U", "."};

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
	throw(func);
}

struct AILIATensor{
	std::vector<float> data;
	AILIAShape shape;
};

std::vector<AILIATensor> forward(AILIANetwork *ailia, std::vector<AILIATensor> inputs){
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
			printf("input blob shape %d %d %d %d dims %d\n",inputs[i].shape.x,inputs[i].shape.y,inputs[i].shape.z,inputs[i].shape.w,inputs[i].shape.dim);
		}

		status = ailiaSetInputBlobShape(ailia,&inputs[i].shape,input_blob_idx,AILIA_SHAPE_VERSION);
		if(status!=AILIA_STATUS_SUCCESS){
			setErrorDetail("ailiaSetInputBlobShape",ailiaGetErrorDetail(ailia));
		}

		status = ailiaSetInputBlobData(ailia, &(inputs[i].data)[0], inputs[i].data.size() * sizeof(float), input_blob_idx);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaSetInputBlobData",ailiaGetErrorDetail(ailia));
		}
	}

	status = ailiaUpdate(ailia);
	if (status != AILIA_STATUS_SUCCESS) {
		setErrorDetail("ailiaUpdate",ailiaGetErrorDetail(ailia));
	}

	std::vector<AILIATensor> outputs;
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
			printf("output_blob_shape %d %d %d %d dims %d\n",output_blob_shape.x,output_blob_shape.y,output_blob_shape.z,output_blob_shape.w,output_blob_shape.dim);
		}

		AILIATensor tensor;
		tensor.data.resize(output_blob_shape.x*output_blob_shape.y*output_blob_shape.z*output_blob_shape.w);

		status = ailiaGetBlobData(ailia, &tensor.data[0], tensor.data.size() * sizeof(float), output_blob_idx);
		if (status != AILIA_STATUS_SUCCESS) {
			setErrorDetail("ailiaGetBlobData",ailiaGetErrorDetail(ailia));
		}

		outputs.push_back(tensor);
	}

	return outputs;
}

static std::vector<float> resample(std::vector<float> pcm, int targetSampleRate, int sampleRate, int nChannels)
{
	if (nChannels == 2){
		for (int i = 0; i < pcm.size() / 2; i++){
			pcm[i] = (pcm[i*2] + pcm[i*2+1])/2;
		}
		pcm.resize(pcm.size() / 2);
	}

	if(sampleRate != targetSampleRate){
		int dst_n = 0;
		int status = ailiaAudioGetResampleLen(&dst_n, targetSampleRate, pcm.size(), sampleRate);
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaAudioGetResampleLen failed %d\n", status);
			throw;
		}
		std::vector<float> new_audio_waveform(dst_n);
		status = ailiaAudioResample(&new_audio_waveform[0], &pcm[0], targetSampleRate, dst_n, sampleRate, pcm.size());
		if (status != AILIA_STATUS_SUCCESS) {
			PRINT_ERR("ailiaAudioResample failed %d\n", status);
			throw;
		}
		return new_audio_waveform;
	}

	return pcm;
}

static std::vector<int> cleaned_text_to_sequence(const char ** data, int size){
	std::vector<int> sequence;
	for (int i = 0; i < size; i++){
		for (int s = 0; s < SYMBOLS_N; s++){
			if (strcmp(data[i], SYMBOLS[s]) == 0){
				if (debug){
					PRINT_OUT("%d ", s);
				}
				sequence.push_back(s);
				break;
			}
		}
	}
	if (debug){
		PRINT_OUT("\n");
	}
	return sequence;
}

static AILIATensor ssl_forward(std::vector<float> ref_audio_16k, AILIANetwork* net)
{
	std::vector<AILIATensor> inputs;
	AILIATensor tensor;
	tensor.data = ref_audio_16k;
	tensor.shape.x = ref_audio_16k.size();
	tensor.shape.y = 1;
	tensor.shape.z = 1;
	tensor.shape.w = 1;
	tensor.shape.dim = 2;
	inputs.push_back(tensor);
	std::vector<AILIATensor> outputs = forward(net, inputs);
	return outputs[0];
}

static std::vector<AILIATensor> t2s_forward(std::vector<int> ref_seq, std::vector<int> text_seq, std::vector<float> ref_bert, std::vector<float> text_bert, AILIATensor ssl_content, AILIANetwork *net[MODEL_N]){
        int hz = 50;
        int max_sec = 54;
        int top_k = 5;
        int early_stop_num = hz * max_sec;

		std::vector<AILIATensor> outputs;
		return outputs;

        //int sess_encoder = sess_encoder
        //int sess_fsdec = sess_fsdec
        //int sess_sdec = sess_sdec
/*
    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = self.early_stop_num

        top_k = np.array([5], dtype=np.int64)
        top_p = np.array([1.0], dtype=np.float32)
        temperature = np.array([1.0], dtype=np.float32)
        repetition_penalty = np.array([1.35], dtype=np.float32)

        EOS = 1024

        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            x, prompts = self.sess_encoder.run(None, {"ref_seq":ref_seq, "text_seq":text_seq, "ref_bert":ref_bert, "text_bert":text_bert, "ssl_content":ssl_content})
        else:
            x, prompts = self.sess_encoder.run({"ref_seq":ref_seq, "text_seq":text_seq, "ref_bert":ref_bert, "text_bert":text_bert, "ssl_content":ssl_content})
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tsencoder processing time {} ms".format(end-start))

        prefix_len = prompts.shape[1]

        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            y, k, v, y_emb, x_example = self.sess_fsdec.run(None, {"x":x, "prompts":prompts, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
        else:
            y, k, v, y_emb, x_example = self.sess_fsdec.run({"x":x, "prompts":prompts, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tfsdec processing time {} ms".format(end-start))

        stop = False
        for idx in range(1, 1500):
            if args.benchmark:
                start = int(round(time.time() * 1000))
            if args.onnx:
                y, k, v, y_emb, logits, samples = self.sess_sdec.run(None, {"iy":y, "ik":k, "iv":v, "iy_emb":y_emb, "ix_example":x_example, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
            else:
                y, k, v, y_emb, logits, samples = self.sess_sdec.run({"iy":y, "ik":k, "iv":v, "iy_emb":y_emb, "ix_example":x_example, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
            if args.benchmark:
                end = int(round(time.time() * 1000))
                logger.info("\tsdec processing time {} ms".format(end-start))
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if np.argmax(logits, axis=-1)[0] == EOS or samples[0, 0] == EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0

        return y[np.newaxis, :, -idx:-1]
*/
}

/*
        pred_semantic = self.t2s.forward(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx or args.onnx_vits:
            audio1 = self.sess.run(None, {
                "text_seq" : text_seq,
                "pred_semantic" : pred_semantic, 
                "ref_audio" : ref_audio
            })
        else:
            audio1 = self.sess.run({
                "text_seq" : text_seq,
                "pred_semantic" : pred_semantic, 
                "ref_audio" : ref_audio
            })
			*/
static int recognize_from_audio(AILIANetwork* net[MODEL_N])
{
	int status = AILIA_STATUS_SUCCESS;

	int sampleRate, nChannels, nSamples;
	std::vector<float> wave = read_wave_file(input_text.c_str(), &sampleRate, &nChannels, &nSamples);
	if (wave.size() == 0){
		PRINT_ERR("Input file not found (%s)\n", input_text.c_str());
		return AILIA_STATUS_ERROR_FILE_API;
	}

	// get sequence
	PRINT_OUT("ref_seq\n");
	std::vector<int> ref_seq = cleaned_text_to_sequence(REF_PHONES, REF_PHONES_SIZE);
	PRINT_OUT("text_seq\n");
	std::vector<int> text_seq = cleaned_text_to_sequence(TEXT_PHONES, TEXT_PHONES_SIZE);

	const int BERT_DIM = 1024;
	std::vector<float> ref_bert(ref_seq.size() * BERT_DIM);
	std::vector<float> text_bert(text_seq.size() * BERT_DIM);

	// resmaple to 16k and 32k
	const int vits_hps_data_sampling_rate = 32000;
	std::vector<float> zero_wav(vits_hps_data_sampling_rate * 0.3);
	std::vector<float> wav16k = resample(wave, 16000, sampleRate, nChannels);
	std::vector<float> ref_audio_16k = wav16k;
	ref_audio_16k.insert(ref_audio_16k.end(), zero_wav.begin(), zero_wav.end());
	
	std::vector<float> wav32k = resample(wave, vits_hps_data_sampling_rate, sampleRate, nChannels);

	// ssl
	AILIATensor ssl_content = ssl_forward(ref_audio_16k, net[MODEL_SSL]);

	// t2s
	std::vector<AILIATensor> t2s_content = t2s_forward(ref_seq, text_seq, ref_bert, text_bert, ssl_content, net);

	/*
	a = gpt_sovits.forward(ref_seq, text_seq, ref_bert, text_bert, wav32k, ssl_content)

	savepath = args.savepath
	logger.info(f'saved at : {savepath}')

	soundfile.write(savepath, a, vits_hps_data_sampling_rate)

	logger.info('Script finished successfully.')
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
