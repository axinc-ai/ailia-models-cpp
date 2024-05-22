/*******************************************************************
*
*    DESCRIPTION:
*      ailia Speech Sample Program
*    AUTHOR:
*      ax Inc.
*    DATE:2023/03/06
*
*******************************************************************/

#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>

#include "ailia.h"
#include "ailia_audio.h"

#include "ailia_speech.h"
#include "ailia_speech_util.h"

#include "wave_reader.h"

int intermediate_callback(void *handle, const char *text){
	printf("\r%s", text);
	fflush(stdout);
	return 0; // 1で中断
}

int get_text(struct AILIASpeech* net, bool live_mode){
	unsigned int count = 0;
	int status = ailiaSpeechGetTextCount(net, &count);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaSpeechGetTextCount Error %d\n", status);
		printf("%s\n", ailiaSpeechGetErrorDetail(net));
		return -1;
	}

	if (live_mode && count > 0){
		printf("\n");
	}

	for (unsigned int idx = 0; idx < count; idx++){
		AILIASpeechText text;
		status = ailiaSpeechGetText(net, &text, AILIA_SPEECH_TEXT_VERSION, idx);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechGetText Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}

		float cur_time = text.time_stamp_begin;
		float next_time = text.time_stamp_end;
		printf("[%02d:%02d.%03d --> %02d:%02d.%03d] ", (int)cur_time/60%60,(int)cur_time%60, (int)(cur_time*1000)%1000, (int)next_time/60%60,(int)next_time%60, (int)(next_time*1000)%1000);
		printf("[%0.4f] ", text.confidence);
		printf("%s\n", text.text);
	}

	return AILIA_STATUS_SUCCESS;
}

int update(struct AILIASpeech* net, const float *wave_buf, int nSamples, int nChannels, int sampleRate, int &push_i, unsigned int &complete, bool translate, bool live_mode, bool post_process){
	int status;

	// Push pcm input to queue
	int push_size = sampleRate;
	int push_samples = std::max(0, std::min(nSamples - push_i, push_size));
	if (push_samples >= 1){
		// Push pcm data
		status = ailiaSpeechPushInputData(net, &wave_buf[nChannels * push_i], nChannels, push_samples, sampleRate);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechPushInputData Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
		push_i += push_size;
	}else{
		// Finalize push pcm data
		status = ailiaSpeechFinalizeInputData(net);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechFinalizeInputData Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
	}
	
	// Transcribe
	while(true){
		// Check enough pcm exists in queue
		unsigned int buffered = 0;
		status = ailiaSpeechBuffered(net, &buffered);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechIsBuffered Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
		if (buffered == 0){
			break;	// Wait next data
		}

		// Process
		status = ailiaSpeechTranscribe(net);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechTranscribe Error %d\n", status);
		}
		if (status != AILIA_STATUS_SUCCESS){
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			if (status == AILIA_STATUS_LICENSE_NOT_FOUND){
				printf("License file not found.\n");
				printf("Please place license file.\n");
			}
			return -1;
		}
		
		// Get results
		status = get_text(net, live_mode);
		if (status != AILIA_STATUS_SUCCESS){
			return status;
		}

		// Post process
		if (post_process){
			status = ailiaSpeechPostProcess(net);
			if (status != AILIA_STATUS_SUCCESS){
				return status;
			}

			status = get_text(net, live_mode);
			if (status != AILIA_STATUS_SUCCESS){
				return status;
			}
		}
	}

	// Check all queued data processed
	if (push_samples == 0){
		status = ailiaSpeechComplete(net, &complete);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechIsBuffered Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
	}

	return AILIA_STATUS_SUCCESS;
}

int get_model_name(std::string &encoder, std::string &decoder, int &model_id, const char *model_type){
	bool model_detected = false;
	if (strcmp(model_type,"large") == 0 || strcmp(model_type,"large_v3") == 0){
		encoder = std::string("encoder_"+std::string(model_type)+".onnx");
		decoder = std::string("decoder_"+std::string(model_type)+"_fix_kv_cache.onnx");
	}else{
		encoder = std::string("encoder_"+std::string(model_type)+".opt3.onnx");
		decoder = std::string("decoder_"+std::string(model_type)+"_fix_kv_cache.opt3.onnx");
	}
	if (strcmp(model_type,"tiny") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_TINY;
		model_detected = true;
	}
	if (strcmp(model_type,"base") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_BASE;
		model_detected = true;
	}
	if (strcmp(model_type,"small") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_SMALL;
		model_detected = true;
	}
	if (strcmp(model_type,"medium") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_MEDIUM;
		model_detected = true;
	}
	if (strcmp(model_type,"large") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_LARGE;
		model_detected = true;
	}
	if (strcmp(model_type,"large_v3") == 0){
		model_id = AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_LARGE_V3;
		model_detected = true;
	}
	if (model_detected == false){
		return -1;
	}
	return AILIA_STATUS_SUCCESS;
}

int get_environment_id(const char *type) {
	unsigned int env_count;
	int status = ailiaGetEnvironmentCount(&env_count);
	if (status != AILIA_STATUS_SUCCESS) {
		printf("ailiaGetEnvironmentCount Failed %d", status);
		return -1;
	}

	int env_id = AILIA_ENVIRONMENT_ID_AUTO;
	for (unsigned int i = 0; i < env_count; i++) {
		AILIAEnvironment* env;
		status = ailiaGetEnvironment(&env, i, AILIA_ENVIRONMENT_VERSION);
		if (status != AILIA_STATUS_SUCCESS) {
			printf("ailiaGetEnvironment Failed %d", status);
			return -1;
		}

		printf("Environment ID:%d TYPE:%d NAME:%s\n", env->id, env->type, env->name);

		if (std::string(type).find("cpu") != std::string::npos && env->type == AILIA_ENVIRONMENT_TYPE_CPU) {
			env_id = env->id;
		}
		if (std::string(type).find("gpu") != std::string::npos && env->type == AILIA_ENVIRONMENT_TYPE_GPU) {
			env_id = env->id;
		}
		if (std::string(type).find("blas") != std::string::npos && env->type == AILIA_ENVIRONMENT_TYPE_BLAS) {
			env_id = env->id;
		}
	}

	if (env_id == AILIA_ENVIRONMENT_ID_AUTO){
		printf("Selected Environment:auto\n");
	}else{
		AILIAEnvironment* env;
		status = ailiaGetEnvironment(&env, env_id, AILIA_ENVIRONMENT_VERSION);
		if (status != AILIA_STATUS_SUCCESS) {
			printf("ailiaGetEnvironment Failed %d", status);
			return -1;
		}
		printf("Selected Environment:%s\n", env->name);
	}

	return env_id;
}

int set_options(struct AILIASpeech *net, const char *option){
	int status = AILIA_STATUS_SUCCESS;

	// Set prompt
	bool prompt = strcmp(option, "prompt") == 0;
	if (prompt){
		const char * prompt_text = u8"ハードウェア ソフトウェア";
		status = ailiaSpeechSetPrompt(net, prompt_text);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechSetPrompt Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
	}

	// Constraint
	bool constraint_char = strcmp(option, "constraint_char") == 0;
	if (constraint_char){
		const char * constraint_text = u8"1234567890,.億千万百";
		status = ailiaSpeechSetConstraint(net, constraint_text, AILIA_SPEECH_CONSTRAINT_CHARACTERS);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechSetConstraint Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
	}

	bool constraint_word = strcmp(option, "constraint_word") == 0;
	if (constraint_word){
		const char * constraint_text = u8"100億,100グラム";
		status = ailiaSpeechSetConstraint(net, constraint_text, AILIA_SPEECH_CONSTRAINT_WORDS);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechSetConstraint Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
	}

	// Dictionary
	bool dictionary = (strcmp(option, "dictionary") == 0);
	if (dictionary){
		status = ailiaSpeechOpenDictionaryFileA(net, "dict.csv", AILIA_SPEECH_DICTIONARY_TYPE_REPLACE);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechOpenDictionaryFileA Error %d\n", status);
		}
	}
	
	// Post Process
	bool t5 = (strcmp(option, "t5") == 0);
	if (t5){
		status = ailiaSpeechOpenPostProcessFileA(net, "t5_whisper_medical-encoder.obf.onnx", "t5_whisper_medical-decoder-with-lm-head.obf.onnx", "spiece.model", NULL, "医療用語の訂正: ", AILIA_SPEECH_POST_PROCESS_TYPE_T5);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechOpenPostProcessFileA Error %d\n", status);
		}
	}

	bool fugumt_en_ja = (strcmp(option, "fugumt_en_ja") == 0);
	if (fugumt_en_ja){
		status = ailiaSpeechOpenPostProcessFileA(net, "fugumt_en_ja_seq2seq-lm-with-past.onnx", NULL, "fugumt_en_ja_source.spm", "fugumt_en_ja_target.spm", NULL, AILIA_SPEECH_POST_PROCESS_TYPE_FUGUMT_EN_JA);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechOpenPostProcessFileA Error %d\n", status);
		}
	}
	
	bool fugumt_ja_en = (strcmp(option, "fugumt_ja_en") == 0);
	if (fugumt_ja_en){
		status = ailiaSpeechOpenPostProcessFileA(net, "fugumt_ja_en_encoder_model.onnx", "fugumt_ja_en_decoder_model.onnx", "fugumt_en_ja_source.spm", "fugumt_en_ja_target.spm", NULL, AILIA_SPEECH_POST_PROCESS_TYPE_FUGUMT_JA_EN);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechOpenPostProcessFileA Error %d\n", status);
		}
	}

	return status;
}

int main(int argc, char **argv){
	const char *input_path="./demo.wav";
	const char *model_type="small";
	const char *language="auto";
	const char *task="transcribe";
	const char *vad="vad_enable";
	const char *option="none";
	const char *env="auto";
	if(argc>=2){
		input_path=argv[1];
	}
	if(argc>=3){
		model_type=argv[2];
	}
	if(argc>=4){
		language=argv[3];
	}
	if(argc>=5){
		task=argv[4];
	}
	if(argc>=6){
		vad=argv[5];
	}
	if(argc>=7){
		option=argv[6];
	}
	if(argc>=8){
		env=argv[7];
	}
	printf("Usage ./ailia_speech_sample input.wav [base/tiny/small/medium/large/large_v3] [auto/ja] [transcribe/translate/live] [vad_enable/vad_disable] [none/silent_threshold/prompt/constraint_char/constraint_word/dictionary/t5/fugumt_en_ja/fugumt_ja_en] [auto/cpu/blas/gpu]\n");
	printf("Input path:%s\n", input_path);
	printf("Model type:%s\n", model_type);
	printf("Language type:%s\n", language);
	printf("Task:%s\n", task);
	printf("Vad:%s\n", vad);
	printf("Option:%s\n", option);
	printf("Env:%s\n", env);

	if (strcmp(task, "transcribe") != 0 && strcmp(task, "translate") != 0 && strcmp(task, "live") != 0){
		printf("task must be transcribe or translate or live\n");
		return -1;
	}
	if (strcmp(vad, "vad_enable") != 0 && strcmp(vad, "vad_disable") != 0){
		printf("vad must be transcribe or vad_enable or vad_disable\n");
		return -1;
	}
	if (strcmp(option, "none") != 0 && strcmp(option, "silent_threshold") != 0 && strcmp(option, "prompt") != 0 && strcmp(option, "vad") != 0 && strcmp(option, "constraint_char") != 0 && strcmp(option, "constraint_word") != 0 && strcmp(option, "dictionary") != 0 && strcmp(option, "t5") != 0 && strcmp(option, "fugumt_en_ja") != 0 && strcmp(option, "fugumt_ja_en") != 0){
		printf("option must be none or prompt or constraint_char or constraint_word or dictionary\n");
		return -1;
	}
	if (strcmp(env, "auto") != 0 && strcmp(env, "cpu") != 0 && strcmp(env, "blas") != 0 && strcmp(env, "gpu") != 0){
		printf("env must be auto or cput or blas or gpu\n");
		return -1;
	}

	// Get environment
	int env_id = get_environment_id(env);

	// Load wave file
	int sampleRate,nChannels,nSamples;
	std::vector<float> wave_buf = read_wave_file(input_path, &sampleRate, &nChannels, &nSamples);
	int status;
	if(wave_buf.size()==0){
		printf("wav file not found or could not open %s\n", input_path);
		return -1;
	}
	printf("Input wave sec %f\n", (float)nSamples/sampleRate);

	struct AILIASpeech* net;

	AILIASpeechApiCallback callback = ailiaSpeechUtilGetCallback();

	bool translate = (strcmp(task, "translate") == 0);
	bool live_mode = (strcmp(task, "live") == 0);

	int task_id = (translate) ? AILIA_SPEECH_TASK_TRANSLATE:AILIA_SPEECH_TASK_TRANSCRIBE;
	int flag = (live_mode) ? AILIA_SPEECH_FLAG_LIVE:AILIA_SPEECH_FLAG_NONE;
	int memory_mode = AILIA_MEMORY_REDUCE_CONSTANT | AILIA_MEMORY_REDUCE_CONSTANT_WITH_INPUT_INITIALIZER | AILIA_MEMORY_REUSE_INTERSTAGE;

	status = ailiaSpeechCreate(&net, AILIA_ENVIRONMENT_ID_AUTO, AILIA_MULTITHREAD_AUTO, memory_mode, task_id, flag, callback, AILIA_SPEECH_API_CALLBACK_VERSION);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaSpeechCreate Error %d\n", status);
		printf("%s\n", ailiaSpeechGetErrorDetail(net));
		return -1;
	}

	std::string encoder;
	std::string decoder;
	int model_id;
	status = get_model_name(encoder, decoder, model_id, model_type);
	if (status != AILIA_STATUS_SUCCESS){
		printf("unknown model type\n");
		ailiaSpeechDestroy(net);
		return -1;
	}

	status = ailiaSpeechOpenModelFileA(net, encoder.c_str(), decoder.c_str(), model_id);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaSpeechOpenModelFileA Error %d\n", status);
		printf("%s\n", ailiaSpeechGetErrorDetail(net));
		if (status == AILIA_STATUS_LICENSE_NOT_FOUND){
			printf("License file not found.\n");
			printf("Please place license file.\n");
		}
		return -1;
	}

	status = ailiaSpeechSetLanguage(net, language);
	if (status != AILIA_STATUS_SUCCESS){
		printf("ailiaSpeechSetLanguage Error %d\n", status);
		printf("%s\n", ailiaSpeechGetErrorDetail(net));
		return -1;
	}

	if (live_mode){ // You can also use setIntermediateCallback for normal mode
		status = ailiaSpeechSetIntermediateCallback(net, &intermediate_callback, NULL);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechSetIntermediateCallback Error %d\n", status);
			printf("%s\n", ailiaSpeechGetErrorDetail(net));
			return -1;
		}
	}

	bool vad_enable = (strcmp(vad, "vad_enable") == 0);
	if (vad_enable){
		status = ailiaSpeechOpenVadFileA(net, "silero_vad.onnx", AILIA_SPEECH_VAD_TYPE_SILERO);
		if (status != AILIA_STATUS_SUCCESS){
			printf("ailiaSpeechOpenVadFileA Error %d\n", status);
		}
	}

	status = set_options(net, option);

	const float THRESHOLD_VOLUME = 0.01f;
	const float THRESHOLD_VAD = 0.5f;
	const float SPEECH_SEC = 1.0f;
	const float NO_SPEECH_SEC = 1.0f;
	bool silent_threshold = strcmp(option, "silent_threshold") == 0;
	if (silent_threshold){
		if (vad_enable){
			status = ailiaSpeechSetSilentThreshold(net, THRESHOLD_VAD, SPEECH_SEC, NO_SPEECH_SEC);
		}else{
			status = ailiaSpeechSetSilentThreshold(net, THRESHOLD_VOLUME, SPEECH_SEC, NO_SPEECH_SEC);
		}
	}

	bool post_process = (strcmp(option, "t5") == 0) || (strcmp(option, "fugumt_en_ja") == 0) || (strcmp(option, "fugumt_ja_en") == 0);

	int push_i = 0;
	while(true){
		unsigned int complete = 0;
		status = update(net, &wave_buf[0], nSamples, nChannels, sampleRate, push_i, complete, translate, live_mode, post_process);
		if (status != AILIA_STATUS_SUCCESS){
			return -1;
		}
		if (complete == 1){
			break;
		}
	}

	ailiaSpeechDestroy(net);
	return 0;
}


