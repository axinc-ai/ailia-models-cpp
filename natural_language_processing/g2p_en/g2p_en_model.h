/*******************************************************************
*
*    DESCRIPTION:
*      AILIA G2P EN model
*    AUTHOR:
*
*    DATE:2024/06/26
*
*******************************************************************/

#pragma once

#include <vector>
#include "ailia.h"
#include "g2p_en_averaged_perceptron.h"

namespace ailiaG2P{

class G2PEnModel{
private:
	static const int MODEL_N = 2;

	static const int MODEL_ENCODER = 0;
	static const int MODEL_DECODER = 1;

	AILIANetwork* net[MODEL_N];
	AveragedPerceptron model;

	std::vector<std::string> predict(const std::string &word);

	std::unordered_map<std::string, std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>> homograph2features;
	std::unordered_map<std::string, std::vector<std::string>> cmudict;

public:
	void open(int env_id, const char *model_encoder_a, const wchar_t *model_encoder_w, const char *model_decoder_a, const wchar_t *model_decoder_w, const char *homograph_a, const wchar_t *homograph_w, const char *cmudict_a, const wchar_t *cmudict_w);
	void import_from_text(const char *weight_a, const wchar_t *weight_w, const char *tagdict_a, const wchar_t *tagdict_w, const char *classes_a, const wchar_t *classes_w);
	void close(void);
	std::vector<std::string> compute(std::string text);
};

}