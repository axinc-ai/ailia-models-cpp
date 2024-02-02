/*******************************************************************
*
*    DESCRIPTION:
*      AILIA clap sample
*    AUTHOR:
*
*    DATE:2024/01/25
*
*******************************************************************/
#include <string.h>
#include <math.h>

#include "clap_utils.h"
#include "ailia.h"
#include "ailia_audio.h"
extern bool debug;

static std::vector<float> get_mel_ailia(std::vector<float>& audio_data, const AUDIO_CONFIG& audio_cfg)
{
    const int center = AILIA_AUDIO_STFT_CENTER_ENABLE;
    const int mel_n = 64;
    int status;
    int frame_n;
    std::vector<float> mel;
    
    status = ailiaAudioGetFrameLen(&frame_n, audio_data.size(), audio_cfg.window_size, audio_cfg.hop_size, center);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaAudioGetFrameLen failed %d\n", status);
        return mel;
    }
    if(debug){
        PRINT_OUT("frame_n = %d\n", frame_n);
    }
    
    mel = std::vector<float>(mel_n * frame_n);  // [mel_n][frame_n]
    status = ailiaAudioGetMelSpectrogram(
        &mel[0],
        &audio_data[0],
        audio_data.size(),
        audio_cfg.sample_rate,
        audio_cfg.window_size,
        audio_cfg.hop_size,
        audio_cfg.window_size,
        AILIA_AUDIO_WIN_TYPE_HANN,
        frame_n,
        center,
        2.0,    // power
        AILIA_AUDIO_FFT_NORMALIZE_NONE,
        audio_cfg.fmin,
        audio_cfg.fmax,
        mel_n,
        AILIA_AUDIO_MEL_NORMALIZE_NONE,
        AILIA_AUDIO_MEL_SCALE_FORMULA_HTK
    );
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaAudioGetMelSpectrogram failed %d\n", status);
        return mel;
    }

    // amplitude_to_db
    const float ref = 1.0;
    const float amin = 1e-10;
    for(auto v=mel.begin(); v!=mel.end(); ++v){
        float s = (*v) * (*v);
        if(s >= 0 && s < amin) s = amin;
        if(s < 0 && s > -amin) s = -amin;
        *v = 10 * log10f(s / ref);
    }
    
    // transpose(1, 0):  [mel_n][frame_n] to [frame_n][mel_n]
    std::vector<float> mel_t(mel_n * frame_n);
    float* dst = &mel_t[0];
    for(int j=0; j<frame_n; j++){
        for(int i=0; i<mel_n; i++){
            *dst++ = mel[i * frame_n + j];
        }
    }
    return mel_t;
}

std::vector<float> get_audio_features(std::vector<float>& audio_data, unsigned int max_len, 
    std::string data_truncating, std::string data_filling, const AUDIO_CONFIG& audio_cfg)
{
    std::vector<float> mel_fusion;
    
    if(audio_data.size() > max_len){
        
    }
    else{   // padding
        if(audio_data.size() < max_len){
            std::vector<float> new_audio_data(max_len, 0);
            if(data_filling == "repeatpad"){
                int n_repeat = max_len / audio_data.size();
                for(int i=0; i<n_repeat; i++){
                    memcpy(&new_audio_data[i * audio_data.size()], &audio_data[0], audio_data.size() * sizeof(float));
                }
            }
            else{
                PRINT_ERR("Not support data_filling: %s\n", data_filling.c_str());
                return mel_fusion;
            }
            audio_data = new_audio_data;
        }
        if(data_truncating == "fusion"){
            std::vector<float> mel = get_mel_ailia(audio_data, audio_cfg);
            if(mel.size() < 1) return mel_fusion;
            mel_fusion = std::vector<float>(mel.size() * 4);
            for(int i=0; i<4; i++){
                memcpy(&mel_fusion[i * mel.size()], &mel[0], mel.size() * sizeof(float));
            }
        }
        else{
            PRINT_ERR("Not support data_truncating: %s\n", data_truncating.c_str());
            return mel_fusion;
        }
    }
    
    return mel_fusion;
}
