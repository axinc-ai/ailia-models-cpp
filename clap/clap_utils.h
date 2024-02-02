/*******************************************************************
*
*    DESCRIPTION:
*      AILIA clap sample
*    AUTHOR:
*
*    DATE:2024/01/25
*
*******************************************************************/
#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

struct AUDIO_CONFIG {
    int audio_length;
    int clip_samples;
    int mel_bins;
    int sample_rate;
    int window_size;
    int hop_size;
    int fmin;
    int fmax;
    int class_num;
    std::string model_type;
    std::string model_name;

    AUDIO_CONFIG(){
        audio_length = 1024;
        clip_samples = 480000;
        mel_bins = 64;
        sample_rate = 48000;
        window_size = 1024;
        hop_size = 480;
        fmin = 50;
        fmax = 14000;
        class_num = 527;
        model_type = "HTSAT";
        model_name = "tiny";
    }
};

std::vector<float> get_audio_features(std::vector<float>& audio_data, unsigned int max_len, 
    std::string data_truncating, std::string data_filling, const AUDIO_CONFIG& audio_cfg);
