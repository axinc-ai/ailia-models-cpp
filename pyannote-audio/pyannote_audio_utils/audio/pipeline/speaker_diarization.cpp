#include <iostream>
#include <string>

#include "diarization.h"

int speaker_diarization_test(int input)
{
    int result = diarization_test(input);
    return result;
}


// class SpeakerDiarization : SpeakerDiarizationMixin, Pipeline
class SpeakerDiarization : SpeakerDiarizationMixin
{

};