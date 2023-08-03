/*******************************************************************
*
*    DESCRIPTION:
*      Wave file reader
*    AUTHOR:
*      ax Inc.
*    DATE:2022/07/16
*
*******************************************************************/

#pragma once

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>

std::vector<float> read_wave_file(const char *path, int *sampleRate, int *nChannels, int *nSamples);
