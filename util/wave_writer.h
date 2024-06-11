/*******************************************************************
*
*    DESCRIPTION:
*      Wave file writer
*    AUTHOR:
*      ax Inc.
*    DATE:2024/05/01
*
*******************************************************************/

#pragma once

#include <stdio.h>
#include <vector>

void write_wave_file(const char *path, std::vector<float> data, int sampling_rate);
