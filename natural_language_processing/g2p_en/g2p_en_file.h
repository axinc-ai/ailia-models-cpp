/*******************************************************************
*
*    DESCRIPTION:
*      AILIA G2P EN file
*    AUTHOR:
*
*    DATE:2024/06/26
*
*******************************************************************/

#pragma once

#include <vector>

namespace ailiaG2P{

std::vector<char> load_file_a(const char *path_a);
std::vector<char> load_file_w(const wchar_t *path_w);

}