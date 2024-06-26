/*******************************************************************
*
*    DESCRIPTION:
*      AILIA G2P EN file
*    AUTHOR:
*
*    DATE:2024/06/26
*
*******************************************************************/

#include <iostream>
#include <regex>
#include <string>
#include <algorithm>
#include <map>
#include <sstream>

using namespace std;

namespace ailiaG2P{
	
std::vector<char> load_file_a(const char *path_a){
    FILE* fp = fopen(path_a, "rb");
    if (fp == NULL) {
        throw std::runtime_error("File could not open");
    }
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> buffer(fileSize);
    fread(buffer.data(), 1, fileSize, fp);
    fclose(fp);
	return buffer;
}

std::vector<char> load_file_w(const wchar_t *path_w){
#ifdef WIN32
    FILE* fp = _wfopen(path_w, "rb");
    if (fp == NULL) {
        throw std::runtime_error("File could not open");
    }
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> buffer(fileSize);
    fread(buffer.data(), 1, fileSize, fp);
    fclose(fp);
	return buffer;
#else
	throw std::runtime_error("Not implemented");
#endif
}

}
