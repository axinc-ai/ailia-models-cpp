#include <stdio.h>
#include <stdlib.h>


#if defined(_WIN32) || defined(_WIN64)
// for Windows

bool check_file_existance(const char* path)
{
    FILE* fp;

    if ((fopen_s(&fp, path, "rb")) == 0) {
        fclose(fp);
        return true;
    }
    else {
        return false;
    }
}

#else
// for Linux and MacOS

bool check_file_existance(const char* path)
{
    FILE* fp;

    if ((fp = fopen(path, "rb")) != NULL) {
        fclose(fp);
        return true;
    }
    else {
        return false;
    }
}

#endif
