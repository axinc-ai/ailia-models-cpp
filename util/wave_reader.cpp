/*******************************************************************
*
*    DESCRIPTION:
*      Wave file reader
*    AUTHOR:
*      ax Inc.
*    DATE:2022/07/16
*
*******************************************************************/

#include "wave_reader.h"

#pragma pack(1)
struct FileHeader{
	char filetype[4];
	unsigned int filesize;
	char rifftype[4];
};

struct FormatHeader{
	unsigned int size;
	unsigned short id;
	unsigned short channel_n;
	unsigned int sampling_rate;
	unsigned int data_speed;
	unsigned short block_size;
	unsigned short bit_per_sample;
};
#pragma pack()

struct WaveReader{
	FileHeader fh;
	FormatHeader format;
	FILE *fp;
	int data_n;
};

static const int STATUS_SUCCESS = 0;
static const int STATUS_BROKEN = -1;
static const int STATUS_ERROR_FILE_API = -2;

namespace{

unsigned int get_tag_size(FILE *fp){
	unsigned int size=0;
	fread(&size,4,1,fp);	
	if(size%2==1){	// size word padding
		size++;
	}
	return size;
}

int skip_tag(FILE *fp,const char *target_tag1,const char *target_tag2){
	while(1){
		char type[4];
		fread(type,4,1,fp);
		
		if(strncmp(type,target_tag1,4)==0 || strncmp(type,target_tag2,4)==0){
			return STATUS_SUCCESS;
		}
		
		unsigned int size=get_tag_size(fp);
		fseek(fp,size,SEEK_CUR);
		if(feof(fp)){
			return STATUS_BROKEN;
		}
	}
	return STATUS_SUCCESS;
}

int read_file_header(FileHeader *fh, FILE *fp){
	fread(fh,sizeof(struct FileHeader),1,fp);
	if(strncmp(fh->filetype,"RIFF",4)!=0 && strncmp(fh->filetype,"riff",4)!=0){
		return STATUS_BROKEN;
	}
	if(strncmp(fh->rifftype,"WAVE",4)!=0 && strncmp(fh->rifftype,"wave",4)!=0){
		return STATUS_BROKEN;
	}
	return STATUS_SUCCESS;
}

int read_format_header(FormatHeader *format, FILE *fp){
	int status=skip_tag(fp,"FMT ","fmt ");
	if(status!=STATUS_SUCCESS){
		return status;
	}

	fread(format,sizeof(FormatHeader),1,fp);
	
	unsigned int size=format->size;
	size-=16;
	fseek(fp,size,SEEK_CUR);
	
	return STATUS_SUCCESS;
}

int read_data_n(WaveReader *instance){
	int ret=skip_tag(instance->fp,"DATA","data");
	if(ret){
		return ret;
	}
	
	fread(&instance->data_n,4,1,instance->fp);
	instance->data_n/=(instance->format.bit_per_sample/8);
	
	return STATUS_SUCCESS;
}

int open_file_core(WaveReader *instance){
	int ret=read_file_header(&instance->fh, instance->fp);
	if(ret){
		return ret;
	}
	
	ret=read_format_header(&instance->format, instance->fp);
	if(ret){
		return ret;
	}
	
	ret=read_data_n(instance);
	if(ret){
		return ret;
	}
	
	return STATUS_SUCCESS;
}

int open_a(WaveReader *instance, const char *path){
	if(path==NULL){
		return STATUS_ERROR_FILE_API;
	}
	instance->fp=fopen(path,"rb");
	if(instance->fp==NULL){
		return STATUS_ERROR_FILE_API;
	}
	int status=open_file_core(instance);
	return status;
}

void close(WaveReader *instance){
	if(instance->fp!=NULL){
		fclose(instance->fp);
		instance->fp=NULL;
	}
}

} // namespace

std::vector<float> read_wave_file(const char *path, int *sampleRate, int *nChannels, int *nSamples){
	//Create instance
	WaveReader instance;
	instance.fp=NULL;
	memset(&instance.format,0,sizeof(instance.format));
	instance.data_n=0;

	//Open file
	int status = open_a(&instance, path);
	std::vector<float> buf;
	if(status!=STATUS_SUCCESS){
		return buf;
	}

	//Format conversion
	if(instance.format.bit_per_sample==24){
		buf.resize(instance.data_n);
		std::vector<unsigned char> plane_buf;
		plane_buf.resize(instance.data_n * 3);
		fread(&plane_buf[0],3*instance.data_n,1,instance.fp);
		for(int i=0;i<instance.data_n;i++){
			int v = (int)((plane_buf[i*3+2]<<24) | (plane_buf[i*3+1]<<16) | (plane_buf[i*3+0]<<8));
			buf[i]=v * 1.0f / (1<<31);
		}
	}else{
		if(instance.format.bit_per_sample==16){
			buf.resize(instance.data_n);
			std::vector<short> plane_buf;
			plane_buf.resize(instance.data_n);
			fread(&plane_buf[0],sizeof(short)*instance.data_n,1,instance.fp);
			for(int i=0;i<instance.data_n;i++){
				buf[i]=plane_buf[i] * 1.0f / (1<<15);
			}
		}else{
			printf("unknown bit per sample\n");
		}
	}

	//Set return value
	*sampleRate = instance.format.sampling_rate;
	*nChannels = instance.format.channel_n;
	*nSamples = instance.data_n / instance.format.channel_n;

	close(&instance);

	return buf;
}