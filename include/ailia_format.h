/*******************************************************************
*
*    DESCRIPTION:
*      AILIA format definition and conversion
*    AUTHOR:
*      AXELL Corporation
*    DATE:August 27, 2019
*
*******************************************************************/

#if       !defined(INCLUDED_AILIA_FORMAT)
#define            INCLUDED_AILIA_FORMAT

/* Calling conventions */

#if defined(_WIN64) || defined(_M_X64) || defined(__amd64__) || defined(__x86_64__) || defined(__APPLE__) || defined(__ANDROID__) || defined(ANDROID) || defined(__linux__)
#define AILIA_API
#else
#define AILIA_API __stdcall
#endif

#ifdef __cplusplus
extern "C" {
#endif

	/****************************************************************
	* Input image formats
	**/

	#define AILIA_IMAGE_FORMAT_RGBA        (0x00)	//RGBA order
	#define AILIA_IMAGE_FORMAT_BGRA        (0x01)	//BGRA order

	#define AILIA_IMAGE_FORMAT_RGBA_B2T    (0x10)	//RGBA order (Bottom to top)
	#define AILIA_IMAGE_FORMAT_BGRA_B2T    (0x11)	//BGRA order (Bottom to top)

	/****************************************************************
	* Network image formats
	**/

	#define AILIA_NETWORK_IMAGE_FORMAT_BGR               (0)	//BGR order
	#define AILIA_NETWORK_IMAGE_FORMAT_RGB               (1)	//RGB order
	#define AILIA_NETWORK_IMAGE_FORMAT_GRAY              (2)	//GrayScale (1ch)
	#define AILIA_NETWORK_IMAGE_FORMAT_GRAY_EQUALIZE     (3)	//EqualizedGrayScale (1ch)

	#define AILIA_NETWORK_IMAGE_CHANNEL_FIRST            (0)	//DCYX order
	#define AILIA_NETWORK_IMAGE_CHANNEL_LAST             (1)	//DYXC order

	#define AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_INT8      (0)	//0 to 255
	#define AILIA_NETWORK_IMAGE_RANGE_SIGNED_INT8        (1)	//-128 to 127
	#define AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_FP32      (2)	//0.0 to 1.0
	#define AILIA_NETWORK_IMAGE_RANGE_SIGNED_FP32        (3)	//-1.0 to 1.0


	/**
	*  Converts image formats.
	*    Arguments:
	*      dst                  - The storage location of the image after conversion (numeric type; a size of sizeof(float) * dst_width * dst_height * dst_channel [bytes] or more must be allocated.)
	*      dst_width            - The width of the image after conversion
	*      dst_height           - The height of the image after conversion
	*      dst_format           - The format of the image after conversion (AILIA_NETWORK_IMAGE_FORMAT_*)
	*      dst_channel          - The channel order of the image after conversion (AILIA_NETWORK_IMAGE_CHANNEL_*)
	*      dst_range            - The range of the image after conversion (AILIA_NETWORK_IMAGE_RANGE_*)
	*      src                  - The storage location of the source image (32 bpp)
	*      src_stride           - The line byte number of the source image
	*      src_width            - The width of the source image
	*      src_height           - The height of the source image
	*      src_format           - The format of the source image (AILIA_IMAGE_FORMAT_*)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*       This function converts image formats. If dst_format is set to AILIA_NETWORK_IMAGE_FORMAT_BGR or AILIA_NETWORK_IMAGE_FORMAT_RGB,
	*      the number of channels is 3, otherwise if set to AILIA_NETWORK_IMAGE_FORMAT_GRAY, the number of channels is 1.
	*/

	int AILIA_API ailiaFormatConvert(void *dst, unsigned int dst_width, unsigned int dst_height, unsigned int dst_format, unsigned int dst_channel, unsigned int dst_range, const void *src, int src_stride, unsigned int src_width, unsigned int src_height, unsigned int src_format);

#ifdef __cplusplus
}
#endif
#endif /* !defined(INCLUDED_AILIA_FORMAT) */
