/*******************************************************************
*
*    DESCRIPTION:
*      AILIA feature extraction library
*    AUTHOR:
*      AXELL Corporation
*    DATE:June 24, 2019
*
*******************************************************************/

#if       !defined(INCLUDED_AILIA_FEATURE_EXTRACTOR)
#define            INCLUDED_AILIA_FEATURE_EXTRACTOR

/* Core libraries */

#include "ailia.h"
#include "ailia_format.h"

/* Calling conventions */

#ifdef __cplusplus
extern "C" {
#endif

	/****************************************************************
	* Feature extraction instance
	**/

	struct AILIAFeatureExtractor;

	/****************************************************************
	* Distance setting
	**/

	#define AILIA_FEATURE_EXTRACTOR_DISTANCE_L2NORM (0)	/* L2 norm */

	/**
	*  Creates a feature extraction instance.
	*    Arguments:
	*      fextractor - A feature extraction instance pointer
	*      net        - A network instance pointer
	*      format     - The network image format (AILIA_NETWORK_IMAGE_FORMAT_*)
	*      channel    - The network image channel (AILIA_NETWORK_IMAGE_CHANNEL_*)
	*      range      - The network image range (AILIA_NETWORK_IMAGE_RANGE_*)
	*      layer_name - The name of the layer corresponding to the feature (fc1 for VGG16 and NULL for the last layer)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaCreateFeatureExtractor(struct AILIAFeatureExtractor **fextractor, struct AILIANetwork *net, unsigned int format, unsigned int channel, unsigned int range, const char *layer_name);

	/**
	*  It destroys the feature extraction instance.
	*    Arguments:
	*      fextractor - A feature extraction instance pointer
	*/
	void AILIA_API ailiaDestroyFeatureExtractor(struct AILIAFeatureExtractor *fextractor);

	/**
	*  Performs feature extraction.
	*    Arguments:
	*      fextractor                  - A feature extraction instance pointer
	*      dst                         - A pointer to the storage location of the feature (numeric type)
	*      dst_size                    - The size of the dst (bytes)
	*      src                         - Image data (32 bpp)
	*      src_stride                  - The number of bytes in 1 line
	*      src_width                   - Image width
	*      src_height                  - Image height
	*      src_format                  - AILIA_IMAGE_FORMAT_*
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaFeatureExtractorCompute(struct AILIAFeatureExtractor *fextractor, void *dst, unsigned int dst_size, const void *src, unsigned int src_stride, unsigned int src_width, unsigned int src_height, unsigned int src_format);

	/**
	*  Computes distances in feature space.
	*    Arguments:
	*      fextractor                  - A feature extraction instance pointer
	*      distance                    - A distance in feature space
	*      distance_type               - The type of the distance in feature space
	*      feature1                    - A pointer to the storage location of one feature (numeric type)
	*      feature1_size               - The size of the dst (bytes)
	*      feature2                    - A pointer to the storage location of the other feature (numeric type)
	*      feature2_size               - The size of the dst (bytes)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaFeatureExtractorMatch(struct AILIAFeatureExtractor *fextractor, float *distance, unsigned int distace_type, const void *feature1, unsigned int feature1_size, const void *feature2, unsigned int feature2_size);

#ifdef __cplusplus
}
#endif
#endif /* !defined(INCLUDED_AILIA_FEATURE_EXTRACTOR) */
