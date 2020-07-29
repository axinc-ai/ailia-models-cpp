/*******************************************************************
*
*    DESCRIPTION:
*      AILIA object classification library
*    AUTHOR:
*      AXELL Corporation
*    DATE:June 24, 2019
*
*******************************************************************/

#if       !defined(INCLUDED_AILIA_CLASSIFIER)
#define            INCLUDED_AILIA_CLASSIFIER

/* Core libraries */

#include "ailia.h"
#include "ailia_format.h"

/* Calling conventions */

#ifdef __cplusplus
extern "C" {
#endif

	/****************************************************************
	* Classifier instance
	**/

	struct AILIAClassifier;

	/****************************************************************
	* Classification information
	**/

	#define AILIA_CLASSIFIER_CLASS_VERSION (1)

	typedef struct _AILIAClassifierClass {
		int category;	// Classification category number
		float prob;		// Estimated probability (0 to 1)
	}AILIAClassifierClass;

	/**
	*  Creates a classifier instance.
	*    Arguments:
	*      classifier - A pointer to a classifier instance pointer
	*      net        - A network instance pointer
	*      format     - The network image format (AILIA_NETWORK_IMAGE_FORMAT_*)
	*      channel    - The network image channel (AILIA_NETWORK_IMAGE_CHANNEL_*)
	*      range      - The network image range (AILIA_NETWORK_IMAGE_RANGE_*)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*   Description:
	*     Creates a network instance.
	*     If the inference backend is set to automatic, CPU mode is used, while if BLAS is available, it uses BLAS.
	*/

	int AILIA_API ailiaCreateClassifier(struct AILIAClassifier **classifier, struct AILIANetwork *net, unsigned int format, unsigned int channel, unsigned int range);

	/**
	*  Destroys the classifier instance.
	*    Arguments:
	*      classifier - A classifier instance pointer
	*/
	void AILIA_API ailiaDestroyClassifier(struct AILIAClassifier *classifier);

	/**
	*  Performs object classification.
	*    Arguments:
	*      classifier                  - A classifier instance pointer
	*      src                         - Image data (32 bpp)
	*      src_stride                  - The number of bytes in 1 line
	*      src_width                   - Image width
	*      src_height                  - Image height
	*      src_format                  - AILIA_IMAGE_FORMAT_*
	*      max_class_count             - The maximum number of classification results
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaClassifierCompute(struct AILIAClassifier *classifier,const void *src, unsigned int src_stride, unsigned int src_width, unsigned int src_height, unsigned int src_format, unsigned int max_class_count);

	/**
	*  Gets the number of classification results.
	*    Arguments:
	*      classifier - A classifier instance pointer
	*      cls_count  - The number of classes
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaClassifierGetClassCount(struct AILIAClassifier *classifier, unsigned int *cls_count);

	/**
	*  Gets the classification results.
	*    Arguments:
	*      classifier - A classifier instance pointer
	*      cls        - Class information
	*      cls_idx    - Class index
	*      version    - AILIA_CLASSIFIER_CLASS_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*      If ailiaPredict is not run at all, the function returns AILIA_STATUS_INVALID_STATE.
	*      The classification results are sorted in the order of estimated probability.
	*/
	int AILIA_API ailiaClassifierGetClass(struct AILIAClassifier *classifier, AILIAClassifierClass* obj, unsigned int cls_idx, unsigned int version);

#ifdef __cplusplus
}
#endif
#endif /* !defined(INCLUDED_AILIA_CLASSIFIER) */
