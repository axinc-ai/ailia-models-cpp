/*******************************************************************
*
*    DESCRIPTION:
*      AILIA object detection library
*    AUTHOR:
*      AXELL Corporation
*    DATE:February 20, 2020
*
*******************************************************************/

#if       !defined(INCLUDED_AILIA_DETECTOR)
#define            INCLUDED_AILIA_DETECTOR

/* Core libraries */

#include "ailia.h"
#include "ailia_format.h"

/* Calling conventions */

#ifdef __cplusplus
extern "C" {
#endif

	/****************************************************************
	* Detector instance
	**/

	struct AILIADetector;

	/****************************************************************
	* Object information
	**/
	#define AILIA_DETECTOR_OBJECT_VERSION (1)

	typedef struct _AILIADetectorObject {
		unsigned int category;	// Object category number (0 to category_count-1)
		float prob;		// Estimated probability (0 to 1)
		float x;		// X position at the top left (1 for the image width)
		float y;		// Y position at the top left (1 for the image height)
		float w;		// Width (1 for the width of the image, negative numbers not allowed)
		float h;		// Height (1 for the height of the image, negative numbers not allowed)
	}AILIADetectorObject;

	#define AILIA_DETECTOR_ALGORITHM_YOLOV1 (0)	//YOLOV1
	#define AILIA_DETECTOR_ALGORITHM_YOLOV2 (1)	//YOLOV2
	#define AILIA_DETECTOR_ALGORITHM_YOLOV3 (2)	//YOLOV3

	#define AILIA_DETECTOR_ALGORITHM_SSD    (8)	//SSD(Single Shot multibox Detector)

	#define AILIA_DETECTOR_FLAG_NORMAL      (0)	//No options

	/****************************************************************
	* Object detection API
	**/

	/**
	*  Creates a detector instance.
	*    Arguments:
	*      detector       - A detector instance pointer
	*      net            - The network instance pointer
	*      format         - The network image format (AILIA_NETWORK_IMAGE_FORMAT_*)
	*      channel        - The network image channel (AILIA_NETWORK_IMAGE_CHANNEL_*)
	*      range          - The network image range (AILIA_NETWORK_IMAGE_RANGE_*)
	*      algorithm      - AILIA_DETECTOR_ALGORITHM_*
	*      caregory_count - The number of detection categories (specify 20 for VOC or 80 for COCO, etc.)
	*      flags          - AILIA_DETECTOR_FLAG_*
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaCreateDetector(struct AILIADetector **detector,struct AILIANetwork *net, unsigned int format, unsigned int channel, unsigned int range, unsigned int algorithm, unsigned int category_count, unsigned int flags);

	/**
	*  Destroys the detector instance.
	*    Arguments:
	*      detector - A detector instance pointer
	*/
	void AILIA_API ailiaDestroyDetector(struct AILIADetector *detector);

	/**
	*  Performs object detection.
	*    Arguments:
	*      detector                    - A detector instance pointer
	*      src                         - Image data (32 bpp)
	*      src_stride                  - The number of bytes in 1 line
	*      src_width                   - Image width
	*      src_height                  - Image height
	*      src_format                  - AILIA_IMAGE_FORMAT_*
	*      threshold                   - The detection threshold (for example, 0.1f) (The smaller it is, the easier the detection will be and the more detected objects found.)
	*      iou                         - Iou threshold (for example, 0.45f) (The smaller it is, the fewer detected objects found, as duplication is not allowed.)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaDetectorCompute(struct AILIADetector *detector, const void *src, unsigned int src_stride, unsigned int src_width, unsigned int src_height, unsigned int src_format, float threshold, float iou);

	/**
	*  Gets the number of detection results.
	*    Arguments:
	*      detector	  - A detector instance pointer
	*      obj_count  - The number of objects
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaDetectorGetObjectCount(struct AILIADetector *detector, unsigned int *obj_count);

	/**
	*  Gets the detection results.
	*    Arguments:
	*      detector	  - A detector instance pointer
	*      obj        - Object information
	*      obj_idx    - Object index
	*      version    - AILIA_DETECTOR_OBJECT_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*      If ailiaPredict is not run at all, the function returns AILIA_STATUS_INVALID_STATE.
	*      The detection results are sorted in the order of estimated probability.
	*/
	int AILIA_API ailiaDetectorGetObject(struct AILIADetector *detector, AILIADetectorObject* obj, unsigned int obj_idx, unsigned int version);

	/**
	*  Sets the anchor information (anchors or biases) for YoloV2 or other systems.
	*    Arguments:
	*      detector	      - A detector instance pointer
	*      anchors        - The anchor dimensions (the shape, height and width of the detection box)
	*      anchors_count  - The number of anchors (half of the anchors array size)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*       YoloV2 and other systems perform object detection with multiple detection boxes determined during training. By using this API function to set the shape of the detection box determined during training, correct inferences can be made.
	*      The {x, y, x, y ...} format is used for anchor storage.
	*      If anchors_count has a value of 5, then anchors is a 10-dimensional array.
	*/
	int AILIA_API ailiaDetectorSetAnchors(struct AILIADetector *detector, float *anchors, unsigned int anchors_count);

	/**
	*  Sets the size of the input image for YoloV3 model.
	*    Arguments:
	*      detector	      - A detector instance pointer
	*      input_width    - width of the model's input image
	*      input_height   - height of the model's input image
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*      The same YoloV3 model can be used for any input image size that is a multiple of 32.
	*      You can use this API if you want to choose the input image size, for example to reduce the calculation complexity.
	*      It must be called between ailiaCreateDetector() and ailiaDetectorCompute().
	*      If this API is not used, a default size of 416x416 is assumed.
	*      If used with some model other than YoloV3, it will return the error status AILIA_STATUS_INVALID_STATE.
	*/
	int AILIA_API ailiaDetectorSetInputShape(struct AILIADetector *detector, unsigned int input_width, unsigned int input_height);

#ifdef __cplusplus
}
#endif
#endif /* !defined(INCLUDED_AILIA_DETECTOR) */
