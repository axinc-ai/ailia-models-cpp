/*******************************************************************
*
*    DESCRIPTION:
*      AILIA library for human pose estimation and human face landmarks extraction
*    AUTHOR:
*      AXELL Corporation
*    DATE:June 22, 2020
*
*******************************************************************/

#if       !defined(INCLUDED_AILIA_POSE_ESTIMATOR)
#define            INCLUDED_AILIA_POSE_ESTIMATOR

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

	struct AILIAPoseEstimator;

	/****************************************************************
	* Object information
	**/

	#define AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_POSE   ( 0) // Human pose estimation
	#define AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_FACE   ( 1) // Human face landmarks extraction
	#define AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_UPPOSE ( 2) // Human upper body pose estimation
	#define AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_UPPOSE_FPGA ( 3) // Human upper body pose estimation(FPGA)
	#define AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_HAND  ( 5) // Human hand estimation
	#define AILIA_POSE_ESTIMATOR_ALGORITHM_OPEN_POSE     (10) // Human pose estimation
	#define AILIA_POSE_ESTIMATOR_ALGORITHM_LW_HUMAN_POSE (11) // Human pose estimation

	/* Definition of body joint positions for human pose estimation */
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_NOSE				(0)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_EYE_LEFT			(1)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_EYE_RIGHT		(2)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_EAR_LEFT			(3)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_EAR_RIGHT		(4)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_SHOULDER_LEFT	(5)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_SHOULDER_RIGHT	(6)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_ELBOW_LEFT		(7)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_ELBOW_RIGHT		(8)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_WRIST_LEFT		(9)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_WRIST_RIGHT		(10)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_LEFT			(11)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_RIGHT		(12)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_KNEE_LEFT		(13)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_KNEE_RIGHT		(14)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_ANKLE_LEFT		(15)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_ANKLE_RIGHT		(16)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_SHOULDER_CENTER	(17)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_BODY_CENTER		(18)
	#define AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_CNT		(19)	// Count

	/* Definition for human face landmarks extraction */
	#define AILIA_POSE_ESTIMATOR_FACE_KEYPOINT_CNT		(68)	// Count

	/* Definition of body joint positions for human upper body pose estimation */
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_NOSE				(0)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_EYE_LEFT			(1)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_EYE_RIGHT			(2)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_EAR_LEFT			(3)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_EAR_RIGHT			(4)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_SHOULDER_LEFT		(5)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_SHOULDER_RIGHT		(6)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_ELBOW_LEFT			(7)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_ELBOW_RIGHT		(8)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_WRIST_LEFT			(9)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_WRIST_RIGHT		(10)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_HIP_LEFT			(11)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_HIP_RIGHT			(12)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_SHOULDER_CENTER	(13)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_BODY_CENTER		(14)
	#define AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_CNT				(15)	// Count

	/* Definition of hand joint position for human hand estimation */
	#define AILIA_POSE_ESTIMATOR_HAND_KEYPOINT_CNT		(21)	// Count

	typedef struct _AILIAPoseEstimatorKeypoint {
		float x;						// Input image X coordinate (0.0, 1.0)
		float y;						// Input image Y coordinate (0.0, 1.0)
		float z_local;					//  Valid only for human pose estimation. The local Z coordinate is estimated when the center of the body is defined as coordinate 0. The unit (scale) is the same as that for X.
		float score;					//  The confidence of this point. If the value is 0.0F, then this point is not available as it has not been detected yet.
		int interpolated;				//  The default is 0. If this point has not been detected and can be interpolated by other points, the x and y values are then interpolated and the value of interpolated is set to 1.
	}AILIAPoseEstimatorKeypoint;

	#define AILIA_POSE_ESTIMATOR_OBJECT_POSE_VERSION (1)	// Version of the struct format
	typedef struct _AILIAPoseEstimatorObjectPose {
		AILIAPoseEstimatorKeypoint points[AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_CNT];	//  Detected body joint positions. The array index corresponding to a body joint number.
		float total_score;			// The confidence of this object
		int num_valid_points;		// The number of body joint positions properly detected in points[]
		int id;						//  A unique ID for this object in the time direction. An integer value of 1 or more.
		float angle[3];				//  Euler angles for this object: yaw, pitch, and roll (in radians). Currently, only yaw is supported. If the angles are not detected, they are set to FLT_MAX.
	}AILIAPoseEstimatorObjectPose;

	#define AILIA_POSE_ESTIMATOR_OBJECT_FACE_VERSION (1)	// Version of the struct format
	typedef struct _AILIAPoseEstimatorObjectFace {
		AILIAPoseEstimatorKeypoint points[AILIA_POSE_ESTIMATOR_FACE_KEYPOINT_CNT];	//  Detected human face landmarks. The array index corresponding to a human face landmark number.
		float total_score;			// The confidence of this object
	}AILIAPoseEstimatorObjectFace;

	#define AILIA_POSE_ESTIMATOR_OBJECT_UPPOSE_VERSION (1)	// Version of the struct format
	typedef struct _AILIAPoseEstimatorObjectUpPose {
		AILIAPoseEstimatorKeypoint points[AILIA_POSE_ESTIMATOR_UPPOSE_KEYPOINT_CNT];	// Detected body joint positions. The array index corresponding to a body joint number.
		float total_score;			// The confidence of this object
		int num_valid_points;		// The number of body joint positions properly detected in points[]
		int id;						//  A unique ID for this object in the time direction. An integer value of 1 or more.
		float angle[3];				//  Euler angles for this object: yaw, pitch, and roll (in radians). Currently, only yaw is supported. If the angles are not detected, they are set to FLT_MAX.
	}AILIAPoseEstimatorObjectUpPose;

	#define AILIA_POSE_ESTIMATOR_OBJECT_HAND_VERSION (1)	// Version of the struct format
	typedef struct _AILIAPoseEstimatorObjectHand {
		AILIAPoseEstimatorKeypoint points[AILIA_POSE_ESTIMATOR_HAND_KEYPOINT_CNT];	// Detected hand joint positions. The array index corresponding to a hand joint number.
		float total_score;			// The confidence of this object
	}AILIAPoseEstimatorObjectHand;

	/****************************************************************
	* API for human pose estimation and human face landmarks extraction
	**/

	/**
	*  Creates a detector instance.
	*    Arguments:
	*      pose_estimator - A detector instance pointer
	*      net            - The network instance pointer
	*      algorithm      - AILIA_POSE_ESTIMATOR_ALGORITHM_*
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*      This function creates a detector instance from AILIANetwork when reading caffemodel and prototxt.
	*/
	int AILIA_API ailiaCreatePoseEstimator(struct AILIAPoseEstimator **pose_estimator, struct AILIANetwork *net, unsigned int algorithm);

	/**
	*  Destroys the detector instance.
	*    Arguments:
	*      pose_estimator - A detector instance pointer
	*/
	void AILIA_API ailiaDestroyPoseEstimator(struct AILIAPoseEstimator *pose_estimator);

	/**
	*  Set the detection threshold. It is valid only for hand posture detection (AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_HAND).
	*    Arguments:
	*      pose_estimator              - A detector instance pointer
	*      threshold                   - The detection threshold (for example, 0.1f) (The smaller it is, the easier the detection will be and the more detected objects found.)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaPoseEstimatorSetThreshold(struct AILIAPoseEstimator *pose_estimator, float threshold);

	/**
	*  Performs human pose estimation and human face landmarks extraction.
	*    Arguments:
	*      pose_estimator              - A detector instance pointer
	*      src                         - Image data (32 bpp)
	*      src_stride                  - The number of bytes in 1 line
	*      src_width                   - Image width
	*      src_height                  - Image height
	*      src_format                  - AILIA_IMAGE_FORMAT_*
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaPoseEstimatorCompute(struct AILIAPoseEstimator *pose_estimator, const void *src, unsigned int src_stride, unsigned int src_width, unsigned int src_height, unsigned int src_format);

	/**
	*  Gets the number of detection results.
	*    Arguments:
	*      pose_estimator  - A detector instance pointer
	*      obj_count       - The number of objects. Set to 1 or 0 for human face landmarks.
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaPoseEstimatorGetObjectCount(struct AILIAPoseEstimator *pose_estimator, unsigned int *obj_count);

	/**
	*  Gets the results of the human pose estimation.
	*    Arguments:
	*      pose_estimator  - A detector instance pointer
	*      obj             - Object information
	*      obj_idx         - Object index
	*      version         - AILIA_POSE_ESTIMATOR_OBJECT_POSE_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaPoseEstimatorGetObjectPose(struct AILIAPoseEstimator *pose_estimator, AILIAPoseEstimatorObjectPose* obj, unsigned int obj_idx, unsigned int version);

	/**
	*  Gets the results of the human face landmarks extraction.
	*    Arguments:
	*      pose_estimator  - A detector instance pointer
	*      obj             - Object information
	*      obj_idx         - Object index. Ensure that 0 is specified.
	*      version         - AILIA_POSE_ESTIMATOR_OBJECT_FACE_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaPoseEstimatorGetObjectFace(struct AILIAPoseEstimator *pose_estimator, AILIAPoseEstimatorObjectFace* obj, unsigned int obj_idx, unsigned int version);

	/**
	*  Gets the results of the human up pose estimation.
	*    引数:
	*      pose_estimator  - A detector instance pointer
	*      obj             - Object information
	*      obj_idx         - Object index
	*      version         - AILIA_POSE_ESTIMATOR_OBJECT_UPPOSE_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaPoseEstimatorGetObjectUpPose(struct AILIAPoseEstimator *pose_estimator, AILIAPoseEstimatorObjectUpPose* obj, unsigned int obj_idx, unsigned int version);

	/**
	*  Gets the results of the human hand estimation.
	*    Arguments:
	*      pose_estimator  - A detector instance pointer
	*      obj             - Object information
	*      obj_idx         - Object index
	*      version         - AILIA_POSE_ESTIMATOR_OBJECT_HAND_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaPoseEstimatorGetObjectHand(struct AILIAPoseEstimator *pose_estimator, AILIAPoseEstimatorObjectHand* obj, unsigned int obj_idx, unsigned int version);

#ifdef __cplusplus
}
#endif
#endif /* !defined(INCLUDED_AILIA_POSE_ESTIMATOR) */
