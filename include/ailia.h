/*******************************************************************
*
*    DESCRIPTION:
*      AILIA inference library
*    AUTHOR:
*      AXELL Corporation
*    DATE:June 4, 2020
*
*******************************************************************/

#if       !defined(INCLUDED_AILIA)
#define            INCLUDED_AILIA

/* Calling conventions */

#if defined(_WIN64) || defined(_M_X64) || defined(__amd64__) || defined(__x86_64__) || defined(__APPLE__) || defined(__ANDROID__) || defined(ANDROID) || defined(__linux__)
#define AILIA_API
#else
#define AILIA_API __stdcall
#endif
#include "ailia_call.h"
#ifdef __cplusplus
extern "C" {
#endif
	/****************************************************************
	* Library status definitions
	**/

	#define AILIA_STATUS_SUCCESS							(   0)  /* Successful */
	#define AILIA_STATUS_INVALID_ARGUMENT					(  -1)  /* Incorrect argument */
	#define AILIA_STATUS_ERROR_FILE_API						(  -2)  /* File access failed. */
	#define AILIA_STATUS_INVALID_VERSION					(  -3)  /* Incorrect struct version */
	#define AILIA_STATUS_BROKEN								(  -4)  /* A corrupt file was passed. */
	#define AILIA_STATUS_MEMORY_INSUFFICIENT				(  -5)  /* Insufficient memory */
	#define AILIA_STATUS_THREAD_ERROR						(  -6)  /* Thread creation failed. */
	#define AILIA_STATUS_INVALID_STATE						(  -7)  /* The internal status of the decoder is incorrect. */
	#define AILIA_STATUS_UNSUPPORT_NET						(  -9)  /* Unsupported network */
	#define AILIA_STATUS_INVALID_LAYER						( -10)  /* Incorrect layer weight, parameter, or input or output shape */
	#define AILIA_STATUS_INVALID_PARAMINFO					( -11)  /* The content of the parameter file is invalid. */
	#define AILIA_STATUS_NOT_FOUND							( -12)  /* The specified element was not found. */
	#define AILIA_STATUS_GPU_UNSUPPORT_LAYER				( -13)  /* A layer parameter not supported by the GPU was given. */
	#define AILIA_STATUS_GPU_ERROR							( -14)  /* Error during processing on the GPU */
	#define AILIA_STATUS_UNIMPLEMENTED						( -15)  /* Unimplemented error */
	#define AILIA_STATUS_PERMISSION_DENIED					( -16)  /* Operation not allowed */
	#define AILIA_STATUS_EXPIRED                            ( -17)  /* Model Expired */
	#define AILIA_STATUS_UNSETTLED_SHAPE 					( -18)  /* The shape is not yet determined */
	#define AILIA_STATUS_DATA_REMOVED 						( -19)  /* Deleted due to optimization */
	#define AILIA_STATUS_LICENSE_NOT_FOUND                  ( -20)  /* No valid license found*/
	#define AILIA_STATUS_LICENSE_BROKEN                     ( -21)  /* License is broken */
	#define AILIA_STATUS_LICENSE_EXPIRED                    ( -22)  /* License expired */
	#define AILIA_STATUS_OTHER_ERROR						(-128)  /* Unknown error */

	/****************************************************************
	* Network instance
	**/

	struct AILIANetwork;

	/****************************************************************
	* Shape information
	**/

	#define AILIA_SHAPE_VERSION (1)

	typedef struct _AILIAShape {
		unsigned int x;			// Size along the X axis
		unsigned int y;			// Size along the Y axis
		unsigned int z;			// Size along the Z axis
		unsigned int w;			// Size along the W axis
		unsigned int dim;		// Dimension information
	}AILIAShape;

	/****************************************************************
	* Number of threads
	**/

	#define  AILIA_MULTITHREAD_AUTO (0)

	/****************************************************************
	* Automatic setup of the inference backend
	**/

	#define  AILIA_ENVIRONMENT_ID_AUTO		(-1)

	/****************************************************************
	* Inference API
	**/

	/**
	*  Creates a network instance.
	*    Arguments:
	*      net        - A pointer to the network instance pointer
	*      env_id     - The ID of the inference backend used for computation (obtained by ailiaGetEnvironment). It is selected automatically if AILIA_ENVIRONMENT_ID_AUTO is specified.
	*      num_thread - The upper limit on the number of threads (It is set automatically if AILIA_MULTITHREAD_AUTO is specified.)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*   Description:
	*     Creates a network instance.
	*     If the inference backend is set to automatic, CPU mode is used, while if BLAS is available, it uses BLAS.
	*     Note that if BLAS is used, num_thread may be ignored.
	*/
	int AILIA_API ailiaCreate(struct AILIANetwork **net, int env_id, int num_thread);

	/**
	*   Initializes the network instance. (Read from file)
	*    Arguments:
	*      net		 - A network instance pointer
	*      path      - The path name to the prototxt file (MBSC or UTF16)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*   Description:
	*      This function reads the network instance from a file and initializes it.
	*/
	int AILIA_API ailiaOpenStreamFileA(struct AILIANetwork *net, const char    *path);
	int AILIA_API ailiaOpenStreamFileW(struct AILIANetwork *net, const wchar_t *path);

	/**
	*   Initializes the network instance. (User-defined file access callback)
	*    Arguments:
	*      net			 - A network instance pointer
	*      fopen_args	 - An argument pointer supplied by AILIA_USER_API_FOPEN
	*      callback		 - A struct for the user-defined file access callback function
	*      version		 - The version of the struct for the file access callback function (AILIA_FILE_CALLBACK_VERSION)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*   Description:
	*      This function reads the network instance from a file and initializes it.
	*/
	int AILIA_API ailiaOpenStreamEx(struct AILIANetwork *net, const void *fopen_args, ailiaFileCallback callback, int version);

	/**
	*   Initializes the network instance. (Read from memory)
	*    Arguments:
	*      net			 - A network instance pointer
	*      buf      	 - A pointer to the data in the prototxt file
	*      buf_size		 - The data size of the prototxt file
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*   Description:
	*      This function reads the network instance from memory and initializes it.
	*/
	int AILIA_API ailiaOpenStreamMem(struct AILIANetwork *net, const void *buf, unsigned int buf_size);

	/**
	*   Reads weights into a network instance. (Read from file)
	*    Arguments:
	*      net		 - A network instance pointer
	*      path      - The path name to the protobuf file (MBSC or UTF16)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*   Description:
	*      This function reads weights into the network instance from a file.
	*/
	int AILIA_API ailiaOpenWeightFileA(struct AILIANetwork *net, const char    *path);
	int AILIA_API ailiaOpenWeightFileW(struct AILIANetwork *net, const wchar_t *path);

	/**
	*   Reads weights into a network instance. (User-defined file access callback)
	*    Arguments:
	*      net			 - A network instance pointer
	*      fopen_args	 - An argument pointer supplied by AILIA_USER_API_FOPEN
	*      callback		 - A struct for the user-defined file access callback function
	*      version		 - The version of the struct for the file access callback function (AILIA_FILE_CALLBACK_VERSION)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*   Description:
	*      This function reads weights into the network instance from a file.
	*/
	int AILIA_API ailiaOpenWeightEx(struct AILIANetwork *net, const void *fopen_args, ailiaFileCallback callback, int version);

	/**
	*   Reads weights into a network instance. (Read from memory)
	*    Arguments:
	*      net		     - A network instance pointer
	*      buf      	 - A pointer to the data in the protobuf file
	*      buf_size		 - The data size of the protobuf file
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*   Description:
	*      This function reads weights into the network instance from memory.
	*/
	int AILIA_API ailiaOpenWeightMem(struct AILIANetwork *net, const void *buf, unsigned int buf_size);

	/**
	*  It destroys the network instance.
	*    Arguments:
	*      net - A network instance pointer
	*/
	void AILIA_API ailiaDestroy(struct AILIANetwork *net);

	/**
	*  Changes the shape of the input data during inference.
	*    Arguments:
	*      net		 - A network instance pointer
	*      shape     - Shape information for the input data
	*      version   - AILIA_SHAPE_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*   Description:
	*      This function changes the input shape defined in prototxt.
	*      The shape must have the same rank as the one contained in prototxt.
	*      Note that an error may be returned if the weights are dependent on the input shapes, among other reasons.
	*/
	int AILIA_API ailiaSetInputShape(struct AILIANetwork *net, const AILIAShape* shape, unsigned int version);

	/**
	*  Gets the shape of the input data during inference.
	*    Arguments:
	*      net		 - A network instance pointer
	*      shape     - Shape information for the input data
	*      version   - AILIA_SHAPE_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetInputShape(struct AILIANetwork *net, AILIAShape* shape, unsigned int version);

	/**
	*  Gets the shape of the output data during inference.
	*    Arguments:
	*      net		 - A network instance pointer
	*      shape     - Shape information of the output data
	*      version   - AILIA_SHAPE_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetOutputShape(struct AILIANetwork *net, AILIAShape* shape, unsigned int version);

	/**
	*  Performs the inferences and provides the inference result.
	*    Arguments:
	*      net                         - A network instance pointer
	*      dest                        - The result is stored in the inference result destination buffer as numeric type data in the order of X, Y, Z, and W. The buffer has the same size as the network file outputSize.
	*      dest_size                   - The number of bytes for the destination buffer for the inference result
	*      src                         - The input is stored as numeric type data in the order of the inference data X, Y, Z, and W. The input has the same size as the network file inputSize.
	*      src_size                    - The size of the inference data
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaPredict(struct AILIANetwork *net, void * dest, unsigned int dest_size, const void *src, unsigned int src_size);

	/****************************************************************
	* API to get the status
	**/

	/**
	*  Gets the amount of internal data (blob) during inference.
	*    Arguments:
	*      net		  - A network instance pointer
	*      blob_count - Storage location of the number of blobs
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetBlobCount(struct AILIANetwork *net, unsigned int *blob_count);

	/**
	*  Gets the shape of the internal data (blob) during inference.
	*    Arguments:
	*      net		  - A network instance pointer
	*      shape     - Storage location of the data shape information
	*      blob_idx   - The index of the blob (0 to ailiaGetBlobCount()-1)
	*      version    - AILIA_SHAPE_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetBlobShape(struct AILIANetwork *net, AILIAShape* shape, unsigned int blob_idx, unsigned int version);

	/**
	*  Gets the internal data (blob) during inference.
	*    Arguments:
	*      net		  - A network instance pointer
	*      dest                        - The result is stored in the inference result destination buffer as numeric type data in the order of X, Y, Z, and W.
	*      dest_size  - The number of bytes for the inference result destination buffer
	*      blob_idx   - The index of the blob (0 to ailiaGetBlobCount()-1)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*      If ailiaPredict is not run at all, the function returns AILIA_STATUS_INVALID_STATE.
	*/
	int AILIA_API ailiaGetBlobData(struct AILIANetwork *net, void* dest, unsigned int dest_size, unsigned int blob_idx);

	/**
	*  Searches by name for the index of the internal data (blob) during inference and returns it.
	*    Arguments:
	*      net		  - A network instance pointer
	*      blob_idx   - The index of the blob (0 to ailiaGetBlobCount()-1)
	*      name       - The name of the blob to search for
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaFindBlobIndexByName(struct AILIANetwork *net, unsigned int* blob_idx, const char* name);

	/**
	*  Gets the size of the buffer needed for output of the name of the internal data (blob).
	*    Arguments:
	*      net		        - A network instance pointer
	*      blob_idx         - The index of the blob (0 to ailiaGetBlobCount()-1)
	*      buffer_size      - The size of the buffer needed for output of the blob name (including the null terminator)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetBlobNameLengthByIndex(struct AILIANetwork *net, const unsigned int blob_idx, unsigned int* buffer_size);

	/**
	*  Searches by index for the name of the internal data (blob) during inference and returns it.
	*    Arguments:
	*      net		    - A network instance pointer
	*      buffer       - The output destination buffer for the blob name
	*      buffer_size      - The size of the buffer (including the null terminator)
	*      blob_idx     - The index of the blob to search for
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaFindBlobNameByIndex(struct AILIANetwork *net, char* buffer, const unsigned int buffer_size, const unsigned int blob_idx);

	/**
	*  Gets the size of the buffer needed for the network summary.
	*    Arguments:
	*      net		    - A network instance pointer
	*      buffer_size      - The storage location of the buffer size (including the null terminator)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetSummaryLength(struct AILIANetwork * net, unsigned int * buffer_size);

	/**
	*  Shows the name and shape of each blob.
	*    Arguments:
	*      net		    - A network instance pointer
	*      buffer       - The output destination of the summary
	*      buffer_size  - The size of the output buffer (including the null terminator)
	*                     Set the value obtained by ailiaGetSummaryLength.
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/

	int AILIA_API ailiaSummary(struct AILIANetwork * net, char* const buffer, const unsigned int buffer_size);

	/****************************************************************
	* API to specify multiple inputs and make inferences
	**/

	/**
	*  Get the number of input data blobs.
	*    Arguments:
	*      net		    - A network instance pointer
	*      input_blob_count - Storage location of the number of input blobs
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetInputBlobCount(struct AILIANetwork *net, unsigned int *input_blob_count);

	/**
	*  Get the blob index of the input data.
	*    Arguments:
	*      net		    - A network instance pointer
	*      blob_idx - index of the blob (between 0 and ailiaGetBlobCount()-1)
	*      input_blob_idx - index among the input blobs (between 0 and ailiaGetInputBlobCount()-1)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetBlobIndexByInputIndex(struct AILIANetwork *net, unsigned int *blob_idx, unsigned int input_blob_idx);

	/**
	*  Provides the specified blob with the input data.
	*    Arguments:
	*      net		    - A network instance pointer
	*      src          - The inference data is stored as numeric type data in the order of X, Y, Z, and W.
	*      src_size     - The size of the inference data
	*      blob_idx     - The index of the blob for input
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*      This function is used to specify the input on networks with multiple inputs.
	*      If something other than a blob in the input layer is specified for blob_idx, the function returns AILIA_STATUS_INVALID_ARGUMENT.
	*/
	int AILIA_API ailiaSetInputBlobData(struct AILIANetwork *net, const void* src, unsigned int src_size, unsigned int blob_idx);

	/**
    *    Change the shape of the blob given by its index
    *    Arguments:
    *      net         - network object pointer
    *      shape    - new shape of the blob
    *      blob_idx - index referencing the blob to reshape
    *      version  - AILIA_SHAPE_VERSION
    *    Return value:
    *      In case of success, AILIA_STATUS_SUCCESS, and otherwise the coresponding error code.
    *    Description:
    *      This is useful to change the network input shape in a context where there are several input blobs.
    *      If blob_idx does not correspond to an input layer, AILIA_STATUS_INVALID_ARGUMENT is returned.
    *      For other related remarks, see the documentation of ailiaSetInputShape().
    */
	int AILIA_API ailiaSetInputBlobShape(struct AILIANetwork *net, const AILIAShape* shape, unsigned int blob_idx, unsigned int version);

	/**
	*  Makes inferences with the input data specified in advance.
	*    Arguments:
	*      net		    - A network instance pointer
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*      This function is used when, for example, the input is provided with ailiaSetInputBlobData.
	*      Get the inference result with ailiaGetBlobData.
	*/
	int AILIA_API ailiaUpdate(struct AILIANetwork *net);

	/**
	*  Get the number of output data blobs.
	*    Arguments:
	*      net		    - A network instance pointer
	*      output_blob_count - Storage location for the number of output blobs.
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetOutputBlobCount(struct AILIANetwork *net, unsigned int *output_blob_count);

	/**
	*  Get the blob index of the input data blob.
	*    Arguments:
	*      net		    - A network instance pointer
	*      blob_idx - blob index (between 0 and ailiaGetBlobCount()-1)
	*      output_blob_idx - index among output blobs (between 0 and ailiaGetOutputBlobCount()-1)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetBlobIndexByOutputIndex(struct AILIANetwork *net, unsigned int *blob_idx, unsigned int output_blob_idx);

	/****************************************************************
	* API to specify and obtain an inference backend
	**/

	#define  AILIA_ENVIRONMENT_VERSION      (2)

	#define  AILIA_ENVIRONMENT_TYPE_CPU	    (0)
	#define  AILIA_ENVIRONMENT_TYPE_BLAS    (1)
	#define  AILIA_ENVIRONMENT_TYPE_GPU     (2)

	#define AILIA_ENVIRONMENT_BACKEND_NONE			(0)
	#define AILIA_ENVIRONMENT_BACKEND_AMP			(1)
	#define AILIA_ENVIRONMENT_BACKEND_CUDA			(2)
	#define AILIA_ENVIRONMENT_BACKEND_MPS			(3)
	#define AILIA_ENVIRONMENT_BACKEND_RENDERSCRIPT	(4)
	#define AILIA_ENVIRONMENT_BACKEND_OPENCL		(5)
	#define AILIA_ENVIRONMENT_BACKEND_VULKAN		(6)

	#define AILIA_ENVIRONMENT_PROPERTY_NORMAL			(0)
    #define AILIA_ENVIRONMENT_PROPERTY_LOWPOWER         (1)    //  Indicates that a low-power GPU (e.g. integrated GPU) will be preferentially used. (for MPS)
	#define AILIA_ENVIRONMENT_PROPERTY_FP16				(2)    //  Indicates that a FP16 mode

	typedef struct _AILIAEnvironment {
		int id;				// The ID to identify the inference backend (passed to ailiaCreate as an argument)
		int type;			// The type of the inference backend (AILIA_ENVIRONMENT_TYPE_CPU, BLAS, or GPU)
		const char* name;	// The device name. It is valid until the AILIANetwork instance is destroyed.
        int backend;        // Computational (hardware) backend enabled by this environment (AILIA_ENVIRONMENT_BACKEND_*)
        int props;          // Additional property (low-power etc) of the environment (AILIA_ENVIRONMENT_PROPERTY_*)
	}AILIAEnvironment;

	/**
	*  Specifies a temporary cache directory.
	*    Arguments:
	*      cache_dir    - Temporary cache directory
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*      This system uses the specified cache directory to generate and store machine code optimized for each inference backend.
	*      Call only once at the start of execution of ailia. It ignores any second and subsequent calls, and automatically returns success.
	*      There is no particular problem if it is called from multiple threads, as it provides exclusive control internally.
	*      Some functions, such as RenderScript in an Android environment, cannot be used until this API function is called.
	*      Specify the file path obtained by Context.getCacheDir() for cache_dir.
	*      You cannot specify the path to external storage due to Permission restrictions from RenderScript.
	*/
	int AILIA_API ailiaSetTemporaryCachePathA(const char    * cache_dir);
	int AILIA_API ailiaSetTemporaryCachePathW(const wchar_t * cache_dir);

	/**
	*  Gets the number of available computational environments (CPU, GPU).
	*    Arguments:
	*      env_count     - The storage location of the computational environment information
	*    Return value:
	*      The number of available computational environments
	*/
	int AILIA_API ailiaGetEnvironmentCount(unsigned int * env_count);

	/**
	*  Gets the list of computational environments.
	*    Arguments:
	*      env          - The storage location of the computational environment information (valid until the AILIANetwork instance is destroyed)
	*      env_idx    - The index of the computational environment information (0 to ailiaGetEnvironmentCount()-1)
	*      version      - AILIA_ENVIRONMENT_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/

	int AILIA_API ailiaGetEnvironment(AILIAEnvironment** env, unsigned int env_idx, unsigned int version);

	/**
	*  Gets the selected computational environment.
	*    Arguments:
	*      net        - A network instance pointer
	*      env          - The storage location of the computational environment information (valid until the AILIANetwork instance is destroyed)
	*      version      - AILIA_ENVIRONMENT_VERSION
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*/
	int AILIA_API ailiaGetSelectedEnvironment(struct AILIANetwork *net, AILIAEnvironment** env, unsigned int version);

	/****************************************************************
	* API to specify memory mode
	**/

	#define AILIA_MEMORY_NO_OPTIMIZATION                        (0) // Do not release the intermediate buffer
	#define AILIA_MEMORY_REDUCE_CONSTANT                        (1) // Releases the intermediate buffer that is a constant such as weight
	#define AILIA_MEMORY_REDUCE_CONSTANT_WITH_INPUT_INITIALIZER (2) // Disable the input specified initializer and release the intermediate buffer that becomes a constant such as weight.
	#define AILIA_MEMORY_REDUCE_INTERSTAGE                      (4) // Release intermediate buffer during inference
	#define AILIA_MEMORY_REUSE_INTERSTAGE                       (8) // Infer by sharing the intermediate buffer. When used with AILIA_MEMORY_REDUCE_INTERSTAGE, the sharable intermediate buffer is not opened.

	#define AILIA_MEMORY_OPTIMAIZE_DEFAULT (AILIA_MEMORY_REDUCE_CONSTANT)

	/**
	*  Set the memory usage policy for inference
	*    Arguments:
	*      net        - A network instance pointer
	*      mode       - Memory mode (Multiple specifications possible with logical sum) AILIA_MEMORY_XXX (Default :AILIA_MEMORY_REDUCE_CONSTANT)
	*    Return value:
	*      If this function is successful, it returns AILIA_STATUS_SUCCESS, or an error code otherwise.
	*    Description:
	*      Change the memory usage policy.
	*      If a value other than AILIA_MEMORY_NO_OPTIMIZATION is specified,
	*      the intermediate buffer secured during inference will be released, so the memory usage during inference can be reduced.
	*      Must be specified immediately after ailiaCreate. It cannot be changed after calling ailiaOpen.
	*      If you specify to release the intermediate buffer, calling ailiaGetBlobData for the corresponding blob will return an AILIA_STATUS_DATA_REMOVED error.
	*/

	int AILIA_API ailiaSetMemoryMode(struct AILIANetwork* net, unsigned int mode);

	/****************************************************************
	* API to get information
	**/

	/**
	* Returns the details of errors.
	* Return value:
	*   Error details
	* Description:
	*   The return value does not have to be released.
	*   The string is valid until the next ailia API function is called.
	*/
	const char * AILIA_API ailiaGetErrorDetail(struct AILIANetwork *net);

	/**
	* Get the version of the library.
	* Return value:
	*   Version number
	* Description:
	*   The return value does not have to be released.
	*/
	const char* AILIA_API ailiaGetVersion(void);

#ifdef UNICODE
#define ailiaOpenStreamFile					ailiaOpenStreamFileW
#define ailiaOpenWeightFile					ailiaOpenWeightFileW
#define ailiaSetTemporaryCachePath          ailiaSetTemporaryCachePathW
#else
#define ailiaOpenStreamFile					ailiaOpenStreamFileA
#define ailiaOpenWeightFile					ailiaOpenWeightFileA
#define ailiaSetTemporaryCachePath          ailiaSetTemporaryCachePathA
#endif

#ifdef __cplusplus
}
#endif
#endif /* !defined(INCLUDED_AILIA) */
