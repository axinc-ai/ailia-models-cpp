/*******************************************************************
*
*    DESCRIPTION:
*      AILIA user-defined callback
*    AUTHOR:
*      AXELL Corporation
*    DATE:June 24, 2019
*
*******************************************************************/

#if       !defined(INCLUDED_AILIA_CALL)
#define            INCLUDED_AILIA_CALL

/* Calling conventions */

#if defined(_WIN64) || defined(_M_X64) || defined(__amd64__) || defined(__x86_64__) || defined(__APPLE__) || defined(__ANDROID__) || defined(ANDROID) || defined(__linux__)
#define AILIA_USER_API
#else
#define AILIA_USER_API __stdcall
#endif

/* Return values of the user-defined callback */
#define AILIA_USER_API_SUCCESS ( 0)  /* Successful */
#define AILIA_USER_API_FAILED  (-1)  /* Failed */

/**
* The file access callback function is called by a single thread
* per network instance.
* If a common callback function is given to multiple instances,
* the callback function must be thread-safe.
* Also, do not throw an exception from the callback function,
* instead use AILIA_USER_API_FAILED to make error notifications.
*/


#ifdef __cplusplus
extern "C" {
#endif

	/**
	*  Opens a file.
	*  Arguments:
	*    const void *  - fopen_args given to ailiaOpenStreamEx or ailiaOpenWeightEx
	*  Return value:
	*    This function returns a user-defined file pointer if successful.
	*    It returns NULL if it fails.
	*/
	typedef void* (AILIA_USER_API *AILIA_USER_API_FOPEN) (const void *);

	/**
	*  It seeks the file specified.
	*  Arguments:
	*    void *        - A user-defined file pointer
	*    const char *  - File name
	*    long long     - Offset in bytes from the beginning of the file
	*  Return value:
	*    This function returns AILIA_USER_API_SUCCESS if successful.
	*    It returns AILIA_USER_API_FAILED if it fails.
	*/
	typedef int (AILIA_USER_API *AILIA_USER_API_FSEEK) (void*, long long);

	/**
	*  Gets the current position in the file.
	*  Arguments:
	*    void * - A user-defined file pointer
	*  Return value:
	*    This function returns the position, in bytes, the file pointer points to if successful.
	*    It returns -1 if it fails.
	*/
	typedef long long (AILIA_USER_API *AILIA_USER_API_FTELL) (void*);

	/**
	*  Gets the size of the file.
	*  Arguments:
	*    void * - A user-defined file pointer
	*  Return value:
	*    This function returns the size of the file in bytes if successful.
	*    It returns -1 if it fails.
	*/
	typedef long long (AILIA_USER_API *AILIA_USER_API_FSIZE) (void*);
 
	/**
	*  Reads data from the file.
	*  Arguments:
	*    void *                 - A pointer to the storage location of the data to be read
	*    long long              - The length in bytes of the data to be read
	*    void *                 - A user-defined file pointer
	*  Return value:
	*    This function returns AILIA_USER_API_SUCCESS if successful.
	*    It returns AILIA_USER_API_FAILED if it fails.
	*    Note that unlike the standard API, the return value will be AILIA_USER_API_*.
	*/
	typedef int (AILIA_USER_API *AILIA_USER_API_FREAD) (void *, long long, void*);

	/**
	*  Closes the file.
	*  Arguments:
	*    void *  - A user-defined file pointer
	*  Return value:
	*    This function returns AILIA_USER_API_SUCCESS if successful.
	*    It returns AILIA_USER_API_FAILED if it fails.
	*/
	typedef int (AILIA_USER_API *AILIA_USER_API_FCLOSE) (void*);

#define AILIA_FILE_CALLBACK_VERSION (1) /* Struct version */

	/* Struct for the file access callback function */
	typedef struct _ailiaFileCallback {
		AILIA_USER_API_FOPEN  fopen;     /* User-defined fopen function */
		AILIA_USER_API_FSEEK  fseek;     /* User-defined fseek function */
		AILIA_USER_API_FTELL  ftell;     /* User-defined ftell function */
		AILIA_USER_API_FREAD  fread;     /* User-defined fread function */
		AILIA_USER_API_FSIZE  fsize;     /* User-defined fsize function */
		AILIA_USER_API_FCLOSE fclose;    /* User-defined fclose function */
	} ailiaFileCallback;



#ifdef __cplusplus
}
#endif
#endif /* !defined(INCLUDED_AILIA_CALL) */
