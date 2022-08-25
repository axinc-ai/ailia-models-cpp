/*******************************************************************
*
*    DESCRIPTION:
*      AILIA mediapipe_iris sample
*    AUTHOR:
*
*    DATE:2022/08/19
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#undef UNICODE

#include "ailia.h"
#include "ailia_detector.h"
#include "utils.h"
#include "mat_utils.h"
#include "image_utils.h"
#include "detector_utils.h"
#include "webcamera_utils.h"
#include "blazeface_utils.h"


// ======================
// Parameters
// ======================

#define DETECTION_WEIGHT_PATH "blazeface.opt.onnx"
#define DETECTION_MODEL_PATH  "blazeface.opt.onnx.prototxt"
#define LANDMARK_WEIGHT_PATH  "facemesh.opt.onnx"
#define LANDMARK_MODEL_PATH   "facemesh.opt.onnx.prototxt"
#define LANDMARK2_WEIGHT_PATH "iris.opt.onnx"
#define LANDMARK2_MODEL_PATH  "iris.opt.onnx.prototxt"

#define IMAGE_PATH      "man.jpg"
#define SAVE_IMAGE_PATH "output.png"

#if defined(_WIN32) || defined(_WIN64)
#define PRINT_OUT(...) fprintf_s(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf_s(stderr, __VA_ARGS__)
#else
#define PRINT_OUT(...) fprintf(stdout, __VA_ARGS__)
#define PRINT_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

#define BENCHMARK_ITERS 5

static std::string detection_weight(DETECTION_WEIGHT_PATH);
static std::string detection_model(DETECTION_MODEL_PATH);
static std::string landmark_weight(LANDMARK_WEIGHT_PATH);
static std::string landmark_model(LANDMARK_MODEL_PATH);
static std::string landmark2_weight(LANDMARK2_WEIGHT_PATH);
static std::string landmark2_model(LANDMARK2_MODEL_PATH);

static std::string image_path(IMAGE_PATH);
static std::string video_path("0");
static std::string save_image_path(SAVE_IMAGE_PATH);

static bool benchmark  = false;
static bool video_mode = false;
static int args_env_id = -1;


// ======================
// Argument Parser
// ======================

static void print_usage()
{
    PRINT_OUT("usage: mediapipe_iris [-h] [-i IMAGE] [-v VIDEO] [-s SAVE_IMAGE_PATH] [-b] [-e ENV_ID]\n");
    return;
}


static void print_help()
{
    PRINT_OUT("\n");
    PRINT_OUT("mediapipe_iris model\n");
    PRINT_OUT("\n");
    PRINT_OUT("optional arguments:\n");
    PRINT_OUT("  -h, --help            show this help message and exit\n");
    PRINT_OUT("  -i IMAGE, --input IMAGE\n");
    PRINT_OUT("                        The input image path.\n");
    PRINT_OUT("  -v VIDEO, --video VIDEO\n");
    PRINT_OUT("                        The input video path. If the VIDEO argument is set to\n");
    PRINT_OUT("                        0, the webcam input will be used.\n");
    PRINT_OUT("  -s SAVE_IMAGE_PATH, --savepath SAVE_IMAGE_PATH\n");
    PRINT_OUT("                        Save path for the output image.\n");
    PRINT_OUT("  -b, --benchmark       Running the inference on the same input 5 times to\n");
    PRINT_OUT("                        measure execution performance. (Cannot be used in\n");
    PRINT_OUT("                        video mode)\n");
    PRINT_OUT("  -e ENV_ID, --env_id ENV_ID\n");
    PRINT_OUT("                        The backend environment id.\n");
    return;
}


static void print_error(std::string arg)
{
    PRINT_ERR("mediapipe_iris: error: unrecognized arguments: %s\n", arg.c_str());
    return;
}


static int argument_parser(int argc, char **argv)
{
    int status = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (status == 0) {
            if (arg == "-i" || arg == "--input") {
                status = 1;
            }
            else if (arg == "-v" || arg == "--video") {
                video_mode = true;
                status = 2;
            }
            else if (arg == "-s" || arg == "--savepath") {
                status = 3;
            }
            else if (arg == "-b" || arg == "--benchmark") {
                benchmark = true;
            }
            else if (arg == "-h" || arg == "--help") {
                print_usage();
                print_help();
                return -1;
            }
            else if (arg == "-e" || arg == "--env_id") {
                status = 4;
            }
            else {
                print_usage();
                print_error(arg);
                return -1;
            }
        }
        else if (arg[0] != '-') {
            switch (status) {
            case 1:
                image_path = arg;
                break;
            case 2:
                video_path = arg;
                break;
            case 3:
                save_image_path = arg;
                break;
            case 4:
                args_env_id = atoi(arg.c_str());
                break;
            default:
                print_usage();
                print_error(arg);
                return -1;
            }
            status = 0;
        }
        else {
            print_usage();
            print_error(arg);
            return -1;
        }
    }

    return AILIA_STATUS_SUCCESS;
}


static void resize_pad(cv::Mat& mat_src, cv::Mat& mat_dst, float& scale, int pad[2])
{
    int h1, w1, padh, padw;
    if (mat_src.rows >= mat_src.cols) {
        h1 = 256;
        w1 = 256 * mat_src.cols / mat_src.rows;
        padh = 0;
        padw = 256 - w1;
        scale = (float)mat_src.cols / (float)w1;
    }
    else {
        h1 = 256 * mat_src.rows / mat_src.cols;
        w1 = 256;
        padh = 256 - h1;
        padw = 0;
        scale = (float)mat_src.rows / (float)h1;
    }

    int padh1 = padh / 2;
    int padh2 = padh / 2 + padh % 2;
    int padw1 = padw / 2;
    int padw2 = padw / 2 + padw % 2;

    cv::Mat mat_rsz;
    cv::resize(mat_src, mat_rsz, cv::Size(w1, h1));

    cv::Mat mat_pad;
    mat_pad.create(padh1 + mat_rsz.rows + padh2, padw1 + mat_rsz.cols + padw2, mat_rsz.type());
    mat_pad.setTo(cv::Scalar::all(0));
    mat_rsz.copyTo(mat_pad(cv::Rect(padw1, padh1, mat_rsz.cols, mat_rsz.rows)));

    cv::resize(mat_pad, mat_dst, cv::Size(128, 128));
}


// TODO
#if 0
def denormalize_detections(detections, scale, pad):
    detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
    detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
    detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
    detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

    detections[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]
    detections[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]
    return detections


def detection2roi(detection, detection2roi_method='box'):
    if detection2roi_method == 'box':
        # compute box center and scale
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        xc = (detection[:, 1] + detection[:, 3]) / 2
        yc = (detection[:, 0] + detection[:, 2]) / 2
        scale = (detection[:, 3] - detection[:, 1])  # assumes square boxes

    elif detection2roi_method == 'alignment':
        # compute box center and scale
        # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
        xc = detection[:, 4+2*kp1]
        yc = detection[:, 4+2*kp1+1]
        x1 = detection[:, 4+2*kp2]
        y1 = detection[:, 4+2*kp2+1]
        scale = np.sqrt(((xc-x1)**2 + (yc-y1)**2)) * 2
    else:
        raise NotImplementedError(
            "detection2roi_method [%s] not supported" % detection2roi_method
        )

    yc += dy * scale
    scale *= dscale

    # compute box rotation
    x0 = detection[:, 4+2*kp1]
    y0 = detection[:, 4+2*kp1+1]
    x1 = detection[:, 4+2*kp2]
    y1 = detection[:, 4+2*kp2+1]
    theta = np.arctan2(y0-y1, x0-x1) - theta0
    return xc, yc, scale, theta


def extract_roi(frame, xc, yc, theta, scale):
    # take points on unit square and transform them according to the roi
    points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]]).reshape(1, 2, 4)
    points = points * scale.reshape(-1, 1, 1)/2
    theta = theta.reshape(-1, 1, 1)
    R = np.concatenate((
        np.concatenate((np.cos(theta), -np.sin(theta)), 2),
        np.concatenate((np.sin(theta), np.cos(theta)), 2),
    ), 1)
    center = np.concatenate((xc.reshape(-1, 1, 1), yc.reshape(-1, 1, 1)), 1)
    points = R @ points + center

    # use the points to compute the affine transform that maps
    # these points back to the output square
    res = resolution
    points1 = np.array([[0, 0, res-1], [0, res-1, 0]], dtype='float32').T
    affines = []
    imgs = []
    for i in range(points.shape[0]):
        pts = points[i, :, :3].T.astype('float32')
        M = cv2.getAffineTransform(pts, points1)
        img = cv2.warpAffine(frame, M, (res, res), borderValue=127.5)
        imgs.append(img)
        affine = cv2.invertAffineTransform(M).astype('float32')
        affines.append(affine)
    if imgs:
        imgs = np.moveaxis(np.stack(imgs), 3, 1).astype('float32') / 127.5 - 1.0
        affines = np.stack(affines)
    else:
        imgs = np.zeros((0, 3, res, res))
        affines = np.zeros((0, 2, 3))

    return imgs, affines, points


def estimator_preprocess(src_img, detections, scale, pad):
    detections = denormalize_detections(detections[0], scale, pad)
    xc, yc, scale, theta = detection2roi(detections)
    img, affine, box = extract_roi(src_img, xc, yc, theta, scale)

    return img, affine, box


def denormalize_landmarks(landmarks, affines):
    landmarks = landmarks.reshape((landmarks.shape[0], -1, 3))
    landmarks[:, :, :2] *= resolution
    for i in range(len(landmarks)):
        landmark, affine = landmarks[i], affines[i]
        landmark = (affine[:, :2] @ landmark[:, :2].T + affine[:, 2:]).T
        landmarks[i, :, :2] = landmark
    return landmarks


def iris_preprocess(imgs, raw_landmarks):
    landmarks = raw_landmarks.reshape((-1, 3))

    imgs_cropped = []
    origins = []
    for i in range(len(imgs)):
        eye_left_center = landmarks[EYE_LEFT_CONTOUR, :2].mean(axis=0)
        eye_right_center = landmarks[EYE_RIGHT_CONTOUR, :2].mean(axis=0)

        x_left, y_left = map(int, np.round(eye_left_center - 32))
        # Horizontal flip
        imgs_cropped.append(imgs[i, :, y_left:y_left+64, x_left+63:x_left-1:-1])
        origins.append((x_left+63, y_left))

        x_right, y_right = map(int, np.round(eye_right_center - 32))
        imgs_cropped.append(imgs[i, :, y_right:y_right+64, x_right:x_right+64])
        origins.append((x_right, y_right))

    return np.stack(imgs_cropped), np.stack(origins)


def iris_postprocess(eyes, iris, origins, affines):
    eyes = eyes.copy().reshape((-1, 71, 3))
    iris = iris.copy().reshape((-1, 5, 3))

    # Horizontally flipped left eye processing
    eyes[::2, :, 0] = -eyes[::2, :, 0]
    iris[::2, :, 0] = -iris[::2, :, 0]

    eyes[:, :, :2] += origins[:, None]
    iris[:, :, :2] += origins[:, None]

    iris_landmarks = np.concatenate((eyes, iris), axis=1)
    iris_landmarks = iris_landmarks.reshape((eyes.shape[0] // 2, -1, 3))
    iris_landmarks = denormalize_landmarks(iris_landmarks / resolution, affines)

    iris_landmarks = iris_landmarks.reshape((-1, 2, 76, 3)).round().astype(int)
    eyes = iris_landmarks[:, :, :71]
    iris = iris_landmarks[:, :, 71:]

    return eyes, iris
#endif


static int detect_face(AILIANetwork* ailia_detection, const cv::Mat& mat_input, std::vector<cv::Mat>& mat_outputs)
{
    int status = AILIA_STATUS_SUCCESS;

    unsigned input_count;
    status = ailiaGetInputBlobCount(ailia_detection, &input_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetInputBlobCount failed %d\n", status);
        return status;
    }

    if (input_count != 1) {
        PRINT_ERR("ailiaGetInputBlobCount returned %u\n", input_count);
        return AILIA_STATUS_OTHER_ERROR;
    }

    unsigned int input_index;
    status = ailiaGetBlobIndexByInputIndex(ailia_detection, &input_index, 0);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobIndexByInputIndex failed %d\n", status);
        return status;
    }

    AILIAShape input_shape;
    status = ailiaGetBlobShape(ailia_detection, &input_shape, input_index, AILIA_SHAPE_VERSION);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
        return status;
    }

    int input_size = input_shape.x * input_shape.y * input_shape.z * input_shape.w * sizeof(float);

    assert(mat_input.total() * mat_input.elemSize() == input_size);

    status = ailiaSetInputBlobData(ailia_detection, mat_input.data, input_size, input_index);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaSetInputBlobData failed %d\n", status);
        return status;
    }

    status = ailiaUpdate(ailia_detection);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaUpdate failed %d\n", status);
        return status;
    }

    unsigned output_count;
    status = ailiaGetOutputBlobCount(ailia_detection, &output_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetOutputBlobCount failed %d\n", status);
        return status;
    }

    if (output_count != mat_outputs.size()) {
        PRINT_ERR("ailiaGetOutputBlobCount returned %u\n", output_count);
        return AILIA_STATUS_OTHER_ERROR;
    }

    for (int i = 0; i < (int)output_count; i++) {
        cv::Mat& mat_output = mat_outputs[i];

        unsigned int output_index;
        status = ailiaGetBlobIndexByOutputIndex(ailia_detection, &output_index, i);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobIndexByOutputIndex failed %d\n", status);
            return status;
        }

        AILIAShape output_shape;
        status = ailiaGetBlobShape(ailia_detection, &output_shape, output_index, AILIA_SHAPE_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetBlobShape failed %d\n", status);
            return status;
        }

        int output_size = output_shape.x * output_shape.y * output_shape.z * output_shape.w * sizeof(float);

        assert(output_shape.dim == 3);

        int size[] = {(int)output_shape.z, (int)output_shape.y, (int)output_shape.x};
        mat_output = cv::Mat(3, size, CV_32FC1);

        assert(mat_output.total() * mat_output.elemSize() == output_size);

        status = ailiaGetBlobData(ailia_detection, mat_output.data, output_size, output_index);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaSetOutputBlobData failed %d\n", status);
            return status;
        }
    }

    return AILIA_STATUS_SUCCESS;
}


// ======================
// Main functions
// ======================

static int recognize_from_image(AILIANetwork* ailia_detection, AILIANetwork* ailia_landmark, AILIANetwork* ailia_landmark2)
{
    int status = AILIA_STATUS_SUCCESS;

    // prepare input data
    cv::Mat mat_img;
    status = load_image(mat_img, image_path.c_str());
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }
    print_shape(mat_img, "input image shape: ");

    cv::Mat mat_rgb;
    cv:cvtColor(mat_img, mat_rgb, cv::COLOR_BGRA2RGB);

    cv::Mat mat_128;
    float scale;
    int pad[2];
    resize_pad(mat_rgb, mat_128, scale, pad);

    cv::Mat mat_input2n;
    normalize_image(mat_128, mat_input2n, "127.5");

    cv::Mat mat_input3;
    reshape_channels_as_dimension(mat_input2n, mat_input3);

    cv::Mat mat_input3t;
    transpose(mat_input3, mat_input3t);

    cv::Mat mat_input4;
    expand_dims(mat_input3t, mat_input4, 0);

    print_shape(mat_input4, "input data shape: ");

#if 0
    int dsize = mat_input4.size[0] * mat_input4.size[1] * mat_input4.size[2] * mat_input4.size[3];
    printf("%d\n", dsize);
    float* ddata = (float*)mat_input4.data;
    for (int i = 5000; i < 5040 && i < dsize; i++) {
        printf("%d %.03f\n", i, ddata[i]);
    }
    exit(0);
#endif

    // inference
    PRINT_OUT("Start inference...\n");
    if (benchmark) {
        PRINT_OUT("BENCHMARK mode\n");
        for (int i = 0; i < BENCHMARK_ITERS; i++) {
            clock_t start = clock();
            // TODO
            clock_t end = clock();
            PRINT_OUT("\tailia processing time %ld ms\n", ((end - start) * 1000) / CLOCKS_PER_SEC);
        }
    }
    else {
        // face detection
        std::vector<cv::Mat> mat_predictions(2);
        status = detect_face(ailia_detection, mat_input4, mat_predictions);
        if (status != AILIA_STATUS_SUCCESS) {
            return -1;
        }

        std::vector<cv::Mat> mat_detections;
        status = blazeface_postprocess(mat_predictions[0], mat_predictions[1], mat_detections);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("blazeface_postprocess failed %d\n", status);
            return -1;
        }

        printf("detections count: %lu\n", mat_detections.size());

// TODO
#if 0
        # Face landmark estimation
        if detections[0].size != 0:
            imgs, affines, box = iut.estimator_preprocess(
                src_img[:, :, ::-1], detections, scale, pad
            )
            estimator.set_input_shape(imgs.shape)
            landmarks, confidences = estimator.predict([imgs])

            # Iris landmark estimation
            imgs2, origins = iut.iris_preprocess(imgs, landmarks)
            estimator2.set_input_shape(imgs2.shape)
            eyes, iris = estimator2.predict([imgs2])

            eyes, iris = iut.iris_postprocess(eyes, iris, origins, affines)
            for i in range(len(eyes)):
                draw_eye_iris(
                    src_img, eyes[i, :, :16, :2], iris[i, :, :, :2], size=1
                )
#endif
    }

    cv::imwrite(save_image_path.c_str(), mat_img);

    PRINT_OUT("Program finished successfully.\n");

    return AILIA_STATUS_SUCCESS;
}


static int recognize_from_video(AILIANetwork* ailia_detection, AILIANetwork* ailia_landmark, AILIANetwork* ailia_landmark2)
{
    return AILIA_STATUS_SUCCESS;
}


int main(int argc, char **argv)
{
    int status = argument_parser(argc, argv);
    if (status != AILIA_STATUS_SUCCESS) {
        return -1;
    }

    // env list
    unsigned int env_count;
    status = ailiaGetEnvironmentCount(&env_count);
    if (status != AILIA_STATUS_SUCCESS) {
        PRINT_ERR("ailiaGetEnvironmentCount failed %d", status);
        return -1;
    }

    int env_id = AILIA_ENVIRONMENT_ID_AUTO;
    for (unsigned int i = 0; i < env_count; i++) {
        AILIAEnvironment* env;
        status = ailiaGetEnvironment(&env, i, AILIA_ENVIRONMENT_VERSION);
        PRINT_OUT("env_id : %d type : %d name : %s\n", env->id, env->type, env->name);
        if (args_env_id == env->id) {
            env_id = env->id;
        }
        if (args_env_id == -1 && env_id == AILIA_ENVIRONMENT_ID_AUTO){
            if (env->type == AILIA_ENVIRONMENT_TYPE_GPU) {
                env_id = env->id;
            }
        }
    }
    if (args_env_id == -1){
        PRINT_OUT("you can select environment using -e option\n");
    }

    // initialize detection net
    AILIANetwork *ailia_detection;
    {
        status = ailiaCreate(&ailia_detection, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(ailia_detection, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(ailia_detection);
            return -1;
        }

        PRINT_OUT("selected env name : %s\n", env_ptr->name);

        status = ailiaOpenStreamFile(ailia_detection, detection_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_detection));
            ailiaDestroy(ailia_detection);
            return -1;
        }

        status = ailiaOpenWeightFile(ailia_detection, detection_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(ailia_detection);
            return -1;
        }
    }

    // initialize landmark net
    AILIANetwork *ailia_landmark;
    {
        status = ailiaCreate(&ailia_landmark, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(ailia_landmark, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(ailia_landmark);
            return -1;
        }

        status = ailiaOpenStreamFile(ailia_landmark, landmark_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_landmark));
            ailiaDestroy(ailia_landmark);
            return -1;
        }

        status = ailiaOpenWeightFile(ailia_landmark, landmark_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(ailia_landmark);
            return -1;
        }
    }

    // initialize landmark2 net
    AILIANetwork *ailia_landmark2;
    {
        status = ailiaCreate(&ailia_landmark2, env_id, AILIA_MULTITHREAD_AUTO);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaCreate failed %d\n", status);
            if (status == AILIA_STATUS_LICENSE_NOT_FOUND || status==AILIA_STATUS_LICENSE_EXPIRED){
                PRINT_OUT("License file not found or expired : please place license file (AILIA.lic)\n");
            }
            return -1;
        }

        AILIAEnvironment *env_ptr = nullptr;
        status = ailiaGetSelectedEnvironment(ailia_landmark2, &env_ptr, AILIA_ENVIRONMENT_VERSION);
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaGetSelectedEnvironment failed %d\n", status);
            ailiaDestroy(ailia_landmark2);
            return -1;
        }

        status = ailiaOpenStreamFile(ailia_landmark2, landmark2_model.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenStreamFile failed %d\n", status);
            PRINT_ERR("ailiaGetErrorDetail %s\n", ailiaGetErrorDetail(ailia_landmark2));
            ailiaDestroy(ailia_landmark2);
            return -1;
        }

        status = ailiaOpenWeightFile(ailia_landmark2, landmark2_weight.c_str());
        if (status != AILIA_STATUS_SUCCESS) {
            PRINT_ERR("ailiaOpenWeightFile failed %d\n", status);
            ailiaDestroy(ailia_landmark2);
            return -1;
        }
    }

    if (video_mode) {
        status = recognize_from_video(ailia_detection, ailia_landmark, ailia_landmark2);
    }
    else {
        status = recognize_from_image(ailia_detection, ailia_landmark, ailia_landmark2);
    }

    ailiaDestroy(ailia_detection);
    ailiaDestroy(ailia_landmark);
    ailiaDestroy(ailia_landmark2);

    return status;
}
