//
// Created by flo on 09.10.18.
//

#ifndef PANORAMA2CUBEMAP_OPENCV_DETECT_H
#define PANORAMA2CUBEMAP_OPENCV_DETECT_H

#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <math.h>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace dnn;

struct prediction {
    int left, top, right, bottom;
    float confidence;
    int classid;
};

class YOLODetector{

public:

    /*
     *  Object Variables
     */
    Net net;

    int inpWidth = 416;
    int inpHeight = inpWidth;

    float confThreshold = 0.5;
    float nmsThreshold = 0.5;
    std::vector<String> classesVec;

    std::vector<prediction> predictions; // after Detect and Display Method, the preoictions will be safed here.

    /*
     * Functions
     */

    YOLODetector(char* pathToConfig, char* pathToWeight, char* pathToNames );
    void detect(Mat & frame);


private:
    Mat inputBlob, detectionMat;
    void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);



};

std::vector<String> getOutputsNames(const Net& net);

#endif //PANORAMA2CUBEMAP_OPENCV_DETECT_H
