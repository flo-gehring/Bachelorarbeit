//
// Created by flo on 30.12.18.
//

/*
 * Teile des Codes folgender Seite entnommen:
 * https://github.com/spmallick/learnopencv/blob/master/Mask-RCNN/mask_rcnn.cpp
 */
#ifndef PANORAMA2CUBEMAP_TESTMASKRCNN_H

#include <string>
#include <iostream>
#include <vector>

#include "PanoramaTracking.h"
#include "PanoramaTrackingImplementations.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

    using namespace std;
    using namespace cv;
    using namespace dnn;

// Draw the predicted bounding box
    void drawBox(Mat &frame, int classId, float conf, Rect box, Mat &objectMask);

// Postprocess the neural network's output for each frame
    void postprocess(Mat &frame, const vector<Mat> &outs);

    void detectOnVideo(const char *filePath, Projector *projector);


#define PANORAMA2CUBEMAP_TESTMASKRCNN_H

#endif //PANORAMA2CUBEMAP_TESTMASKRCNN_H
