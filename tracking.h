//
// Created by flo on 15.10.18.
//

#ifndef PANORAMA2CUBEMAP_TRACKING_H
#define PANORAMA2CUBEMAP_TRACKING_H

#include <string>
#include <vector>
#include "detect.h"

#include <opencv2/core/utility.hpp>

#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>

#include "cubetransform.h"

using namespace std;
using namespace cv;

class CustomMultiTracker{
public:
    CustomMultiTracker();
    void initialize_darknet (Mat & frame);
    MultiTracker multiTracker[6];
    void update(Mat & frame);
    int track_video_stream(char * filename);




private:
    vector<Rect2d> objects[6];
    MatDetector darknetDetector;
    std::vector<Ptr<Tracker>> algorithms[6]; // An Array of Vectors, one Vector for every Face side of a cube.


};

#endif //PANORAMA2CUBEMAP_TRACKING_H
