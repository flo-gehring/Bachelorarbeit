//
// Created by flo on 15.10.18.
//

#ifndef PANORAMA2CUBEMAP_TRACKING_H
#define PANORAMA2CUBEMAP_TRACKING_H

#include <string>
#include <vector>
#include <detect.h>

#include <opencv2/core/utility.hpp>

#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>

using namespace std;
using namespace cv;

class Tracker{
public:
    Tracker();
    void initialize_darknet ();
    MultiTracker trackers("KCF");  // Just use KCF for now



private:
    vector<AbsoluteBoundingBoxes> objects;
    MatDetector darknetDetector();

};

#endif //PANORAMA2CUBEMAP_TRACKING_H
