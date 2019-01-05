//
// Created by flo on 28.10.18.
//

#ifndef PANORAMA2CUBEMAP_OPENCV_TRACKING_H
#define PANORAMA2CUBEMAP_OPENCV_TRACKING_H

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <cstring>
#include <vector>
#include <ctime>

#include "../cubetransform.h"
#include "../Detectors/detect.h"

using namespace std;
using namespace cv;

void trackVideo(char * videoPath, string const & outFile ,string const & tracker, MatDetector);

void createTrackerByName(string name);

#endif //PANORAMA2CUBEMAP_OPENCV_TRACKING_H
