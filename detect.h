//
// Created by flo on 06.10.18.
//

#ifndef PANORAMA2CUBEMAP_DETECT_H
#define PANORAMA2CUBEMAP_DETECT_H

#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <cv.h>
#include <stack>

// The OpenCV definition is needed to include every function in the darknet.h file.
#ifndef OPENCV
#define OPENCV
#endif

#include "/home/flo/Workspace/darknet/include/darknet.h"


image ipl_to_image(IplImage* src); // Aus darknet sources.
int size_network(network *net);

struct AbsoluteBoundingBoxes {
    int top, left, right, bottom; // 2 Corner coordinates
    float prob; // Confidence of network
    std::string class_name;
};

class MatDetector {
// Config Parameters
public:
    MatDetector();

std::stack<AbsoluteBoundingBoxes> found;
char *  cfgfile;
char * weightfile;
char* datacfg;
float thresh = 0.5f;
char ** names;
int classes = 20;
int delay;
char * prefix;
int avg_frames;
float hier;
int w;
int h;
int frames;
int fullscreen;

// ---------

char **demo_names;
image **demo_alphabet;
int demo_classes;

network *net;

void * cap;
float fps;
float demo_thresh = 0.5f;

float demo_hier = 0.5f;
int running = 1;

image darknet_image;
int demo_frame;
int demo_index;
float **predictions;
float *avg;
int demo_done;
int demo_total;
double demo_time;


void detect_and_display(cv::Mat input_mat);
void *detect_in_thread(void *ptr);
void remember_network(network *net);

detection *avg_predictions(network *net, int *nboxes);

void print_detections(image im, detection *dets, int num);

};

#endif //PANORAMA2CUBEMAP_DETECT_H
