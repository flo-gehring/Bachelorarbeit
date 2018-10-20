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

    void initialize_darknet (Mat & frame);
    MultiTracker multiTracker[6];
    void update(Mat & frame);
    int track_video_stream(char * filename);




private:
    MatDetector darknetDetector;
    int no_faces = 6;



};

struct TrackedObject{
    string identifier;  //Identifier of Object
    vector<int> frames; // The Frames in which the Object occurs
    vector<AbsoluteBoundingBoxes> occurences; // The Occurences of the Object
    int currentFaceId;
    Mat histogramm;
};

class DarknetTracker{
public:
    void initialize(Mat & frame);
    void update(Mat & frame);
    int track_video_stream(char  * file);

    void drawObjects(Mat & frame, int frameNum);

    string baseName = "Player";

private:

    vector<vector<TrackedObject>> allTrackedObjects;

    MatDetector darknetDetector;

    unsigned int numberObjectsLastFrame;
    unsigned int maxNumberObjects;

    unsigned int sideLength = 500;

    int detectObjects(Mat &frame,  vector<TrackedObject> & objects);

    static Mat calcHistForRect(Mat inputImage, Rect rectangle);

    /* Calculate the intersectios a single Object has with a vector of Objects
     */
    void getIntersections(TrackedObject trackedObject, vector<TrackedObject> & possibleIntersections,
            vector<TrackedObject &> & intersections );

};
#endif //PANORAMA2CUBEMAP_TRACKING_H
