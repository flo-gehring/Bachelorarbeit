//
// Created by flo on 24.11.18.
//

#ifndef PANORAMA2CUBEMAP_PANORAMATRACKING_H
#define PANORAMA2CUBEMAP_PANORAMATRACKING_H

#include <vector>
#include <tuple>
#include <map>

#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;

const static  char * TRACKER_NAMES[7] = {

        "Boosting",
        "KCF",
        "MedianFlow",
        "MIL",
        "MOSSE",
        "TLD"
};

class Projector{
public:


    /*
     * Call this Function before beginning to project a new Frame. It will return the number of total Projections
     * and prepare the class to work on a new Frame.
     */
    virtual int beginProjection() = 0;

    /*
     * Call this Functions to project the input Mat onto the output Mat.
     * To deal with the whole Frame, call it as often as beginProjection returns.
     * The Method returns the current Projection Number.
     */
    virtual int project(Mat const & input, Mat & output) = 0;

    /*
     * Project with a given Projection Id.
     */
    virtual void project(Mat const & input, int projectionId, Mat & output) = 0;

    /*
     * Get the coordinates the input Rect has on the input Mat, given the projection Number.
     */
    virtual Rect sourceCoordinates(Mat const & input, Rect const & coordinates, int projectionNumber) = 0;

    // Width and Height of the projected Frame.
     int projectionWidth;
     int projectionHeight;

protected:
    short currentProjectionId;



};

class DetectorWrapper{
public:
    virtual std::vector<Rect> detect(Mat const & input) = 0;
};

class PanoramaTracking {
public:



    FILE * trackingResult;

    PanoramaTracking(DetectorWrapper * detector, const char * tracker, Projector * projector);
    DetectorWrapper * detector;
    const char * trackerType;
    Projector * projector;
    void trackVideo(const char * fileName, const char * videoFile = nullptr);

    void createNewTracker(Mat const & projection, Rect const & coordinates, int projectonId);

    bool update();

    int detectionUpdateIntervall;
protected:

    int frameCounter;
    time_t startTime;



    Mat  currentFrame;
    std::vector<Ptr<Tracker>> trackers;
    std::map<int, std::vector<Ptr<Tracker>>> projectionIdMapping;

    std::map<Ptr<Tracker>, int> objectIdentifier;
    std::vector<std::tuple<Rect, Ptr<Tracker>>> panoramaAOI;



};

void printMOTFormatTrackingLine(FILE * outfile, int frame, int id, Rect coordinates);


#endif //PANORAMA2CUBEMAP_PANORAMATRACKING_H
