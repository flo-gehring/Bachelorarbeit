//
// Created by flo on 15.10.18.
//

#ifndef PANORAMA2CUBEMAP_TRACKING_H
#define PANORAMA2CUBEMAP_TRACKING_H

#include <string>
#include <vector>
#include "../Detectors/detect.h"

#include <opencv2/core/utility.hpp>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/tracking.hpp>

#include "opencv2/optflow/pcaflow.hpp"
#include "opencv2/optflow/sparse_matching_gpc.hpp"
#include "opencv2/optflow/motempl.hpp"
#include <opencv2/optflow.hpp>


#include <iostream>
#include <cstring>
#include <ctime>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <stdio.h>

#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <algorithm>
#include "array"

#include "TrackingHelpers.h"


class Region;
class MetaRegion;
class FootballPlayer;

using namespace std;

class RegionTracker{
public:

    double assignmentThreshold;
    double minDistanceThreshold;

    RegionTracker();
    explicit RegionTracker(const char * aoiFilePath, const char * videoPath = nullptr);

    void setAOIFile(const char * aoiFilePath);
    void enableVideoSave(const char * videoFilePath);


    void trackVideo(const char *filename);
    int initialize(Mat frame);
    bool update(Mat frame);

    void printTrackingResults(const char * resultFilePath);


    virtual ~RegionTracker();

    Mat matCurrentFrame;
    Mat matLastFrame; // Add to support optical Flow stuff.

    FILE * roiData;
    FILE * debugData;
    double calcWeightedSimiliarity(const Region  * oldRegion, const Region *newRegion, Rect area);
    FILE * detectorData;


    // Switch to save Data about the decision making process
    bool analysisData;
    FILE * analysisDataFile;
    void setupAnalysisOutFile(const char * filename);

#undef PANORAMA2CUBEMAP_TRACKING_H
#ifdef PANORAMA2CUBEMAP_TRACKING_H
    DetectionFromFile darknetDetector;
#else
    MatDetector darknetDetector;
#endif

// protected:

    vector<MetaRegion> calcMetaRegions();
    void interpretMetaRegions(vector<MetaRegion> & mr);
    void assignRegions(MetaRegion  & metaRegion);
    FootballPlayer * createNewFootballPlayer(Rect const &);
    FootballPlayer * createAmbiguousPlayer(Rect const &);

    vector<Rect> detectOnFrame(Mat  & frame);


    /*
     * Factory Methods: Get Regions etc.
     */
    void deleteFromOutOfSight(FootballPlayer *);
    void addToOutOfSight(Region *);

    // Output Methods
    void drawOnFrame(Mat frame, vector<MetaRegion> const & mr);
    void printInfo(vector<MetaRegion> const &);


    // Data Storage
    vector<Region *> outOfSightRegions;
    vector<Region *> noMatchFound;

    vector<Region> regionsNewFrame;
    vector<Region> regionLastFrame;

    vector<FootballPlayer *> footballPlayers;

    unordered_map<FootballPlayer *, FootballPlayer *> occludedPlayers;

    int objectCounter = 0;
    int currentFrame = 0;


    bool saveVideo;
    char * saveVideoPath;

    /*
     *  Color Recognition
     */
    Ptr<BackgroundSubtractor> pBGSubtractor;
    Mat foregroundMask;

    // OPticalFlow
    void calcOpticalFlow(Rect const & area);



};

/*
 * Some Helper Functions.
 */
void textAboveRect(Mat frame, Rect rect, string text);
void histFromRect(Mat const & input, Rect const & rect, Mat & output);
/* Color difference as described by https://en.wikipedia.org/wiki/Color_difference#CIE94
 * Look at this Page for information
*/
double deltaECIE94(unsigned char L1, char  a1, char b1, unsigned char L2, char a2, char b2);

/*
 * Performs k-Mean on frame, with k = clusterCount.
 * labes and centers are as described in the opencv documentation.
 * Labels has one Row and frame.cols * frame.rows columns, representing the frames pixel like this : frame[row, col] = labels[row * frame.cols + col]
 */
Mat helperBGRKMean(Mat const &frame, int clusterCount, Mat &labels, Mat &centers);
bool playerInRegionVector(FootballPlayer * fp, vector<Region> const & vr);
// void optimizeWeightSelection(int rows, int cols, double * weightMatrix, int * selected)
#endif //PANORAMA2CUBEMAP_TRACKING_H


void printDetectionsToFile(FILE * output, int frameNumber, std::vector<Rect> const & detections);