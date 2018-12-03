//
// Created by flo on 15.10.18.
//

#ifndef PANORAMA2CUBEMAP_TRACKING_H
#define PANORAMA2CUBEMAP_TRACKING_H

#include <string>
#include <vector>
#include "detect.h"

#include <opencv2/core/utility.hpp>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <cstring>
#include <ctime>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <stdio.h>
#include <unordered_set>
#include <algorithm>
#include "array"

#include "TrackingHelpers.h"

class RegionTracker{
public:

    RegionTracker();
    explicit RegionTracker(const char * aoiFilePath, const char * videoPath = nullptr);

    void setAOIFile(const char * aoiFilePath);
    void enableVideoSave(const char * videoFilePath);


    void trackVideo(const char *filename);
    int initialize(Mat frame);
    bool update(Mat frame);



    virtual ~RegionTracker();


    Mat matCurrentFrame;

    FILE * roiData;
    FILE * debugData;
    double calcWeightedSimiliarity(Region  * oldRegion, Region *newRegion, Rect area);


    // Switch to save Data about the decision making process
    bool analysisData;
    FILE * analysisDataFile;

protected:


    void setupAnalysisOutFile(const char * filename);


    vector<MetaRegion> calcMetaRegions();
    void interpretMetaRegions(vector<MetaRegion> & mr);
    void assignRegions(MetaRegion  & metaRegion);
    FootballPlayer * createNewFootballPlayer(Rect const &);
    vector<Rect> detectOnFrame(Mat  & frame);


    /*
     * Factory Methods: Get Regions etc.
     */

    FootballPlayer playerById(string id);
    void deleteFromOutOfSight(FootballPlayer *);
    void addToOutOfSight(Region *);

    // Output Methods
    void drawOnFrame(Mat frame, vector<MetaRegion> const & mr);
    void printInfo(vector<MetaRegion> const &);

    vector<Region *> outOfSightRegions;
    vector<Region *> noMatchFound;

    vector<Region> regionsNewFrame;
    vector<Region> regionLastFrame;

    vector<FootballPlayer *> footballPlayers;

    #ifdef UNDEF
    DetectionFromFile darknetDetector;
    #else
    MatDetector darknetDetector;
    #endif

    int objectCounter = 0;
    int currentFrame = 0;


    bool saveVideo;
    char * saveVideoPath;



};

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
void helperBGRKMean(Mat const &frame, int clusterCount, Mat &labels, Mat &centers);

// void optimizeWeightSelection(int rows, int cols, double * weightMatrix, int * selected)
#endif //PANORAMA2CUBEMAP_TRACKING_H
