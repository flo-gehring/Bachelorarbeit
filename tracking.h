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

#include "cubetransform.h"

using namespace std;
using namespace cv;

#define P2C_AOI_FROM_FILE



class FootballPlayer {
public:
    FootballPlayer(Rect coordinate, int frame, string const & identifier);
    void addPosition(Rect coordinates, int frame);
    void update(Rect const & coordinates, int frame);

    Rect predictPosition(int frameNum);

    /*
     * Interpret as Follows: If i is an integer in "frames" at position X, then the Football Player appeared
     * in the Video in the i-th frame on the position saved in coordinates[X].
     */
    vector<Rect> coordinates;
    vector<int> frames;

    string identifier; // Display Name

    /*
     * The following Data will be (hopefully) usefull in identifying the Player after occlusion.
     */
    double x_vel, y_vel; // "Velocity in Pixels per frame"
    Mat hist; // Histogramm of the Player when he was first detected.
    Mat bgrShirtColor;

};


class Region{
public:
    Region(const Rect &coordinates,  FootballPlayer * ptrPlayer);
    Region(Rect coordinates);
    Region(Region const & r1);

    Rect coordinates;
    FootballPlayer * playerInRegion;
    unsigned char bgrShirtColor[3];
    unsigned char labShirtColor[3]; // Color Informaion given in the LAB format


    static bool regionsIntersect(const Region & r1, const Region & r2);
    static bool regionsInRelativeProximity(Region const & r1, Region const &r2, int framesPassed);
    void createColorProfile(Mat const & frame);

    void updatePlayerInRegion(int frameNum);

    /* Calcs the k-mean for colorCount colors and returns them converted into the L*A*B color space.
     * Side effect: sets colorInformation property.
     */
    Mat getLabColors(Mat const & frame, int colorCount);

    /*
     * Returns a Mat of size 3 with color
     * of the T-Shirt / Shorts of the Football Player, saved as one Pixel in BGR Format.
     */
    Mat getShirtColor(Mat const& frame);

};


class MetaRegion{
public:
    Rect area;
    vector<Region *> metaOldRegions;
    vector<Region *> metaNewRegions;
    int* matchOldAndNewRegions(Mat  frame, int * matching, int currentFrame);

};

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



protected:


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

    #ifdef P2C_AOI_FROM_FILE
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
