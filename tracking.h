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
#include <iomanip>
#include <stdio.h>
#include <unordered_set>
#include <algorithm>

#include "cubetransform.h"

using namespace std;
using namespace cv;




class FootballPlayer {
public:
    FootballPlayer(Rect coordinate, int frame, string identifier);
    void addPosition(Rect coordinates, int frame);
    void update(Rect const & coordinates, int frame);

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
    int x_vel, y_vel; // "Velocity in Pixels per frame"
    Mat hist; // Histogramm of the Player when he was first detected.
    Mat colorInformation;
};


class Region{
public:
    Region(const Rect &coordinates,  FootballPlayer * ptrPlayer);
    Region(Rect coordinates);
    Region(Region const & r1);

    Rect coordinates;
    FootballPlayer * playerInRegion;
    Mat colorInformation; // Color Informaion given in the LAB format

    static bool regionsIntersect(const Region & r1, const Region & r2);
    static bool regionsInRelativeProximity(Region const & r1, Region const &r2, int framesPassed);
    void updatePlayerInRegion(const Region * oldRegion, int frameNum);

    /* Calcs the k-mean for colorCount colors and returns them converted into the L*A*B color space.
     * Side effect: sets colorInformation property.
     */
    Mat getLabColors(Mat const & frame, int colorCount);

    /*
     * Returns an array of size 3 with the L*A*B colors of the T-Shirt / Shorts of the Football Player.
     * Don't forget to delete[].
     */
    int* getShirtColor(Mat const& frame);

};


class MetaRegion{
public:
    Rect area;
    vector<Region *> metaOldRegions;
    vector<Region *> metaNewRegions;
    int* matchOldAndNewRegions(Mat  frame, int * matching);

};

class RegionTracker{
public:
    int initialize(Mat frame);
    bool update(Mat frame);

    void workOnFile(char * filename);

    virtual ~RegionTracker();
    Mat matCurrentFrame;

protected:

    vector<MetaRegion> calcMetaRegions();
    void interpretMetaRegions(vector<MetaRegion> & mr);
    void assignRegions(MetaRegion  & metaRegion);
    FootballPlayer * createNewFootballPlayer(Rect const &);

    /*
     * Factory Methods: Get Regions etc.
     */

    FootballPlayer playerById(string id);
    void deleteFromOutOfSight(FootballPlayer *);
    void addToOutOfSight(Region *);


    void detectOnFrame(Mat frame, vector<Rect> & detected);
    void drawOnFrame(Mat frame);


    double areaThreshold = 0.3;

    static const int matrixDimensions = 22;
    unsigned short  matrix[matrixDimensions][matrixDimensions];


    vector<Region> outOfSightRegions;
    vector<Region *> noMatchFound;

    vector<Region> regionsNewFrame;
    vector<Region> regionLastFrame;

    vector<FootballPlayer *> footballPlayers;

    DetectionFromFile darknetDetector;

    int objectCounter = 0;
    int currentFrame = 0;

    FILE * roiData;
    FILE * debugData;



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
void helperRGBKMean(Mat const & frame, int clusterCount, Mat & labels, Mat & centers);

#endif //PANORAMA2CUBEMAP_TRACKING_H
