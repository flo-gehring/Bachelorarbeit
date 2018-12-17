//
// Created by flo on 03.12.18.
//

#ifndef PANORAMA2CUBEMAP_TRACKINGHELPERS_H
#define PANORAMA2CUBEMAP_TRACKINGHELPERS_H


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


#include "cubetransform.h"
#include "tracking.h"

using namespace std;
using namespace cv;

#define P2C_AOI_FROM_FILE


class RegionTracker;

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

    bool isAmbiguous;

};




class Region{
public:
    Region() = default;
    Region(const Rect &coordinates,  FootballPlayer * ptrPlayer);
    Region(Rect coordinates);
    Region(Region const & r1);

    Rect coordinates;
    FootballPlayer * playerInRegion;
    unsigned char bgrShirtColor[3];
    unsigned char labShirtColor[3]; // Color Informaion given in the LAB format


    static bool regionsIntersect(const Region & r1, const Region & r2);
    static bool regionsInRelativeProximity(Region const & r1, Region const &r2, int framesPassed);
    void createColorProfile(Mat const & frame, Mat const & foregroundMask);

    void updatePlayerInRegion(int frameNum);

    /* Calcs the k-mean for colorCount colors and returns them converted into the L*A*B color space.
     * Side effect: sets colorInformation property.
     */
    Mat getLabColors(Mat const & frame, int colorCount);

    /*
     * Returns a Mat of size 3 with color
     * of the T-Shirt / Shorts of the Football Player, saved as one Pixel in BGR Format.
     */
    Mat getShirtColor(Mat const& frame, Mat const & foregroundMask);

};


class MetaRegion{
public:
    Rect area;
    vector<Region *> metaOldRegions;
    vector<Region *> metaNewRegions;
    int *  matchOldAndNewRegions(Mat frame, int * matching, int frameNum, RegionTracker * rt);

};


#endif //PANORAMA2CUBEMAP_TRACKINGHELPERS_H
