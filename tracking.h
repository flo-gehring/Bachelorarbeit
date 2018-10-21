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


class FootballPlayer {

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
    Mat hist; // Histogramm of the Player
};

class Region{
    Rect coordinates;
    vector<FootballPlayer *> playersInRegion;
};

class RegionTracker{
public:
    int initialize(Mat frame);
    bool update(Mat frame);

protected:
    /*
     * Calculates a Matrix as described in the Paper from regionsNewFrame and regionsLastFrame.
     */
    void calcMatrix();

    /*
     * Applies the handle* Methods according to the calculated Matrix.
     */
    void interpretMatrix();

    // Handles the Appearance of a completely new Region which has index regionIndex in regionsNewFrame.
    void handleAppearance(int regionIndex);

    // Handles the Dissapearance of a Region which has index regionIndex in regionsLastFrame.
    void handleDissapearance(int regionIndex);

    // Handles the Splitting of the Region regionsLastFrame[regionsIndex] into the given new regions.
    void handleSplitting(int regionIndex, int splitInto[]);

    // Handles the merging of multiple old regions into a single new region.
    void handleMerging(int regions[], int mergeInto);

    // If an old region directly corresponds to a new region, this method is applied.
    void handleContinuation(int regionIndexOld, int regionIndexNew);



    Mat matrix;


    vector<Region> outOfSightRegions;

    vector<Region> regionsNewFrame;
    vector<Region> regionLastFrame;

    vector<FootballPlayer> footballPlayers;

    MatDetector darknetDetector;


};
#endif //PANORAMA2CUBEMAP_TRACKING_H
