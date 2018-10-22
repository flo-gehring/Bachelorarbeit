//
// Created by flo on 15.10.18.
//

#include "tracking.h"



int RegionTracker::initialize(Mat frame) {

    vector<Rect> detectedRects;

    // All the detected Objects will be stored as recangles in detectedRects
    detectOnFrame(frame, detectedRects);

    Rect panoramaCoords;

    int objectCount = 0;

    for(auto it = detectedRects.begin(); it != detectedRects.end(); ++it){

        footballPlayers.emplace_back(FootballPlayer((*it), 1, to_string(objectCount) ));
        regionsNewFrame.emplace_back(Region(*it, &footballPlayers.back()));
        objectCount++;

    }
    return 0;
}

bool RegionTracker::update(Mat frame) {

    // Update Region Vectors, new regions from last frame are now old, get new Regions from detector
    regionLastFrame.swap(regionsNewFrame);
    regionsNewFrame.clear();

    vector<Rect> newRects;
    detectOnFrame(frame, newRects);

    for(auto it = newRects.begin(); it != newRects.end(); ++it){

        regionsNewFrame.emplace_back(Region(*it));
    }

    calcMatrix();

    interpretMatrix();

    return false;
}

void RegionTracker::calcMatrix() {

    matrix = Mat::zeros(Size(regionLastFrame.size(),
            regionsNewFrame.size()), CV_8UC1);


    Region & oldRegion = regionLastFrame.at(0);
    Region & newRegion = regionsNewFrame.at(0);

    uchar * row_ptr;

    for (int  oldRegionCounter = 0; oldRegionCounter < regionLastFrame.size(); ++oldRegionCounter){
        oldRegion = regionLastFrame.at(oldRegionCounter);

        row_ptr = matrix.ptr<uchar>(oldRegionCounter);

        for(int  newRegionCounter = 0; newRegionCounter < regionsNewFrame.size(); ++newRegionCounter){
            newRegion = regionsNewFrame.at(newRegionCounter);

            // If the Regions are associated somehow,
            // highlight this by setting the appropriate entry in the matrix to 1.
            if(Region::regionsAssociated(oldRegion, newRegion)){
                row_ptr[newRegionCounter] = 1;
            }

        }

    }


}

void RegionTracker::interpretMatrix() {

}

void RegionTracker::handleAppearance(int regionIndex) {

}

void RegionTracker::handleDissapearance(int regionIndex) {

}

void RegionTracker::handleSplitting(int regionIndex, int *splitInto) {

}

void RegionTracker::handleMerging(int *regions, int mergeInto) {

}

void RegionTracker::handleContinuation(int regionIndexOld, int regionIndexNew) {

}

void RegionTracker::detectOnFrame(Mat frame, vector<Rect> &detected) {

    int sideLength = 500;

    Mat face;
    Rect panoramaCoords;
    for(int faceId = 0; faceId < 6; ++faceId){
        createCubeMapFace(frame, face, faceId, sideLength, sideLength);
        darknetDetector.detect_and_display(face);

        for(auto detectedObjects = darknetDetector.found.begin();
        detectedObjects != darknetDetector.found.end(); ++detectedObjects){

            mapRectangleToPanorama(frame, faceId, sideLength, sideLength, (* detectedObjects).rect, panoramaCoords);

            detected.push_back(panoramaCoords);

        }

    }

}

Region::Region(const Rect &coordinates, FootballPlayer *playersInRegion) {
    this->coordinates = coordinates;
    this->playersInRegion.push_back(playersInRegion);
}

FootballPlayer::FootballPlayer(Rect coordinate, int frame, string identifier) {

    coordinates.emplace_back(coordinate);
    frames.emplace_back(frame);
    this->identifier = identifier;

}

Region::Region(Rect coordinates) {
    this->coordinates = coordinates;
}


bool Region::regionsAssociated(const Region &r1, const Region &r2) {

}