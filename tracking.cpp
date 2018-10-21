//
// Created by flo on 15.10.18.
//

#include "tracking.h"


int RegionTracker::initialize(Mat frame) {
    return 0;
}

bool RegionTracker::update(Mat frame) {
    return false;
}

void RegionTracker::calcMatrix() {

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
