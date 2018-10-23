//
// Created by flo on 15.10.18.
//

#include "tracking.h"



int RegionTracker::initialize(Mat frame) {
    currentFrame = 0;
    vector<Rect> detectedRects;

    // All the detected Objects will be stored as recangles in detectedRects
    detectOnFrame(frame, detectedRects);

    Rect panoramaCoords;

    objectCounter = 0;

    for(auto it = detectedRects.begin(); it != detectedRects.end(); ++it){

        footballPlayers.emplace_back(FootballPlayer((*it), 1, to_string(objectCounter) ));
        regionsNewFrame.emplace_back(Region(*it, &footballPlayers.back()));
        objectCounter++;

    }

    return objectCounter;
}

bool RegionTracker::update(Mat frame) {
    ++currentFrame;
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

/*
 * Construct a Matrix which shows relationships between Regions in the last and the new Frame.
 * For every two Regions which are associated whith each other, there is a "1" entered.
 */
void RegionTracker::calcMatrix() {

    for (int i = 0; i < 22 ; ++i){
        for (int r = 0; r<22; ++r){
            matrix[i][r] = 0;
        }
    }


    Region & oldRegion = regionLastFrame.at(0);
    Region & newRegion = regionsNewFrame.at(0);


    for (int  oldRegionCounter = 0; oldRegionCounter < regionLastFrame.size(); ++oldRegionCounter){

        oldRegion = regionLastFrame.at(oldRegionCounter);



        for(int  newRegionCounter = 0; newRegionCounter < regionsNewFrame.size(); ++newRegionCounter){
            newRegion = regionsNewFrame.at(newRegionCounter);

            // If the Regions are associated somehow,
            // highlight this by setting the appropriate entry in the matrix to 1.
            if(Region::regionsAssociated(oldRegion, newRegion)){
                matrix[oldRegionCounter][ newRegionCounter ] = 1;
            }

        }

    }


}

/*
 * Interpret the Matrix and handle the entries.
 * Cases:
 *  Old Matrices [associated with] New Matrices:
 *
 *  0               1(or more)  -> Appearance
 *  1               1           -> Continuation
 *  1 (or more)     0           -> Dissapearance
 *  1               n > 1       -> Split
 *  n > 1           1           -> Merge
 */
void RegionTracker::interpretMatrix() {

    int associationCounter;
    vector<int> associatedIndexes;

    // Iterate Row wise to get information about


    int row = 0;

    for(int row = 0; row< regionLastFrame.size() ; ++row){

        associationCounter = 0;
        associatedIndexes.clear();

        for(int col = 0; col < regionsNewFrame.size() ; ++col) {
            if (matrix[row][col] != 0) {
                associatedIndexes.push_back(col);
                ++associationCounter;
            }
        }


        if(associationCounter == 0){ // Old Regions Dissapears
            handleDisapearance(row);
        }
        else if(associationCounter  == 1){ // Either Merge or Continue
            handleContinuation(row, associatedIndexes[0]);
        }
        else if(associationCounter > 1) { // Split
            handleSplitting(row, &associatedIndexes[0], associationCounter);
        }

        ++row;

    }



    // Iterate Column Wise
    for (int col = 0; col< regionsNewFrame.size() ; ++col){

        associationCounter = 0;
        associatedIndexes.clear();
        for (int row = 0; row < regionLastFrame.size(); ++row){


            if( matrix[row][col] != 0){
                ++associationCounter;
                associatedIndexes.push_back(row);
            }
        }
        if(associationCounter == 0){ // Region appeared
            handleAppearance(col);
        }
        else if(associationCounter == 1){ // Either Continuity or split

        }
        else if(associationCounter > 1){ // handle Merge
            handleMerging(&associatedIndexes[0], associationCounter, col);
        }
    }

}


/*
 * Check if the Region is completely new, or if its a region that reappeared.
 * Assumption: Only one Objects is in this Region
 * Fill Football Player with information
 */
void RegionTracker::handleAppearance(int regionIndex) {
    Region newRegion = regionsNewFrame[regionIndex];
    for (auto region = outOfSightRegions.begin(); region != outOfSightRegions.end(); ++region){

        if(Region::regionsAssociated(* region, regionsNewFrame[regionIndex])){
            // TODO: Perform Additional Checking and such. This is very basic and probably not usefull

            region->coordinates = newRegion.coordinates;
            region->updateObjectsInRegion(currentFrame);
            regionsNewFrame[regionIndex] = *region;
            outOfSightRegions.erase(region);

            return;
        }
    }
    // If no suiting out of sight region was found, create a new one.
    ++objectCounter;
    footballPlayers.emplace_back(FootballPlayer(newRegion.coordinates, currentFrame, to_string(objectCounter)));

    newRegion.playersInRegion.push_back(& footballPlayers.back());
}

void RegionTracker::handleDisapearance(int regionIndex) {

    outOfSightRegions.push_back(regionLastFrame[regionIndex]);

}

/*
 * If a region splits into multiple other Regions, try to determine where each player in the old region went.
 * At first, we'll simply compare histogramms.
 */
void RegionTracker::handleSplitting(int regionIndex, int *splitInto, int num) {

    Region oldRegion = regionLastFrame[regionIndex];

    vector<Region *> newRegions;
    for(int i = 0; i < num; ++i) {
        newRegions.push_back(&regionsNewFrame[*(splitInto + i)]);
    }

    /*  1.  If there are less objects in the old region than there are new Regions, then try to determine where the old objects
            went and create new ones.
        2.  If the amount of objects in the old Region is the same as there are new Regions, then try to assign the objects.
        3. If there are more objects in the old Region than there are new Regions, do some precission guesswork and assign the objects.
     */
    if(oldRegion.playersInRegion.size() <= newRegions.size()){
        // TODO: Do this right.
        // For now, i simply put the First few tracked objects into the first few regions and create some new Players
        int playerCounter = 0;
        for(auto r =  newRegions.begin(); r != newRegions.end(); ++r){

            if(playerCounter < newRegions.size())
                (*r)->playersInRegion.push_back(oldRegion.playersInRegion[playerCounter]);
            else{ // New Football player

                ++objectCounter;
                footballPlayers.emplace_back(FootballPlayer((*r)->coordinates, currentFrame, to_string(objectCounter)));

                (*r)->playersInRegion.push_back(& footballPlayers.back());
            }
        }

    }
    else{
        // TODO this |||
        // TODO      vvv
        // Assign the first couple of players to the first region, the rest gets one player
        int playerCounter;
        for (playerCounter  = 0; playerCounter < newRegions.size() - oldRegion.playersInRegion.size(); ++playerCounter){
            newRegions[0]->playersInRegion.emplace_back(oldRegion.playersInRegion[playerCounter]);
        }
        for(auto r: newRegions){

            r->playersInRegion.emplace_back(oldRegion.playersInRegion[playerCounter]);
            ++playerCounter;
            r->updateObjectsInRegion(currentFrame);
        }
    }
}

/*
 * Throw stuff together in one region.
 */
void RegionTracker::handleMerging(int *regions, int num, int mergeInto) {

    Region * newRegion = &regionsNewFrame[mergeInto];
    vector<Region *> oldRegions;
    for(int i = 0; i < num; ++i) {
        oldRegions.push_back(&regionLastFrame[*(regions + i)]);
    }

    // Insert all the tracked Objects into the new Region
    for(auto r = oldRegions.begin(); r != oldRegions.end(); ++r ){
        newRegion->playersInRegion.insert(newRegion->playersInRegion.end(),
                (*r)->playersInRegion.begin(), (*r)->playersInRegion.end());
    }

    // Update the tracked Objects
    newRegion->updateObjectsInRegion(currentFrame);
}

/*
 * Supply the new region with the Players of the old region.
 */
void RegionTracker::handleContinuation(int regionIndexOld, int regionIndexNew) {
    //TODO: This.
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

void RegionTracker::drawOnFrame(Mat frame) {
    for (Region & r: regionsNewFrame){
        string id;
        for(FootballPlayer * player : r.playersInRegion) id += " " + player->identifier;

        textAboveRect(frame, r.coordinates, id);
        rectangle(frame, r.coordinates, Scalar(0, 0, 255), 2);

    }
}

void RegionTracker::workOnFile(char *filename) {
    VideoCapture video(filename);
    Mat frame, resizedFrame;
    const char * windowName = "Occlusion Tracker";
    namedWindow(windowName);

    if(! video.isOpened()){
        cerr << "Could not open file";
        exit(-1);
    }

    video >> frame;
    if(frame.empty()) {
        cerr << "Empty video";
        exit(-1);
    }

    initialize(frame);
    drawOnFrame(frame);
    resize(frame, resizedFrame, Size(1980, 1020));
    imshow(windowName, resizedFrame);
    video >> frame;
    while(! frame.empty()){
     update(frame);
     drawOnFrame(frame);

     resize(frame, resizedFrame, Size(1980, 1020));
     imshow(windowName, resizedFrame);

     video >> frame;

    }

}


/*
 * Adds the Position of the Region and the current Frame to the tracked objects in the Region.
 */
void Region::updateObjectsInRegion(int frameNum) {
    for(auto trackedObject : playersInRegion){
        trackedObject->addPosition(coordinates, frameNum);
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

void FootballPlayer::addPosition(Rect coordinates, int frame) {
    this->coordinates.push_back(coordinates);
    this->frames.push_back(frame);
}

Region::Region(Rect coordinates) {
    this->coordinates = coordinates;
}


bool Region::regionsAssociated(const Region &r1, const Region &r2) {
    return ((r1.coordinates & r2.coordinates).area() > 0);

}

void textAboveRect(Mat frame, Rect rect, string text) {
    int x,y;
    x = rect.x;
    y = rect.y;

    y -= 5; // Shift the text above the rect
    putText(frame,text, Point(x,y), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0,0,255));

}
