//
// Created by flo on 15.10.18.
//

#include "tracking.h"
#include <set>
#ifndef DEBUG
#define DEBUG
#endif
void printMatrix(unsigned  short mat[][22]){
    int rowsToPrint = 13;

    cout << endl << " ";
    for(int i = 0; i < rowsToPrint; ++i) cout << i;
    cout << endl;
    for(int row = 0; row < rowsToPrint; ++row){
        cout << row;
        for(int col = 0; col < rowsToPrint; ++col){

            if(mat[row][col] != 0){
                cout << "1";
            }
            else{
                cout << " ";
            }

        }
        cout << endl;
    }

}
int RegionTracker::initialize(Mat frame) {
    currentFrame = 0;
    vector<Rect> detectedRects;

    // All the detected Objects will be stored as recangles in detectedRects
    detectOnFrame(frame, detectedRects);

    Rect panoramaCoords;

    objectCounter = 0;

    for(auto it = detectedRects.begin(); it != detectedRects.end(); ++it){

        footballPlayers.push_back(FootballPlayer((*it), 1, to_string(objectCounter) ));
        regionsNewFrame.emplace_back(Region(*it, footballPlayers.back().identifier));
        footballPlayers.back().coordinates.push_back((*it));
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

    for(auto it1 = regionsNewFrame.begin(); it1 != regionsNewFrame.end(); ++it1){
        for(auto it2 = regionsNewFrame.begin(); it2 != regionsNewFrame.end(); ++it2){
            if((it1 != it2) && (it1->coordinates & it2->coordinates).area() > 0)
            {cerr << "Two Regions cross: "<< endl;
            printf("Reg1(%i, %i), Reg2(%i, %i)", it1->coordinates.x, it1->coordinates.y, it2->coordinates.x, it2->coordinates.y);
            }
        }
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

    float intersectionThreshold = 0.2;

    for (int i = 0; i < 22 ; ++i){
        for (int r = 0; r<22; ++r){
            matrix[i][r] = 0;
        }
    }


    Region  oldRegion(Rect(0,0,0,0), "-1");
    Region  newRegion(Rect(0,0,0,0), "-1");

    vector<int> intersections;

    double sharedArea, areaNewRegion;

    // vector<int> associationUnclear;

    vector<int> regionsNewFrameIndices(regionsNewFrame.size());
    std::iota(regionsNewFrameIndices.begin(), regionsNewFrameIndices.end(), 0);

    // Get Associations between regions by amount of overlapping
    // Assign Candiate which matches the most.
    for(int oldRegionIndex = 0; oldRegionIndex < regionLastFrame.size(); ++oldRegionIndex){

        // Sort by how much the Regions intersect with the region from the last frame associated with the current
        // index.
        auto sortingFunction =  [oldRegionIndex, this] (int  index1, int index2 )
        { return
                (regionsNewFrame[index1].coordinates & regionLastFrame[oldRegionIndex].coordinates).area() <
                (regionsNewFrame[index2].coordinates & regionLastFrame[oldRegionIndex].coordinates).area();};

        std::sort(regionsNewFrameIndices.begin(), regionsNewFrameIndices.end(), sortingFunction);
        int bestMatch = regionsNewFrameIndices.back();

        sharedArea = (regionLastFrame[oldRegionIndex].coordinates & regionsNewFrame[bestMatch].coordinates).area();
        if( sharedArea / regionsNewFrame[bestMatch].coordinates.area() > areaThreshold){
            matrix[oldRegionIndex][bestMatch] = 1;
        }

    }

    // If there is a new Region left which is not associated with any old Region and there is an old Region with which
    // it overlaps greatly, thats a case for splitting. Iterate matrix

    int associationCounter = 0;


    for(int col = 0; col < regionsNewFrame.size(); ++col){
        associationCounter = 0;
        newRegion = regionsNewFrame[col];
        areaNewRegion = newRegion.coordinates.area();
        for(int row = 0; row < regionLastFrame.size(); ++row){
            oldRegion = regionLastFrame[row];

            sharedArea = (newRegion.coordinates & oldRegion.coordinates).area();
            if( (sharedArea/  areaNewRegion) > areaThreshold && matrix[row][col] != 1){ // matrix[row][col] != 1 is unnecessary but when debugging it only stops in relevant cases
#ifdef DEBUG
            if(matrix[row][col] != 1)    printf("New Region %i will split old Region %i", col, row );
#endif
                matrix[row][col] = 1;

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
    printMatrix(matrix);
    // Iterate Row wise to get information about

    // remember the new Regions which came from a split, so you dont confuse it with
    // a continuation later on.
    set<int> regionsFromSplit;

    int row = 0;

    /*Iterate every row.
     * We now know, if the old Region which is represented by this row, either disappears or splits into multiple new.
     */
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
        else if(associationCounter > 1) { // Split
            handleSplitting(row, &associatedIndexes[0], associationCounter);
            regionsFromSplit.insert(associatedIndexes.begin(), associatedIndexes.end());
        }
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
        else if(associationCounter == 1 && regionsFromSplit.find(col) == regionsFromSplit.end()){ // Either Continuity or split

            handleContinuation(associatedIndexes[0], col);
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
    Region & newRegion = regionsNewFrame[regionIndex];
    for (auto region = outOfSightRegions.begin(); region != outOfSightRegions.end(); ++region){

        if(Region::regionsIntersect(* region, regionsNewFrame[regionIndex])){
            // TODO: Perform Additional Checking and such. This is very basic and probably not useful
#ifdef DEBUG
            printf("Region %i appeared and was identified to be an old Region.\n", regionIndex);
#endif
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
#ifdef DEBUG
    printf("Region %i appeared and was identified to be new Region. \n", regionIndex);
#endif
    newRegion.playerIds.emplace_back(string(footballPlayers.back().identifier));
}

void RegionTracker::handleDisapearance(int regionIndex) {
#ifdef DEBUG
    printf("Region %i disappears.\n", regionIndex);
#endif

    outOfSightRegions.push_back(regionLastFrame[regionIndex]);

}

/*
 * If a region splits into multiple other Regions, try to determine where each player in the old region went.
 * At first, we'll simply compare histogramms.
 */
void RegionTracker::handleSplitting(int regionIndex, int *splitInto, int num) {
#ifdef DEBUG
    string s;
    for (int i= 0; i < num; ++i) s += " " + to_string(splitInto[i]) +",";
    const char * s_char = s.c_str();
    printf("Splitting Region %i into %s. \n", regionIndex, s_char);
#endif

    // TODO Just add new Football Players for the beginning
    for(int i = 0; i < num; ++i){
        Region & r = regionsNewFrame.at(splitInto[i]);
        ++objectCounter;
        footballPlayers.emplace_back(FootballPlayer(r.coordinates, currentFrame, to_string(objectCounter)));
        r.playerIds.push_back(to_string(objectCounter));
    }

}

/*
 * Throw stuff together in one region.
 */
void RegionTracker::handleMerging(int *regions, int num, int mergeInto) {

#ifdef DEBUG
    string s = "";
    for (int i= 0; i < num; ++i) s += " " + to_string(regions[i]) +",";
    const char * s_char = s.c_str();
    printf("Merging Regions%s into %i. \n", s_char, mergeInto);
#endif

    Region * newRegion = &regionsNewFrame[mergeInto];
    vector<Region *> oldRegions;
    for(int i = 0; i < num; ++i) {
        oldRegions.push_back(&regionLastFrame[*(regions + i)]);
    }

    // Insert all the tracked Objects into the new Region
    for(auto r = oldRegions.begin(); r != oldRegions.end(); ++r ){
        for (auto n = (*r)->playerIds.begin(); n != (*r)->playerIds.end(); ++n){
            newRegion->playerIds.emplace_back(string(*n));
        }

    }

}

/*
 * Supply the new region with the Players of the old region.
 */
void RegionTracker::handleContinuation(int regionIndexOld, int regionIndexNew) {
    Region & oldRegion = regionLastFrame[regionIndexOld];
    Region & newRegion = regionsNewFrame[regionIndexNew];
    /*
    for(auto trackedObject = oldRegion.playersInRegion.begin(); trackedObject != oldRegion.playersInRegion.end();++trackedObject ){
        newRegion.playersInRegion.push_back(* trackedObject);
    }
     */
#ifdef DEBUG
    printf("Continue Region %i to %i.\n", regionIndexOld, regionIndexNew);
#endif
    newRegion.playerIds.emplace_back(string(oldRegion.playerIds.back()));
    //newRegion.updateObjectsInRegion(currentFrame);

}

void RegionTracker::detectOnFrame(Mat frame, vector<Rect> &detected) {
    int sideLength = 500;
    Mat face;
    Rect panoramaCoords;

    vector<float> scores;
    vector<int> indices;
    vector<Rect> tmpDetected;

    for(int faceId = 0; faceId < 6; ++faceId){
        createCubeMapFace(frame, face, faceId, sideLength, sideLength);
        darknetDetector.detect_and_display(face);

        for(auto detectedObjects = darknetDetector.found.begin();
        detectedObjects != darknetDetector.found.end(); ++detectedObjects){

            mapRectangleToPanorama(frame, faceId, sideLength, sideLength, (* detectedObjects).rect, panoramaCoords);

            tmpDetected.push_back(panoramaCoords);
            scores.push_back((*detectedObjects).prob);

        }

    }

    // Perfrorm NMS on detected Recangles and copy accepted rects to detected
    dnn::NMSBoxes(tmpDetected, scores, 0.4f, 0.2f, indices);
    for(int i : indices){
        detected.emplace_back(Rect(tmpDetected[i]));
    }

}

void RegionTracker::drawOnFrame(Mat frame) {
    for (Region & r: regionsNewFrame){
        string id;
        for(string player : r.playerIds) id += " " + player;

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
    waitKey(30);
    video >> frame;
    while(! frame.empty()){
     update(frame);

     int counter = 0;
     for(Region region : regionsNewFrame){

         CV_Assert(region.playerIds.size() > 0);
         ++counter;
     }

     drawOnFrame(frame);

     resize(frame, resizedFrame, Size(1980, 1020));
     imshow(windowName, resizedFrame);
     waitKey(30);

     video >> frame;

    }

}


/*
 * Adds the Position of the Region and the current Frame to the tracked objects in the Region.
 */
void Region::updateObjectsInRegion(int frameNum) {
 // TODO    for(auto id = this->playerIds.begin(); id != playerIds.end(); ++id){
  //  }
}


Region::Region(const Rect &coordinates, string playerID) {
    this->coordinates = coordinates;
    this->playerIds.push_back(playerID);
}

FootballPlayer::FootballPlayer(Rect coordinate, int frame, string identifier) {

    // coordinates.emplace_back(coordinate);
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


bool Region::regionsIntersect(const Region &r1, const Region &r2){
    return ((r1.coordinates & r2.coordinates).area() > 0);

}

void textAboveRect(Mat frame, Rect rect, string text) {
    int x,y;
    x = rect.x;
    y = rect.y;

    y -= 5; // Shift the text above the rect
    putText(frame,text, Point(x,y), FONT_HERSHEY_PLAIN, 3, Scalar(0,0,255), 2);

}
