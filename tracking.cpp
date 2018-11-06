//
// Created by flo on 15.10.18.
//

#include "tracking.h"
#include <set>

#define SHOW



// TODO: Save Information about every Player in their Objects

// TODO: Velocity information for better prediction

// TODO: Save Information about Regions and so on in File

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

    roiData = fopen("roidata.txt", "w");
    debugData = fopen("debugdata.txt", "w");

    currentFrame = 0;
    matCurrentFrame = frame;
    vector<Rect> detectedRects;

    // All the detected Objects will be stored as recangles in detectedRects
    detectOnFrame(frame, detectedRects);

    objectCounter = 0;

    FootballPlayer * newPlayer;
    darknetDetector = DetectionFromFile();


    for(auto it = detectedRects.begin(); it != detectedRects.end(); ++it){

        newPlayer = new FootballPlayer((*it), 1, to_string(objectCounter));

        footballPlayers.emplace_back(newPlayer);
        regionsNewFrame.emplace_back(Region(*it,  footballPlayers.back()));
        footballPlayers.back()->coordinates.push_back((*it));

        histFromRect(frame, (*it), newPlayer->hist);
        objectCounter++;

    }

    return objectCounter;
}

bool RegionTracker::update(Mat frame) {

    matCurrentFrame = Mat(frame);
    ++currentFrame;
    // Update Region Vectors, new regions from last frame are now old, get new Regions from detector
    regionLastFrame.swap(regionsNewFrame);
    regionsNewFrame.clear();

    vector<Rect> newRects;
    detectOnFrame(frame, newRects);

    for(auto it = newRects.begin(); it != newRects.end(); ++it){

        regionsNewFrame.emplace_back(Region(*it));
    }


    vector<MetaRegion> vectorMetaRegion = calcMetaRegions();
    interpretMetaRegions(vectorMetaRegion);

    /*
    calcMatrix();

    interpretMatrix();
     */

    return false;
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
        id = r.playerInRegion->identifier;

        textAboveRect(frame, r.coordinates, id);
        rectangle(frame, r.coordinates, Scalar(0, 0, 255), 2);

    }
}

void RegionTracker::workOnFile(char *filename) {

    VideoCapture video(filename);
    Mat frame, resizedFrame;

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
#ifdef SHOW
    const char * windowName = "Occlusion Tracker";
    namedWindow(windowName);
    resize(frame, resizedFrame, Size(1980, 1020));
    imshow(windowName, resizedFrame);
#else
    VideoWriter vw("darknetTracker.mp4", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                   video.get(CAP_PROP_FPS),
                   frame.size(), true);
#endif

    drawOnFrame(frame);

    time_t timeStart = time(0);
    float avgFrameRate = 0;
    float secondsPassed;
    int frameCounter = 0;

    video >> frame;


    while(! frame.empty()){
        waitKey(30);
        #ifdef DEBUG
        fprintf(debugData, "----------------------\n");
        fprintf(debugData, "Frame %i: \n", frameCounter+1);
        printMatrix(matrix);
        #endif
        cout << "Frame No.: "<< frameCounter << endl;
        matCurrentFrame = frame;

        update(frame);

        /* vector<MetaRegion> mr = calcMetaRegions();
        for(MetaRegion const & metaRegion : mr){
         rectangle(frame, metaRegion.area, Scalar(255, 0, 0), 2);
        }
         */

         int counter = 0;
        #ifdef DEBUG
        cout << "Updated Frame " << frameCounter << endl;
        #endif
         ++frameCounter;
         secondsPassed = (time(0) - timeStart);

        if(secondsPassed != 0){
            avgFrameRate = float(frameCounter) / secondsPassed;
        }

        putText(frame,"FPS: " + to_string(int(avgFrameRate)), Point(0,50), FONT_HERSHEY_PLAIN, 4, Scalar(0,0,255), 2);
        putText(frame, "Tracker: Darknet" , Point(0, 100), FONT_HERSHEY_PLAIN, 4, Scalar(0,0,255), 2);
        putText(frame, "Frame No." + to_string(frameCounter), Point(0, 150), FONT_HERSHEY_PLAIN, 4, Scalar(0,0,255), 2);

        for(Region region : regionsNewFrame){

             CV_Assert(region.playerInRegion->identifier.size() > 0);
             ++counter;
        }

         drawOnFrame(frame);


    #ifdef SHOW
         resize(frame, resizedFrame, Size(1980, 1020));
         imshow(windowName, resizedFrame);
         char c = waitKey(30);
         bool pauseVid = c == 32; // Space
          while(pauseVid){
             c = waitKey(10);
             if (c == 32) break;
         }
    #else
         vw.write(frame);
    #endif

         video >> frame;

        }

}

/********************************************
 *                                          *
 *      Football Player Class Methods       *
 *      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       *
 ********************************************/

FootballPlayer RegionTracker::playerById(string id) {
    for(FootballPlayer * player: footballPlayers){
        if (id == player->identifier){
            return *player;
        }
    }
    return FootballPlayer(Rect(), 0, "");
}

RegionTracker::~RegionTracker() {
    for(FootballPlayer * footballPlayer : footballPlayers){
        delete footballPlayer;
    }
}


/*
 * Helper Function:
 * Iterates Vector and puts matching Regions into the Meta Region.
 * Returns if the area has changed (For now this is iff a region was added).
 */
bool helperRegionsInMeta(MetaRegion & mr, vector<Region> & vectorOfRegions, unordered_set<Region *> & regionFound){

    bool areaUnchanged = true;
    Rect commonArea;
    bool regionInMeta = false;

    auto search = [](MetaRegion & mr, Region & r){

        for(Region * r_ptr: mr.metaOldRegions) if(r_ptr == &r) return true;
        for(Region * r_ptr: mr.metaNewRegions) if(r_ptr == &r) return true;
        return false;
    };
    for(Region & region : vectorOfRegions){
        commonArea = region.coordinates & mr.area;

        regionInMeta = search(mr, region);

        if(Region::regionsInRelativeProximity(region, Region(mr.area), 10) && ! regionInMeta ){ // TODO Num of frames passed is magic literal

            areaUnchanged = false;
            mr.area |= region.coordinates;
            mr.metaOldRegions.push_back(&region);

            regionFound.insert(&region);

        }
    }
    return areaUnchanged;
}


/*
 * Calcs the Meta Regions.
 * TODO: Put all the Regions from the old frame which don't find a new MetaRegion to be into into outOfSight
 */
vector<MetaRegion> RegionTracker::calcMetaRegions() {

    vector<int> indicesUnhandledRegions(regionsNewFrame.size());
    std::iota(std::begin(indicesUnhandledRegions),std::end(indicesUnhandledRegions), 0);

    Region * currentRegion;
    Region * regionToCompare;
    bool areaUnchanged = false;

    vector<MetaRegion> metaRegions;

    // A set in which every Region from the last frame is, which found an associated MetaRegion
    unordered_set<Region *> associatedMRFound;
    associatedMRFound.reserve(regionLastFrame.size());
    unordered_set<Region *> outOfSightFound; // Theres no need for this, but the function needs the argument

    while(! indicesUnhandledRegions.empty()){

        // Create a new Meta Region, with an arbitrary available Region from the new Frame
        currentRegion = &regionsNewFrame[indicesUnhandledRegions.back()];

        indicesUnhandledRegions.pop_back();

        metaRegions.emplace_back(MetaRegion());

        MetaRegion & currentMetaRegion = metaRegions.back();
        currentMetaRegion.area = Rect(currentRegion->coordinates);
        currentMetaRegion.metaNewRegions.push_back(currentRegion);

        areaUnchanged = false;

        // Iterate through old, new and out of sight regions to get the Meta Region
        while(!areaUnchanged){

            areaUnchanged = true;

            for(auto newRegionIndex = indicesUnhandledRegions.begin();
            newRegionIndex != indicesUnhandledRegions.end(); /*Do nothing */){

                regionToCompare = &regionsNewFrame[*newRegionIndex];

                // A new Region from the current Frame for the Meta Region
                if(Region::regionsInRelativeProximity(*regionToCompare, Region(currentMetaRegion.area), 10)){ // TODO Num of frames passed is magic literal
                //if((regionToCompare->coordinates & currentMetaRegion.area).area() > 0){

                    areaUnchanged = false;

                    currentMetaRegion.area |= currentRegion->coordinates;
                    currentMetaRegion.metaNewRegions.push_back(& regionsNewFrame[* newRegionIndex]);

                    indicesUnhandledRegions.erase(newRegionIndex);

                }
                else{ // Only increase iterator if no Element was deleted.
                    ++newRegionIndex;
                }
            }

            areaUnchanged &= helperRegionsInMeta(currentMetaRegion, regionLastFrame, associatedMRFound);
            areaUnchanged &= helperRegionsInMeta(currentMetaRegion, outOfSightRegions, outOfSightFound);

        }


    }

    for(Region & region:regionLastFrame){
        Region * r = & region;
        if(associatedMRFound.count(r) == 0)
            addToOutOfSight(r);
    }

    int tmp = 0;
    for(MetaRegion const & mr : metaRegions) {
        if(mr.metaNewRegions.size() != 1 || mr.metaOldRegions.size() != 1){
            cout << "MetaReg " << tmp << endl;
            cout << "size old: " << mr.metaOldRegions.size() << " size new: " << mr.metaNewRegions.size() << endl;
        }
        ++tmp;
    }
    return metaRegions;
}

/*
 * Calculate how similiar two Regions are to each other.
 * Things taken into Consideration: Size, Histogramm, Position.
 * The bigger the value, the more the likelyhood of them representing the same area.
 */
double calcWeightedSimiliarity(Region  * r1, Region *r2, Rect area, Mat frame){
    double similiaritySize, similiarityPosition, similiarityHistogramm;

    // Size
    similiaritySize = float(r1->coordinates.area()) / float(r2->coordinates.area());
    if(similiaritySize > 1) similiaritySize = 1 / similiaritySize;

    // Position
    // Compare the Position of the upper left corner, hypotenuseMetaRegion is the maximum possible distance
    // and would result in a value of zero.
    double hypotenuseMetaRegion = sqrt(pow(area.width , 2) + pow(area.height, 2));
    double lengthVector = sqrt(pow(r1->coordinates.x - r2->coordinates.x, 2) + pow(r1->coordinates.y - r2->coordinates.y, 2));
    similiarityPosition = (hypotenuseMetaRegion - lengthVector) / hypotenuseMetaRegion;

    //Histogramm
    // http://answers.opencv.org/question/8154/question-about-histogram-comparison-return-value/
    // Return Value of CV_COMP_CORRELL: -1 is worst, 1 is best. -> Map to  [0,1]
    Mat hist1, hist2;
    histFromRect(frame, r1->coordinates, hist1);
    histFromRect(frame, r2->coordinates, hist2);

    similiarityHistogramm = compareHist(hist1, hist2, CV_COMP_CORREL);
    similiarityHistogramm = (similiarityHistogramm / 2) + 0.5; // Map [-1,1] to [0,1]

    return similiaritySize + similiarityPosition + similiarityHistogramm;
}
bool regionsMatch(Region * r1, Region * r2){
    return false;
}

int *  MetaRegion::matchOldAndNewRegions(Mat frame, int * matching){

    unsigned long oldRegionSize = metaOldRegions.size();
    unsigned long newRegionSize = metaNewRegions.size();

    double matchingMatrix[oldRegionSize * newRegionSize];

    Region * currentOldRegion;

    // Calculate the weighted similarities for each of the Regions in the Meta Region
    for(int rowCounter = 0; rowCounter < oldRegionSize; ++rowCounter){
        currentOldRegion = metaOldRegions[rowCounter];
        for(int colCounter = 0; colCounter < metaNewRegions.size(); ++colCounter){

            matchingMatrix[(rowCounter * newRegionSize) + colCounter] =
                    calcWeightedSimiliarity(currentOldRegion, metaNewRegions[colCounter], area, frame);
        }
    }

    // Create a matrix which shows how the Regions correspond to each other.
    int bestMatchingOld;
    double weightedMax, currentWeight;

    for(int colCounter = 0; colCounter < metaNewRegions.size(); ++colCounter){
        weightedMax= -1;
        bestMatchingOld = -1;
        for(int rowCounter = 0; rowCounter < oldRegionSize; ++rowCounter){
            currentWeight = matchingMatrix[(rowCounter * newRegionSize) + colCounter];


            if(weightedMax < currentWeight){
                // Search if there is a new region which fits the current old Region better.
                bool bestOption = true;

                for(int colCounter_1 = 0; colCounter_1 < metaNewRegions.size(); ++colCounter_1){
                    assert(((rowCounter * newRegionSize) + colCounter_1) < metaNewRegions.size() * oldRegionSize);
                    if (matchingMatrix[(rowCounter * newRegionSize) + colCounter_1] > currentWeight){
                        bestOption = false;
                        break;
                    }
                }
                if(bestOption) {
                    weightedMax = currentWeight;
                    bestMatchingOld = rowCounter;
                }
            }
        }
        if(bestMatchingOld != -1) {
            (*(matching + (bestMatchingOld * newRegionSize) + colCounter)) = 1;
        }
    }
    return matching;
}

/*
 * Assign a Player to every Region in the Meta Region, mark the rest as out of sight.
 */
void RegionTracker::interpretMetaRegions(vector<MetaRegion> & mr) {


    for(MetaRegion& metaRegion: mr){

        if(metaRegion.metaNewRegions.size() == 1 && metaRegion.metaOldRegions.size() == 1){
            metaRegion.metaNewRegions[0]->playerInRegion = metaRegion.metaOldRegions[0]->playerInRegion;
        }
        else if(metaRegion.metaNewRegions.empty()){
            for(Region * region: metaRegion.metaOldRegions){
                addToOutOfSight(region);
            }
        }
        else if(metaRegion.metaOldRegions.empty()){
            for(Region * region: metaRegion.metaNewRegions){
                ++objectCounter;
                FootballPlayer * fp = new FootballPlayer(Rect(region->coordinates), currentFrame,
                                                         string(to_string(objectCounter)));
                footballPlayers.push_back(fp);
                region->playerInRegion = fp;
                cout << "Create new Player " << to_string(objectCounter) << endl;
            }
        }

        else{
            int * matching =  new int[metaRegion.metaNewRegions.size() * metaRegion.metaOldRegions.size()];
            //int * matching_asdf = matching;
            metaRegion.matchOldAndNewRegions(matCurrentFrame, matching);

            // Print matching
            for(int rowCounter = 0; rowCounter < metaRegion.metaOldRegions.size(); ++rowCounter){
                for(int colCounter = 0; colCounter < metaRegion.metaNewRegions.size(); ++colCounter){
                    if(matching[(rowCounter * metaRegion.metaNewRegions.size()) + colCounter] == 1)
                        cout << 1;
                    else cout << 0;

                }
                cout << endl;
            }

            bool newRegionMatched;
            for(int rowCounter = 0; rowCounter < metaRegion.metaOldRegions.size(); ++rowCounter){
                newRegionMatched = false;
                for(int colCounter = 0; colCounter < metaRegion.metaNewRegions.size(); ++colCounter){
                    if(matching[(rowCounter * metaRegion.metaNewRegions.size()) + colCounter] == 1){
                        newRegionMatched = true;
                        metaRegion.metaNewRegions[colCounter]->playerInRegion = metaRegion.metaOldRegions[rowCounter]->playerInRegion;

                        // If the oldRegion was part of outOfSight Regions, delete it from there
                        deleteFromOutOfSight(metaRegion.metaOldRegions[rowCounter]->playerInRegion);
                        break;
                    }

                }
                if(!newRegionMatched){ // Add to out of sight Regions, if it's not already there
                    addToOutOfSight(metaRegion.metaOldRegions[rowCounter]);
                    cout << "Push out of sight region" << endl;
                }
            }
            delete[] matching;
        }
        // If no matching old Region was found for a new Region, create a new Player.
        for(Region * newRegion: metaRegion.metaNewRegions){
            if(newRegion->playerInRegion == nullptr){
                ++objectCounter;
                FootballPlayer * fp = new FootballPlayer(newRegion->coordinates, currentFrame,
                        string(to_string(objectCounter)));
                footballPlayers.push_back(fp);
                newRegion->playerInRegion = fp;
                cout << "Create new Player " << to_string(objectCounter) << endl;


            }
        }

    }
    for(MetaRegion const & metaRegion1: mr ){

        rectangle(matCurrentFrame, metaRegion1.area, Scalar(255,0,0), 1 );
    }

}

/*
 * If the given FootballPlayer is in outOfSight Regions, the corresponding Region will be deleted.
 */
void RegionTracker::deleteFromOutOfSight(FootballPlayer * searchFor) {
    auto iterator = outOfSightRegions.begin();
    for(Region & r : outOfSightRegions){
        if(r.playerInRegion == searchFor)
            outOfSightRegions.erase(iterator);
        ++iterator;
    }
}

void RegionTracker::addToOutOfSight(Region * regionPtr) {

    for(Region const & region: outOfSightRegions) {
        if (region.playerInRegion->identifier == regionPtr->playerInRegion->identifier) return;
    }
    outOfSightRegions.emplace_back(*regionPtr);
}

FootballPlayer::FootballPlayer(Rect coordinate, int frame, string identifier) {

    // coordinates.emplace_back(coordinate);
    frames.emplace_back(frame);
    this->identifier = identifier;
    hist = Mat();

}

void FootballPlayer::addPosition(Rect coordinates, int frame) {
    this->coordinates.push_back(coordinates);
    this->frames.push_back(frame);
}

/********************************************
 *                                          *
 *          Region Class Methods            *
 *          ^^^^^^^^^^^^^^^^^^^^            *
 ********************************************/

/*
 * Adds the Position of the Region and the current Frame to the tracked objects in the Region.
 */
void Region::updateObjectsInRegion(const Region * oldRegion, int frameNum) {
    // TODO    for(auto id = this->playerIds.begin(); id != playerIds.end(); ++id){
    //  }
}


Region::Region(const Rect &coordinates, FootballPlayer *  ptrPlayer) {
    this->coordinates = coordinates;
    playerInRegion = ptrPlayer;
}

Region::Region(Rect coordinates) {
    this->coordinates = coordinates;
    playerInRegion = nullptr;
}


bool Region::regionsIntersect(const Region &r1, const Region &r2){
    return ((r1.coordinates & r2.coordinates).area() > 0);

}

void estimateLocalization(Region const & r1, int framesPassed, Rect & r){
    int pixelPerFrame = 3;
    Rect rect1 = Rect(r1.coordinates);
    int x1, y1,width1 ,height1;
    x1 = rect1.x - (framesPassed * pixelPerFrame);
    y1 = rect1.y - (framesPassed * pixelPerFrame);
    width1 = rect1.width + (pixelPerFrame * framesPassed);
    height1 = rect1.height + (pixelPerFrame * framesPassed);

    r.x = x1;
    r.y = y1;
    r.width = width1;
    r.height = height1;
}

bool Region::regionsInRelativeProximity(Region const &r1, Region const &r2, int framesPassed) {


    Rect estimate1, estimate2;
    estimateLocalization(r1, framesPassed, estimate1);
    estimateLocalization(r2, framesPassed, estimate2);

    return (estimate1 & estimate2).area() > 0;
}

Region::Region(Region const &r1) {
    coordinates = Rect(r1.coordinates);
    playerInRegion = r1.playerInRegion;
}

void textAboveRect(Mat frame, Rect rect, string text) {
    int x,y;
    x = rect.x;
    y = rect.y;

    y -= 5; // Shift the text above the rect
    x -= 5;
    putText(frame,text, Point(x,y), FONT_HERSHEY_PLAIN, 3, Scalar(0,0,255), 2);

}

void histFromRect(Mat const &input, Rect const &rect, Mat &output) {
    Mat hsv;
    if(input.empty())
        exit(-55);
    cvtColor(input(rect), hsv, CV_BGR2HSV);
    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };

    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};


    calcHist( &hsv, 1, channels, Mat(), // do not use mask
              output, 2, histSize, ranges,
              true, // the histogram is uniform
              false );

}