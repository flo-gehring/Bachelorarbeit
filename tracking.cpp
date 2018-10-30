//
// Created by flo on 15.10.18.
//

#include "tracking.h"
#include <set>
#ifndef DEBUG
#define DEBUG
#endif
#define SHOW


// TODO: Change so that every Region only has one Player (<-> Distinction between Region and Player Reasonable? I guess yes!)

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
    for(auto it = detectedRects.begin(); it != detectedRects.end(); ++it){

        newPlayer = new FootballPlayer((*it), 1, to_string(objectCounter));

        footballPlayers.emplace_back(newPlayer);
        regionsNewFrame.emplace_back(Region(*it,  footballPlayers.back()));
        footballPlayers.back()->coordinates.push_back((*it));

        histFromRect(frame, (*it), footballPlayers.back()->hist);
        objectCounter++;

    }

    return objectCounter;
}

bool RegionTracker::update(Mat frame) {

    matCurrentFrame = frame;
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

    if(regionsNewFrame.empty()){
#ifdef DEBUG
      cout << "Not a single Region found!" << endl;
#endif
        return;
    }

    float intersectionThreshold = 0.2;

    for (int i = 0; i < 22 ; ++i){
        for (int r = 0; r<22; ++r){
            matrix[i][r] = 0;
        }
    }


    Region  oldRegion(Rect(0,0,0,0), NULL);
    Region  newRegion(Rect(0,0,0,0), NULL);

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
#ifdef UNDEF

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
#endif


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


/********************************************************************************************************
 *                                                                                                      *
 *                                                                                                      *
 *      Handling Functions                                                                              *
 *      ------------------                                                                              *
 *                                                                                                      *
 *      These Functions will handle the events (appearing, disappearing, merging, splitting             *
 *      and contuinutation of Regions) in the video.                                                    *
 *                                                                                                      *
 *                                                                                                      *
 ********************************************************************************************************/

/*
 * Check if the Region is completely new, or if its a region that reappeared.
 * Assumption: Only one Objects is in this Region
 * Fill Football Player with information
 */
void RegionTracker::handleAppearance(int regionIndex) {
    Region & newRegion = regionsNewFrame[regionIndex];
    for (auto region = outOfSightRegions.begin(); region != outOfSightRegions.end(); ++region){

        if(Region::regionsInRelativeProximity(* region, regionsNewFrame[regionIndex], 5)){ // TODO: Frame count

            // TODO: Perform Additional Checking and such. This is very basic and probably not useful
#ifdef DEBUG
            printf("Region %i appeared and was identified to be an old Region.\n", regionIndex);


            fprintf(debugData, "Region %i:  Appeared and was identified to be an old Region.\n It now contains Player %s. \n", regionIndex, region->playerInRegion->identifier.c_str());
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

    FootballPlayer * newPlayer = new FootballPlayer(newRegion.coordinates, currentFrame, to_string(objectCounter));
    footballPlayers.emplace_back(newPlayer);

    // Calculate Histogramm
    histFromRect(matCurrentFrame, newRegion.coordinates, footballPlayers.back()->hist);
#ifdef DEBUG
    fprintf(debugData, "Region %i: Appeared and is a new Region.  It now contains NEW Player %s. \n", regionIndex, to_string(objectCounter).c_str());
    printf("Region %i appeared and was identified to be new Region. \n", regionIndex);
#endif

    newRegion.playerInRegion = footballPlayers.back();
}

void RegionTracker::handleDisapearance(int regionIndex) {

    #ifdef DEBUG
    printf("Region %i disappears.\n", regionIndex);
    fprintf(debugData, "Region %i: \n Disappears. It did contain Player %s",
            regionIndex, regionLastFrame[regionIndex].playerInRegion->identifier.c_str());
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
    fprintf(debugData, "Splitting Region %i into %s. \n", regionIndex, s_char);
    #endif

    vector<Region *> splitRegions;
    for(int i = 0; i < num; ++i){
        splitRegions.push_back(&regionsNewFrame[splitInto[i]]);
    }



    int bestGuess;
    Mat histPlayer;

    Region & splittingRegion = regionLastFrame[regionIndex];

    histPlayer = splittingRegion.playerInRegion->hist;
    auto sortingFunction = [histPlayer, this](Region const * reg1, Region const * reg2)
        {        Mat hist1, hist2;
            histFromRect(matCurrentFrame, reg1->coordinates, hist1);
            histFromRect(matCurrentFrame, reg2->coordinates, hist2);
            return compareHist(histPlayer, hist1, CV_COMP_CORREL) < compareHist(histPlayer, hist2, CV_COMP_CORREL); };

    // TODO: Im assigning this Player in any case. This probably shouldnt be like this.
    std::sort(splitRegions.begin(), splitRegions.end(), sortingFunction);

    // TODO: When there is movement data etc, available, do i need to tranfer or should i simply save it in the Player?
    splitRegions.front()->playerInRegion = splittingRegion.playerInRegion;

#ifdef DEBUG
        fprintf(debugData, "Player %s from the old Region was assigned to a new Region\n", splittingRegion.playerInRegion->identifier.c_str());
#endif



    // If A region was not assigned a player from the Region which split up, search for a player which is currently
    // out of sight and would fit.
    // If there is no fitting Region, create a new Player.
    bool regionsIntersect = false;
    bool histogrammsSimmiliar = false;
    float histTreshold = 0.5;
    Mat histOfRegion, histOfPlayer;

    for (Region * region: splitRegions){
        if(region->playerInRegion == NULL){

            // Search out of sight regions
            for (auto  outofsight = outOfSightRegions.begin(); outofsight != outOfSightRegions.end(); ++outofsight){

                regionsIntersect = Region::regionsInRelativeProximity(* outofsight, *region, 7); // TODO: A way to get the frames that have passed.
                histOfPlayer = outofsight->playerInRegion->hist;
                histFromRect(matCurrentFrame, region->coordinates, histOfRegion);

                histogrammsSimmiliar = compareHist(histOfPlayer, histOfRegion, CV_COMP_CORREL) > histTreshold;

                if(regionsIntersect  && histogrammsSimmiliar){

                    region->playerInRegion =  outofsight->playerInRegion;
#ifdef DEBUG
                        fprintf(debugData, "Player %s which was out of sight was assigned to a Region emerging from the split \n", outofsight->playerInRegion->identifier.c_str());
#endif


                    outOfSightRegions.erase(outofsight);
                    break; // Break inner loop
                }
            }

            // if no such region was found, create a new player
            if (! regionsIntersect || ! histogrammsSimmiliar){
                ++objectCounter;
                FootballPlayer * newPlayer = new FootballPlayer(region->coordinates,currentFrame, string(to_string(objectCounter)));
                footballPlayers.emplace_back(newPlayer);
                histFromRect(matCurrentFrame, region->coordinates, footballPlayers.back()->hist);
                region->playerInRegion = newPlayer;
#ifdef DEBUG
                fprintf(debugData, "New Player %s created! \n", to_string(objectCounter).c_str());
#endif
            }

        }
    }
}

/*
 * Handle the "merging" of two Regions.
 * The new Region will only have one player in it, the rest will be marked as out of sight.
 */
void RegionTracker::handleMerging(int *regions, int num, int mergeInto) {

#ifdef DEBUG
    string s = "";
    for (int i= 0; i < num; ++i) s += " " + to_string(regions[i]) +",";
    const char * s_char = s.c_str();
    printf("Merging Regions%s into %i. \n", s_char, mergeInto);
    fprintf(debugData, "Region%i, Merging Regions%s into %i. \n",mergeInto, s_char, mergeInto);
#endif

    Region * newRegion = &regionsNewFrame[mergeInto];

    vector<Region *> oldRegions;
    for(int i = 0; i < num; ++i) {
        oldRegions.push_back(&regionLastFrame[*(regions + i)]);
    }

    vector<FootballPlayer *> playersInOldRegion;

    for(auto r = oldRegions.begin(); r != oldRegions.end(); ++r ){
            playersInOldRegion.push_back( (*r)->playerInRegion);


    }
    Mat histOfRegion;
    histFromRect(matCurrentFrame, newRegion->coordinates, histOfRegion);


    // Get the Best Guess as ID for the new region
    auto sortingFunction = [histOfRegion](FootballPlayer * f1, FootballPlayer * f2){

        return compareHist(histOfRegion, f1->hist, CV_COMP_CORREL) < compareHist(histOfRegion, f2->hist, CV_COMP_CORREL);
    };

    sort(playersInOldRegion.begin(), playersInOldRegion.end(), sortingFunction);
    newRegion->playerInRegion = playersInOldRegion[0];

#ifdef DEBUG
    fprintf(debugData, "Put %s into new Region, Rest out of sight.\n", playersInOldRegion[0]->identifier.c_str());
#endif

    // Mark all the other Players and Regions as out of sight.
    bool markAsLost;
    int oldRegionIndex;
    Region  & oldRegion = regionLastFrame[0];
    for(int i = 0; i < num; ++i){
        oldRegionIndex = regions[i];
        oldRegion = regionLastFrame[oldRegionIndex];
        if(oldRegion.playerInRegion != newRegion->playerInRegion){ // Mark as out of sight
            outOfSightRegions.emplace_back(regionLastFrame[oldRegionIndex]);
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
    fprintf(debugData, "Region %i. Continue as Region %i with Player %s. \n", regionIndexOld, regionIndexNew, oldRegion.playerInRegion->identifier.c_str());
#endif

    newRegion.playerInRegion = oldRegion.playerInRegion;

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

    #ifdef DEBUG
    int numSkipFrames = 0;
    for(int fc = 0; fc < numSkipFrames; ++fc){
        video >> frame;
    }
#endif


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
#ifdef DEBUG
    fprintf(debugData, "----------------------\n");
    fprintf(debugData, "Frame %i: \n", frameCounter+1);
#endif
     update(frame);

     int counter = 0;
        cout << "Updated Frame " << frameCounter << endl;
        ++frameCounter;

        secondsPassed = (time(0) - timeStart);

        if(secondsPassed != 0){
            avgFrameRate = float(frameCounter) / secondsPassed;
        }

        putText(frame,"FPS: " + to_string(avgFrameRate), Point(0,50), FONT_HERSHEY_PLAIN, 4, Scalar(0,0,255), 2);

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
         pauseVid = c == 32;
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


FootballPlayer::FootballPlayer(Rect coordinate, int frame, string identifier) {

    // coordinates.emplace_back(coordinate);
    frames.emplace_back(frame);
    this->identifier = identifier;

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
void Region::updateObjectsInRegion(int frameNum) {
    // TODO    for(auto id = this->playerIds.begin(); id != playerIds.end(); ++id){
    //  }
}


Region::Region(const Rect &coordinates, FootballPlayer *  ptrPlayer) {
    this->coordinates = coordinates;
    playerInRegion = ptrPlayer;
}

Region::Region(Rect coordinates) {
    this->coordinates = coordinates;
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