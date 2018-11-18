//
// Created by flo on 15.10.18.
//

#include "tracking.h"
#include <set>




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

/*
 * Initializes the Tracker.
 * Performs the first round of detection which differs slightly from the following.
 */
int RegionTracker::initialize(Mat frame) {

    roiData = fopen("roidata.txt", "w");
    debugData = fopen("debugdata.txt", "w");

    currentFrame = 0;
    matCurrentFrame = frame;

    // All the detected Objects will be stored as rectangles in detectedRects
    vector<Rect> detectedRects = detectOnFrame(frame);

    objectCounter = 0;

    FootballPlayer * newPlayer;
    darknetDetector = DetectionFromFile();


    for(auto it = detectedRects.begin(); it != detectedRects.end(); ++it){

        CV_Assert(it->area() != 0);

        newPlayer = new FootballPlayer((*it), 1, to_string(objectCounter));

        footballPlayers.emplace_back(newPlayer);
        regionsNewFrame.emplace_back(Region(*it,  footballPlayers.back()));
        footballPlayers.back()->coordinates.push_back((*it));

        histFromRect(frame, (*it), newPlayer->hist);
        objectCounter++;

    }
    for(Region & r: regionsNewFrame){
        r.createColorProfile(frame);
    }

    return objectCounter;
}

/*
 * Everytime you grab a new Frame from your Videostream, pass it to this Method and
 */
bool RegionTracker::update(Mat frame) {

    matCurrentFrame = Mat(frame);
    ++currentFrame;
    // Update Region Vectors, new regions from last frame are now old, get new Regions from detector
    regionLastFrame.swap(regionsNewFrame);
    regionsNewFrame.clear();

    vector<Rect> newRects = detectOnFrame(frame);

    for(auto it = newRects.begin(); it != newRects.end(); ++it){

        regionsNewFrame.emplace_back(Region(*it));
        assert((*it).area() != 0);
    }

    for(Region  & newRegion: regionsNewFrame) newRegion.createColorProfile(matCurrentFrame);

    vector<MetaRegion> vectorMetaRegion = calcMetaRegions();
    interpretMetaRegions(vectorMetaRegion);

    for(Region & newRegion: regionsNewFrame) newRegion.updatePlayerInRegion(currentFrame);

    /*
    calcMatrix();

    interpretMatrix();
     */

    return false;
}

/*
 * Projects the frame, which is an equirectangular Panorama, onto the six sides of a cube and performs object
 * Detection on them.
 * The coordinates of the detected Bounding Boxes are projected back onto the original frame.
 */
vector<Rect> RegionTracker::detectOnFrame(Mat  & frame) {
    vector<Rect> detected;
    int sideLength = 500;
    Mat face;
    Rect panoramaCoords;

    vector<float> scores;
    vector<int> indices;
    vector<Rect> tmpDetected;

    for(int faceId = 0; faceId < 6; ++faceId){
        createCubeMapFace(frame, face, faceId, sideLength, sideLength);
        darknetDetector.detect_and_display(face);

        // We usually dont need top and bottom, this also increases performance massively,
        // because the bb of the cameraman was rather large, which slowed down the k-mean Algorithm.
        if( faceId != 5 && faceId != 6) {
            for (auto detectedObjects = darknetDetector.found.begin();
                 detectedObjects != darknetDetector.found.end(); ++detectedObjects) {

                mapRectangleToPanorama(frame, faceId, sideLength, sideLength, (*detectedObjects).rect, panoramaCoords);

                tmpDetected.push_back(panoramaCoords);
                scores.push_back((*detectedObjects).prob);

            }
        }
    }
    // Perfrorm NMS on detected Recangles and copy accepted rects to detected
    dnn::NMSBoxes(tmpDetected, scores, 0.4f, 0.2f, indices);
    for(int i : indices){
        detected.emplace_back(Rect(tmpDetected[i]));
    }

    return detected;

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
    VideoWriter vw("darknetTracker_meta.mp4", VideoWriter::fourcc('M', 'J', 'P', 'G'),
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


bool helperOutOfSightInMeta(MetaRegion & mr, vector<Region *> & outOfSight, unordered_set<Region *> alreadyFound, int currentFrame){
    bool areaUnchanged = true;
    Rect commonArea;
    bool regionInMeta = false;

    auto search = [](MetaRegion & mr, Region * r){

        for(Region * r_ptr: mr.metaOldRegions) if(r_ptr == r) return true;
        for(Region * r_ptr: mr.metaNewRegions) if(r_ptr == r) return true;
        return false;
    };
    for(Region * region : outOfSight){
        commonArea = region->coordinates & mr.area;

        regionInMeta = search(mr, region);

        int framesDifference = currentFrame - region->playerInRegion->frames.back();

        if(Region::regionsInRelativeProximity(*region, Region(mr.area), framesDifference) && ! regionInMeta &&
        alreadyFound.find(region) != alreadyFound.end()){ // TODO Num of frames passed is magic literal

            areaUnchanged = false;
            mr.area |= region->coordinates;
            mr.metaOldRegions.push_back(region);
            alreadyFound.insert(region);

        }
    }
    return areaUnchanged;
}

/*
 * Helper Function:
 * Iterates Vector and puts matching Regions into the Meta Region.
 * Returns if the area has changed (For now this is iff a region was added).
 */
bool helperRegionsInMeta(MetaRegion & mr, vector<Region> & vectorOfRegions, unordered_set<Region *> & regionFound, int currentFrame){

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
        int framesDifference =   currentFrame - region.playerInRegion->frames.back();

        //if(regionFound.count(&region) != 0) break; // Region was already handled in a different MetaRegion.


        if(Region::regionsInRelativeProximity(region, Region(mr.area), 10) && ! regionInMeta ){

            areaUnchanged = false;
            mr.area |= region.coordinates;
            mr.metaOldRegions.push_back(&region);
            regionFound.insert(&region);

        }
    }
    return areaUnchanged;
}


/*
 * Tries to determine some "Meta Regions" by grouping all the Regions in physical proximity together.
 * If a region from the last frame has no new region at all nearby, they will be marked as out of sight.
 */
vector<MetaRegion> RegionTracker::calcMetaRegions() {

    vector<int> indicesUnhandledRegions(regionsNewFrame.size());
    std::iota(std::begin(indicesUnhandledRegions),std::end(indicesUnhandledRegions), 0);

    Region * currentRegion;
    Region * regionToCompare;
    bool areaUnchanged;

    vector<MetaRegion> metaRegions;

    // A set in which every Region from the last frame is, which found an associated MetaRegion
    unordered_set<Region *> associatedMRFound;
    associatedMRFound.reserve(regionLastFrame.size());
    unordered_set<Region *> outOfSightFound;

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
                if(Region::regionsIntersect(*regionToCompare, Region(currentMetaRegion.area))
                || Region::regionsInRelativeProximity(*regionToCompare, Region(currentMetaRegion.area), 10)){ // TODO Num of frames passed is magic literal
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

            areaUnchanged &= helperRegionsInMeta(currentMetaRegion, regionLastFrame, associatedMRFound, currentFrame);
            areaUnchanged &= helperOutOfSightInMeta(currentMetaRegion, outOfSightRegions, outOfSightFound, currentFrame);

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

// TODO: Would it be good to make Color a "hard" criterion and just dismiss every color combination which differs too much?
/*
 * Calculate how similiar two Regions are to each other.
 * Things taken into Consideration: Size, Histogramm, Position, Colorscheme.
 * The bigger the value, the more the likelihood of them representing the same area.
 */

double calcWeightedSimiliarity(Region  * r1, Region *r2, Rect area, Mat frame){
    double similaritySize, similarityPosition, similarityHistogramm, similarityColor;

    // Size
    similaritySize = float(r1->coordinates.area()) / float(r2->coordinates.area());
    if(similaritySize > 1) similaritySize = 1 / similaritySize;

    // Position
    // Compare the Position of the upper left corner, hypotenuseMetaRegion is the maximum possible distance
    // and would result in a value of zero.
    double hypotenuseMetaRegion = sqrt(pow(area.width , 2) + pow(area.height, 2));
    double lengthVector = sqrt(pow(r1->coordinates.x - r2->coordinates.x, 2) + pow(r1->coordinates.y - r2->coordinates.y, 2));
    similarityPosition = (hypotenuseMetaRegion - lengthVector) / hypotenuseMetaRegion;

    //Histogram
    // http://answers.opencv.org/question/8154/question-about-histogram-comparison-return-value/
    // Return Value of CV_COMP_CORRELL: -1 is worst, 1 is best. -> Map to  [0,1]
    /* Mat hist1, hist2;
    histFromRect(frame, r1->coordinates, hist1);
    histFromRect(frame, r2->coordinates, hist2);

    similarityHistogramm = compareHist(hist1, hist2, CV_COMP_CORREL);
    similarityHistogramm = (similarityHistogramm / 2) + 0.5; // Map [-1,1] to [0,1]
     */
    similarityHistogramm = 0;
    
    // Color
    // CIE94 Formula. If the difference has a value greater than 100, similarityColor turns negativ.
    // TODO: Think about the difference Value: Maybe make it turn negative with an even smaller Value like 20.
    // 0x56215dc9c318


    similarityColor = deltaECIE94(
                r1->labShirtColor[0], r1->labShirtColor[1], r1->labShirtColor[2],
                r2->labShirtColor[0], r2->labShirtColor[1], r2->labShirtColor[2]
        );

    similarityColor = (100 - similarityColor) / 100;


    return similaritySize + similarityPosition + similarityHistogramm + similarityColor;
}


/*
 * Uses matchOldAndNewRegions to get a matrix which represents how the Regions correspond to each other.
 * If two matching Regions are found the correct FootballPlayer will be assigned and the old Region deleted from
 * outOfSightRegions if its needed.
 */
void RegionTracker::assignRegions( MetaRegion & metaRegion) {

    // Create and Initalize Array filled with zeroes.
    int * matching =  new int[metaRegion.metaNewRegions.size() * metaRegion.metaOldRegions.size()];
    for(int it = 0; it < metaRegion.metaOldRegions.size() * metaRegion.metaNewRegions.size(); ++it){
        matching[it] = 0;
    }


    metaRegion.matchOldAndNewRegions(matCurrentFrame, matching);

#define DEBUGPRINT
    #ifdef DEBUGPRINT
    // Print matching
            for(int rowCounter = 0; rowCounter < metaRegion.metaOldRegions.size(); ++rowCounter){
                for(int colCounter = 0; colCounter < metaRegion.metaNewRegions.size(); ++colCounter){
                    if(matching[(rowCounter * metaRegion.metaNewRegions.size()) + colCounter] == 1)
                        cout << 1;
                    else cout << 0;

                }
                cout << endl;
            }
#endif

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

        }
    }
    delete[] matching;
}

/*
 * Get the selection of weights which has the highest possible Sum.
 * Constraint: There can be at most one "1" in every row and column.
 * Bruteforce solution, as the arrays are fairly small.
 */
void optimizeWeightSelection(int rows, int cols, double * const weightMatrix, int * selected){
    short selectedCells[rows];
    short temp_selectedCells[rows];

    for(int i = 0; i < rows; ++i){
        selectedCells[i] = -2;
        temp_selectedCells[i] = -2;
    }
    bool finished = false;
    double temp_Weight;
    double maxWeight = 0;

    int currentRow = 0;

    while(!finished){

        ++temp_selectedCells[currentRow];

        if(temp_selectedCells[currentRow] == cols){ // Last Column of Row was checked
            // Check if loop is finished
            finished = true;
            for(int i = 0; i < rows; ++i){
                finished &= temp_selectedCells[i] >= cols-1;
            }
            if(! finished){
                temp_selectedCells[currentRow] = -2;
                --currentRow;
            }
            continue;
        }
        else if(currentRow != rows-1 ){ // Last Column, combination was checked in last iteration.
            ++currentRow;
            continue;
        }
        else{ // Last Row, increase counter and  check new selection

            // Check if Selection is valid and Calc selected Weight
            bool selectionValid = true;
            temp_Weight = 0;
            for(int i = 0; i < rows; ++i){
                for(int r = i+1; r < rows; ++r){
                    selectionValid &= ((temp_selectedCells[i] != temp_selectedCells[r]) || temp_selectedCells[i] == -1);
                }

                if(temp_selectedCells[i] > -1){
                    temp_Weight += weightMatrix[(cols * i) + temp_selectedCells[i]];
                }
            }

            // If new Maxmium was found,
            if(selectionValid && temp_Weight > maxWeight){
                maxWeight = temp_Weight;
                for(int i = 0; i < rows; ++i)
                    selectedCells[i] = temp_selectedCells[i];
            }

        }
        // Check if loop is finished
        finished = true;
        for(int i = 0; i < rows; ++i){
            finished &= temp_selectedCells[i] == cols-1;
        }


    }

    for(int i = 0; i < rows; ++i){
        if(selectedCells[i] != -1){
            selected[(i * cols) + selectedCells[i]] = 1;
        }
    }
}


/*
 * Create a matrix M in which is saved which regions are the most similiar to each other.
 * If M[i,j] == 1, then metaOldRegions[i] corresponds to metaNewRegion[j]
 */
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

    // Print Matrix for debugging purposes
#define P2C_SHOW_MATCHING_MATRIX
#ifdef P2C_SHOW_MATCHING_MATRIX
    cout << std::fixed << std::setw(5) << std::setprecision(4);

    for(int rowCounter = 0; rowCounter < oldRegionSize; ++rowCounter){
        currentOldRegion = metaOldRegions[rowCounter];
        for(int colCounter = 0; colCounter < metaNewRegions.size(); ++colCounter){

            cout  << matchingMatrix[(rowCounter * newRegionSize) + colCounter]<< " ";
        }
        cout << endl;
    }
    cout.clear();
#endif

    // Create a matrix which shows how the Regions correspond to each other.
    optimizeWeightSelection(metaOldRegions.size(), metaNewRegions.size(), matchingMatrix, matching);
    return matching;
}

/*
 * old Regions: regions from the last frame and out of sight regions.
 * new Regions: Regions from the current frame.
 *
 * Assign a Player to every Region in the Meta Region, mark the rest as out of sight.
 * Every new Region which has only one old Region nearby will be assigned the player of the  old region.
 *
 * When multiple old Regions correspond to one or more new Regions, assignRegions will be invoked.
 *
 * If a new Region has no old Region nearby, all the out of sight regions will be taken into consideration
 * and also matchOldAndNewRegions will be used on a new Meta Region containing all unmatched old and new regions.
 *
 * If absolutely no old Region was found, a new FootBallPlayer will be created.
 */
void RegionTracker::interpretMetaRegions(vector<MetaRegion> & mr) {

    noMatchFound.clear();

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
                noMatchFound.push_back(region);
            }
        }

        else{
            assignRegions(metaRegion);
        }
        // If no matching old Region was found for a new Region, put it in "no Match found" and check for a different
        // out of sight region, as simply crating a new player was not very successful.
        for(Region * newRegion: metaRegion.metaNewRegions){
            if(newRegion->playerInRegion == nullptr){
                noMatchFound.push_back(newRegion);
            }
        }

    }

    // Try to find a matching Region for all the Regions which did not find a Partner.
    Rect area;
    unsigned char * shirtColorOld, * shirtColorNew;
    double distance;
    int framesPassed;
    for(Region * outOfSight : outOfSightRegions){
         shirtColorOld = outOfSight->labShirtColor;
        for(Region * noPartner: noMatchFound){
            shirtColorNew = noPartner->labShirtColor;
            double deltaColor = deltaECIE94(
                    shirtColorOld[0], shirtColorOld[1], shirtColorOld[2],
                    shirtColorNew[0], shirtColorNew[1], shirtColorNew[2]

                    );

            bool colorsMatch =  deltaColor < 15;

            distance = sqrt(pow(outOfSight->coordinates.y - noPartner->coordinates.y , 2) + pow(outOfSight->coordinates.x - noPartner->coordinates.x ,2));
            framesPassed = currentFrame - outOfSight->playerInRegion->frames.back();
            bool distancePlausible = distance < framesPassed * (noPartner->coordinates.height / 6); // Rough estimate of max velocity. TODO This, better.


            if(colorsMatch && distancePlausible) {
                noPartner->playerInRegion = outOfSight->playerInRegion;
                deleteFromOutOfSight(outOfSight->playerInRegion);
                break; // No double assignments.
            }
        }
    }


    // For the Regions still not having an assigned Player, create a new one
    for(Region * noAssignedPlayer: noMatchFound){
        if(noAssignedPlayer->playerInRegion == nullptr){
            noAssignedPlayer->playerInRegion = createNewFootballPlayer(noAssignedPlayer->coordinates);
            cout << " New Player created: " << noAssignedPlayer->playerInRegion->identifier << endl;
            cout << "Out of sight size: " << outOfSightRegions.size() << endl;
        }
    }


    //Draw the meta Regions
    for(MetaRegion const & metaRegion1: mr ){

        rectangle(matCurrentFrame, metaRegion1.area, Scalar(255,0,0), 1 );
    }


}

/*
 * If the given FootballPlayer is in outOfSight Regions, the corresponding Region will be deleted.
 */
void RegionTracker::deleteFromOutOfSight(FootballPlayer * searchFor) {
    auto iterator = outOfSightRegions.begin();
    for(Region * r : outOfSightRegions){
        if(r->playerInRegion == searchFor){
            outOfSightRegions.erase(iterator);
            delete r;
        }
        ++iterator;
    }
}

void RegionTracker::addToOutOfSight(Region * regionPtr) {

    for(Region * region: outOfSightRegions) {
        if (region->playerInRegion->identifier == regionPtr->playerInRegion->identifier) return;
    }
    assert(regionPtr->coordinates.area() != 0);

    auto * newOutOfSight = new Region(*regionPtr);
    outOfSightRegions.push_back(newOutOfSight);
    assert(outOfSightRegions.back()->coordinates.area() != 0);
}

FootballPlayer *RegionTracker::createNewFootballPlayer(Rect const & coordinates) {
    ++objectCounter;
    FootballPlayer * fp = new FootballPlayer(coordinates, currentFrame, string(to_string(objectCounter)));
    return fp;
}


FootballPlayer::FootballPlayer(Rect coordinate, int frame, string const & identifier) {

    coordinates.emplace_back(coordinate);
    frames.emplace_back(frame);
    this->identifier = string(identifier);
    hist = Mat();
    x_vel = 0;
    y_vel = 0;

}

void FootballPlayer::addPosition(Rect coordinates, int frame) {
    if(frames.back() != frame) {
        this->coordinates.emplace_back(Rect(coordinates));
        this->frames.push_back(frame);
    }
}


/*
 * Calc new velocity, update coordinates and frame num.
 */
void FootballPlayer::update(Rect const &coordinates, int frame) {

    int numKnownFrames = frames.size();

    addPosition(coordinates, frame);

    // Get the last 5 Frames to estimate the position.
    int lookUpNFrames = 5;

    auto coordinatesIterator = this->coordinates.begin();
    auto frameIterator = this->frames.begin();
    if(numKnownFrames < lookUpNFrames) lookUpNFrames = numKnownFrames;

    Rect & lastPosition = * coordinatesIterator;

    double deltaX, deltaY;
    while(coordinatesIterator != this->coordinates.end()){
        deltaX = lastPosition.x - coordinatesIterator->x;
        deltaY = lastPosition.y - coordinatesIterator->y;

        x_vel += deltaX;
        y_vel += deltaY;

        lastPosition = * coordinatesIterator;
        ++coordinatesIterator;

    }

    x_vel = x_vel / numKnownFrames;
    y_vel = y_vel / numKnownFrames;


}

/********************************************
 *                                          *
 *          Region Class Methods            *
 *          ^^^^^^^^^^^^^^^^^^^^            *
 ********************************************/

/*
 * Adds the Position of the Region and the current Frame to the tracked objects in the Region.
 */
void Region::updatePlayerInRegion(int frameNum) {

    playerInRegion->update(coordinates, frameNum);

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

void estimatePosition(Region const & r1, int framesPassed, Rect & r){
    int pixelPerFrame = 3;
    Rect rect1 = Rect(r1.coordinates);
    int x1, y1,width1 ,height1;
    x1 = rect1.x - (framesPassed * pixelPerFrame);
    y1 = rect1.y - (framesPassed * pixelPerFrame);
    width1 = rect1.width + (2 * pixelPerFrame * framesPassed);
    height1 = rect1.height + (2 * pixelPerFrame * framesPassed);

    r.x = x1;
    r.y = y1;
    r.width = width1;
    r.height = height1;
}

bool Region::regionsInRelativeProximity(Region const &r1, Region const &r2, int framesPassed) {

    Rect estimate1, estimate2;
    estimatePosition(r1, framesPassed, estimate1);
    estimatePosition(r2, framesPassed, estimate2);

    return (estimate1 & estimate2).area() > 0;
}

Region::Region(Region const &r1) {
    coordinates = Rect(r1.coordinates);
    playerInRegion = r1.playerInRegion;
    for(unsigned char i = 0; i < 3; ++i){
      labShirtColor[i] = r1.labShirtColor[i];
      bgrShirtColor[i] = r1.bgrShirtColor[i];
    }

}


Mat Region::getLabColors(Mat const &frame, int colorCount) {
    Mat regionImgReference, regionImgCopy;

        regionImgReference = frame(coordinates);
        regionImgReference.copyTo(regionImgCopy);

        // Copy all the Pixel values to the samples Mat
        int clusterCount = colorCount;
        Mat labels, centers;

    helperBGRKMean(regionImgCopy, colorCount, labels, centers);

        int  clusterColorCount[clusterCount];
        for(int i = 0; i < clusterCount; ++i) *(clusterColorCount + i) = 0;


        for(int x = 0; x < labels.rows; ++x){
            (* (clusterColorCount + labels.at<int>(x, 0)))++;
        }

        Mat rgbColorClusters(Size(clusterCount,1), CV_8UC3);
        for(int i = 0; i < clusterCount; ++i) {
            float blue = centers.at<float>(i, 0);
            float green = centers.at<float>(i, 1);
            float red = centers.at<float>(i, 2);
            /*
            printf("Cluster %i: %i with (B,G,R) %f, %f, %f \n",
                   i, *(clusterColorCount + i), blue, green,
                   red);*/
            rgbColorClusters.at<Vec3b>(0, i)[0] = red;
            rgbColorClusters.at<Vec3b>(0, i)[1] = green;
            rgbColorClusters.at<Vec3b>(0, i)[2] = blue;

        }
        Mat labColorCluster;
        cvtColor(rgbColorClusters, labColorCluster, COLOR_BGR2Lab);

// #define P2C_SHOW_KMEAN_WINDOW
#ifdef P2C_SHOW_KMEAN_WINDOW
        namedWindow("KMEAN");
        Mat new_image( regionImgCopy.size(), regionImgCopy.type() );

        cout << labColorCluster << endl;
        for( int y = 0; y < regionImgCopy.rows; y++ ) {
            for (int x = 0; x < regionImgCopy.cols; x++) {
                // Save the index of the Cluster the current pixel is in in cluster_idx
                // (Labels is (imgCopy.rows * imgCopy.cols) long and saves the cluster every pixel is in).
                // centers saves the rgb values the center of every cluster.
                int cluster_idx = labels.at<int>(y + x * regionImgCopy.rows, 0);
                new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
                new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
                new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
            }

        }
        putText(new_image,  playerInRegion->identifier, Point(0,coordinates.height), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255), 2);
        imshow("KMEAN", new_image);

        while(true){
            char c = waitKey(30);
            if (c == 32) break;
        }
#endif


    return Mat(labColorCluster);
}

/*
 * Try to determine the color of the shirt.
 * Perform k-Mean on the region, and check in which "Bucket" the pixels who are in the middle of the region are mostly in.
 *  ->  Iterate rows and columns, add the appearances of the pixels and multiply by a factor which is determined by the position in the region.
 *      Then, normalize the result.
 */
Mat Region::getShirtColor(Mat const &frameFull) {
    Mat regionImgReference, regionImgCopy;

    regionImgReference = frameFull(coordinates);
    regionImgReference.copyTo(regionImgCopy);
    Mat frame = regionImgReference;

    const int colorCount  = 2;
    Mat labels, centers;
    helperBGRKMean(frame, colorCount, labels, centers);
    double weight[colorCount];
    int numColorAppearances[colorCount];

    // Prepare Arrays
    for(int index = 0; index < colorCount; ++index){
        weight[index] = 0;
        numColorAppearances[index] = 0;
    }



    double yFactor, xFactor; // Range [1,2]
    int clusterId;
    float frameRows = frame.rows;
    float frameCols = frame.cols;
    for(int y = 0; y < frame.rows; ++y){
        yFactor = 2 - (abs(float(y) - ( frameRows /2.0f)) / (frameRows/2));
        for(int x = 0; x < frame.cols; ++x){
            xFactor = 2 - (abs(y - (frameCols /2.0f)) / (frameCols/2.0f));
             clusterId = labels.at<int>(y + x * frame.rows, 0);
             weight[clusterId] += (xFactor + yFactor);
             ++numColorAppearances[clusterId];
        }
    }

    for(int index = 0; index < colorCount; ++index){
        weight[index] = weight[index] / numColorAppearances[index];
    }

    double * elementPtr = max_element(weight, weight + colorCount );
    int colorIndex = elementPtr - weight;



    Mat colorValues(1,1,CV_8UC3);
    uchar * cvPtr = colorValues.ptr(0);
    cvPtr[0] = centers.at<float>(colorIndex, 0);
    cvPtr[1] = centers.at<float>(colorIndex, 1);
    cvPtr[2] = centers.at<float>(colorIndex, 2);


    return colorValues;
}

/*
 * Fills the Paramters bgrShirtColor and labShirtColor.
 */
void Region::createColorProfile(Mat const &frame) {
    Mat bgrShirtColorTemp = getShirtColor(frame);
    Mat labShirtColorTemp;
    cvtColor(bgrShirtColorTemp, labShirtColorTemp, CV_BGR2Lab);

    uchar * bgrColorPtr = bgrShirtColorTemp.ptr<uchar>(0);
    uchar * labColorPtr = labShirtColorTemp.ptr<uchar>(0);

    for(int i = 0; i < 3; ++i){
        labShirtColor[i] = labColorPtr[i];
        bgrShirtColor[i] = bgrColorPtr[i];
    }


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


// See https://en.wikipedia.org/wiki/Color_difference#CIE94 for the formula
double deltaECIE94(unsigned char L1, char a1, char b1, unsigned char L2, char a2, char b2) {
    // Two constants for our purpos (graphics)
    const double K1 = 0.045;
    const double K2 = 0.015;
    // Weighting factors. Might adapt later
    const double kc = 1;
    const double kh = 1;
    const int kL  = 1;


    // Components of the formula
    int deltaL = L1 - L2;
    double C1 = sqrt((pow(a1, 2) + pow(b1, 2)));
    double C2 = sqrt((pow(a2, 2) + pow(b2, 2)));
    double deltaC = C1 - C2;

    int deltaA = a1 - a2;
    int deltaB = b1 - b2;

    double SC = 1 + (K1 * C1);
    double SH = 1 + (K2 * C1);

    double deltaH = sqrt(pow(deltaA, 2) + pow(deltaB, 2)  - pow(deltaC, 2));

    double deltaE94 = sqrt(
            pow((double(deltaL)/ kL), 2) +
            pow((deltaC / (kc * SC)),2 ) +
            pow(deltaH / (kh * SH), 2)
            );

    return deltaE94;

}

void helperBGRKMean(Mat const &frame, int clusterCount, Mat &labels, Mat &centers) {

    Mat samples(frame.rows * frame.cols, 3, CV_32F);
    for( int y = 0; y < frame.rows; y++ ){
        for( int x = 0; x < frame.cols; x++ ) {
            for (int z = 0; z < 3; z++){
                samples.at<float>(y + x * frame.rows, z) = frame.at<Vec3b>(y, x)[z];
            }
        }
    }


    int attempts = 5;

    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );
    //    #define P2C_SHOW_KMEAN_WINDOW

#ifdef P2C_SHOW_KMEAN_WINDOW
    namedWindow("KMEAN");
        Mat new_image( frame.size(), frame.type() );

        // cout << labColorCluster << endl;
        for( int y = 0; y < frame.rows; y++ ) {
            for (int x = 0; x < frame.cols; x++) {
                // Save the index of the Cluster the current pixel is in in cluster_idx
                // (Labels is (imgCopy.rows * imgCopy.cols) long and saves the cluster every pixel is in).
                // centers saves the rgb values the center of every cluster.
                int cluster_idx = labels.at<int>(y + x * frame.rows, 0);
                new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
                new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
                new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
            }

        }
        //  putText(new_image,  playerInRegion->identifier, Point(0,coordinates.height), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255), 2);
        imshow("KMEAN", new_image);

        while(true){
            char c = waitKey(30);
            if (c == 32) break;
        }
#endif
}
