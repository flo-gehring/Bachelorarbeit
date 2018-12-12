//
// Created by flo on 15.10.18.
//

#include "tracking.h"
#include <set>


// TODO: Velocity information for better prediction



/*
 * Initializes the Tracker.
 * Performs the first round of detection which differs slightly from the following.
 * Cleans up from eventual past detections.
 */
int RegionTracker::initialize(Mat frame) {

    // Clean up before tracking Players in a new video.
    for(FootballPlayer * fp: footballPlayers) {
        delete fp;
    }
    footballPlayers.clear();

    regionsNewFrame.clear();
    regionLastFrame.clear();

    for (Region * r: outOfSightRegions){
        delete r;
    }
    outOfSightRegions.clear();
    occludedPlayers.clear(); // Cleanup complete.

    objectCounter = 0;
    currentFrame = 0;
    matCurrentFrame = frame;


    // All the detected Objects will be stored as rectangles in detectedRects
    vector<Rect> detectedRects = detectOnFrame(frame);

    objectCounter = 0;

    FootballPlayer * newPlayer;

    for(auto it = detectedRects.begin(); it != detectedRects.end(); ++it){

        CV_Assert(it->area() != 0);

        newPlayer = new FootballPlayer((*it), 0, to_string(objectCounter));

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

    for(Rect const & rect: newRects){

        regionsNewFrame.emplace_back(Region(rect));
        assert((rect).area() != 0);
    }

    for(Region  & newRegion: regionsNewFrame) newRegion.createColorProfile(matCurrentFrame);

    vector<MetaRegion> vectorMetaRegion = calcMetaRegions();
    interpretMetaRegions(vectorMetaRegion);
    for(Region * r1:outOfSightRegions){
        for (Region const & r2 : regionsNewFrame) assert(r1->playerInRegion != r2.playerInRegion);
    }



    for(Region & newRegion: regionsNewFrame) newRegion.updatePlayerInRegion(currentFrame);

    printInfo(vectorMetaRegion);


    return false;
}

/*
 * Projects the frame, which is an equirectangular Panorama, onto the six sides of a cube and performs object
 * Detection on them.
 * The coordinates of the detected Bounding Boxes are projected back onto the original frame.
 */
vector<Rect> RegionTracker::detectOnFrame(Mat  & frame) {

    const int numOfFaces = 4; // if 4 -> neither top or bottom face. 6 -> all faces.

    vector<Rect> detected;
    int sideLength = 500;
    Mat face;
    Rect panoramaCoords;

    array<vector<float>, numOfFaces> scores;
    vector<int> indices;

    array<vector<Rect>, numOfFaces> tmpDetected;
    tmpDetected.fill(vector<Rect>());

    unsigned long countDetectedBoxes = 0;



    // We usually dont need top and bottom, this also increases performance massively,
    // because the bb of the cameraman was rather large, which slowed down the k-mean Algorithm.
    for(int faceId = 0; faceId < numOfFaces; ++faceId){
        createCubeMapFace(frame, face, faceId, sideLength, sideLength);
        darknetDetector.detect_and_display(face);



        for (auto detectedObjects = darknetDetector.found.begin();
            detectedObjects != darknetDetector.found.end(); ++detectedObjects) {

            tmpDetected[faceId].push_back((*detectedObjects).rect);
            scores[faceId].push_back((*detectedObjects).prob);
            ++countDetectedBoxes;
            }


    }

    // Sort out bounding boxes on the edge of cube sides who probably denote the same person (or merge them)
    // This only makes sense for the first 4 face sides.
    // TODO: Actually we may be able to remove scores, which makes the loop simpler.
    int rightFace;
    int spaceThreshold  = 5; //
    for(int leftFace = 0; leftFace < 4; ++leftFace){
        rightFace = (leftFace + 1) % 4; //Wrap around if you reach the right most cube face.

        for(Rect  & leftRect: tmpDetected[leftFace]){
            if(leftRect.x + leftRect.width > sideLength - spaceThreshold){

                for(int  iteratorOffset = 0 ; tmpDetected[rightFace].begin() + iteratorOffset !=  tmpDetected[rightFace].end(); ){

                    Rect & rightRect = tmpDetected[rightFace][iteratorOffset];
                    if(rightRect.x < spaceThreshold && // Check if the Bounding Boxes overlap on the vertical axis.
                            ((leftRect.y - spaceThreshold <=  rightRect.y && leftRect.y + spaceThreshold >= rightRect.y)
                            || (leftRect.y - spaceThreshold <= rightRect.y + rightRect.height && leftRect.y + spaceThreshold >= rightRect.y + rightRect.height))){ // Bounding Boxes probably denote the same player.

                        if(rightFace != 0){
                            leftRect.width += rightRect.width;
                        }
                        tmpDetected[rightFace].erase(tmpDetected[rightFace].begin() + iteratorOffset); //
                        scores[rightFace].erase(scores[rightFace].begin() + iteratorOffset);
                        --countDetectedBoxes;
                    }
                    else {
                       ++iteratorOffset;
                    }

                }

            }
        }

    }

    // Flatten the detected Boxes and project them onto the whole frame
    vector<Rect> flattenedDetected;
    flattenedDetected.reserve(countDetectedBoxes);
    vector<float> flattenedScores;
    flattenedScores.reserve(countDetectedBoxes);

    vector<Rect> & tmpFlattenRects = tmpDetected[0];
    vector<float> & tmpFlattenScores = scores[0];

    Rect currentFlattenRect;
    float currentFlattenScore;
    int flattenCounter = 0;
    for(int faceSide = 0; faceSide < numOfFaces; ++ faceSide){

        tmpFlattenRects = tmpDetected[faceSide];
        tmpFlattenScores = scores[faceSide];

        for(int i = 0; i < tmpDetected[faceSide].size(); ++i){

            currentFlattenRect = tmpFlattenRects[i];
            currentFlattenScore = tmpFlattenScores[i];

            mapRectangleToPanorama(frame, faceSide, sideLength, sideLength, currentFlattenRect, panoramaCoords);

            flattenedDetected.emplace_back(Rect(panoramaCoords));
            flattenedScores.emplace_back(currentFlattenScore);
        }
    }
    return flattenedDetected;

}

void RegionTracker::drawOnFrame(Mat frame, vector<MetaRegion> const & metaRegions) {
    for (Region & r: regionsNewFrame){
        string id;
        id = r.playerInRegion->identifier;

        textAboveRect(frame, r.coordinates, id);
        rectangle(frame, r.coordinates, Scalar(0, 0, 255), 2);

    }

    for(MetaRegion const & mr : metaRegions){
        string players;
        rectangle(frame, mr.area, Scalar(255, 0, 0), 2);
        for(Region * region : mr.metaOldRegions){
            players.append(region->playerInRegion->identifier);
            players.append(", ");
        }
        if(players.length() != 0)
            players.erase(players.size() - 2, 2);
        putText(frame, players, Point(mr.area.x , mr.area.y + mr.area.height + 20), FONT_HERSHEY_PLAIN, 3, Scalar(255, 0, 0, 2));
        players.clear();
    }
}

void RegionTracker::trackVideo(const char *filename) {

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
    const char * windowName;
    VideoWriter vw;

    initialize(frame);
    if(! saveVideo) {
        windowName = "Occlusion Tracker";
        namedWindow(windowName);
        resize(frame, resizedFrame, Size(1980, 1020));
        imshow(windowName, resizedFrame);
    }
    else {

        vw = VideoWriter(saveVideoPath, VideoWriter::fourcc('M', 'J', 'P', 'G'),
                       video.get(CAP_PROP_FPS),
                       frame.size(), true);
    }


    drawOnFrame(frame, vector<MetaRegion>());

    time_t timeStart = time(0);
    float avgFrameRate = 0;
    float secondsPassed;
    int frameCounter = 0;

    video >> frame;
/*
 * Skip some frames-
    while(frameCounter < 140){
        video >> frame;
        ++frameCounter;
   } */



    while(! frame.empty()){

        if(analysisData){
            fprintf(analysisDataFile, "------------ \n Frame: %i: \n", currentFrame + 1);
        }
        waitKey(30);
        matCurrentFrame = frame;

        update(frame);

        vector<MetaRegion> mr = calcMetaRegions();

         ++frameCounter;
         secondsPassed = (time(0) - timeStart);

        if(secondsPassed != 0){
            avgFrameRate = float(frameCounter) / secondsPassed;
        }

        putText(frame,"FPS: " + to_string(int(avgFrameRate)), Point(0,50), FONT_HERSHEY_PLAIN, 4, Scalar(0,0,255), 2);
        putText(frame, "Tracker: Darknet" , Point(0, 100), FONT_HERSHEY_PLAIN, 4, Scalar(0,0,255), 2);
        putText(frame, "Frame No." + to_string(frameCounter), Point(0, 150), FONT_HERSHEY_PLAIN, 4, Scalar(0,0,255), 2);

        for(Region const & region : regionsNewFrame){

            CV_Assert(! region.playerInRegion->identifier.empty());

        }


        drawOnFrame(frame, mr);

    if(! saveVideo) {
        resize(frame, resizedFrame, Size(1980, 1020));
        imshow(windowName, resizedFrame);
        int c = waitKey(30);
        bool pauseVid = c == 32; // Space
        while (pauseVid) {
            c = waitKey(10);
            if (c == 32) break;
        }
    }
    else {
        vw.write(frame);
    }

    video >> frame;

    }

    vw.release();

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
    delete[] saveVideoPath;
    for(Region *r : outOfSightRegions){
        delete r;
    }
}


bool helperOutOfSightInMeta(MetaRegion & mr, vector<Region *> & outOfSight, unordered_set<Region *> alreadyFound, int currentFrame){
    bool areaUnchanged = true;
    Rect commonArea;
    bool regionInMeta;

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
       alreadyFound.count(region) == 0){ // TODO Num of frames passed is magic literal

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

    vector<int> indicesUnhandledOutOfSight(outOfSightRegions.size());
    std::iota(std::begin(indicesUnhandledOutOfSight), std::end(indicesUnhandledOutOfSight), 0);

    vector<int> indicesUnhandledOld(regionLastFrame.size());
    std::iota(std::begin(indicesUnhandledOld), std::end(indicesUnhandledOld),0);

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
            // TODO Refactor
            for(auto newRegionIndex = indicesUnhandledRegions.begin();
            newRegionIndex != indicesUnhandledRegions.end(); /*Do nothing */){

                regionToCompare = &regionsNewFrame[*newRegionIndex];

                // A new Region from the current Frame for the Meta Region
                if(Region::regionsIntersect(*regionToCompare, Region(currentMetaRegion.area))
                || Region::regionsInRelativeProximity(*regionToCompare, Region(currentMetaRegion.area), 10)){ // TODO Num of frames passed is magic literal
                //if((regionToCompare->coordinates & currentMetaRegion.area).area() > 0){

                    areaUnchanged = false;

                    currentMetaRegion.area |= regionToCompare->coordinates;
                    currentMetaRegion.metaNewRegions.push_back(& regionsNewFrame[* newRegionIndex]);

                    indicesUnhandledRegions.erase(newRegionIndex);

                }
                else{ // Only increase iterator if no Element was deleted.
                    ++newRegionIndex;
                }
            }

            for(auto outOfSightIndex = indicesUnhandledOutOfSight.begin();
                outOfSightIndex!= indicesUnhandledOutOfSight.end(); /*Do nothing */){

                regionToCompare = outOfSightRegions[*outOfSightIndex];



                // A new Region from the current Frame for the Meta Region
                if(Region::regionsIntersect(*regionToCompare, Region(currentMetaRegion.area))
                   || Region::regionsInRelativeProximity(*regionToCompare, Region(currentMetaRegion.area), 10)){ // TODO Num of frames passed is magic literal
                    //if((regionToCompare->coordinates & currentMetaRegion.area).area() > 0){

                    areaUnchanged = false;

                    currentMetaRegion.area |= regionToCompare->coordinates;
                    currentMetaRegion.metaOldRegions.push_back(outOfSightRegions[* outOfSightIndex]);

                    outOfSightFound.insert(outOfSightRegions[*outOfSightIndex]);

                    indicesUnhandledOutOfSight.erase(outOfSightIndex);

                }
                else{ // Only increase iterator if no Element was deleted.
                    ++outOfSightIndex;
                }
            }

            for(auto lastFrameIndex = indicesUnhandledOld.begin();
                lastFrameIndex != indicesUnhandledOld.end(); /*Do nothing */){

                regionToCompare = &regionLastFrame[*lastFrameIndex];

                // A new Region from the current Frame for the Meta Region
                if(Region::regionsIntersect(*regionToCompare, Region(currentMetaRegion.area))
                   || Region::regionsInRelativeProximity(*regionToCompare, Region(currentMetaRegion.area), 10)){ // TODO Num of frames passed is magic literal
                    //if((regionToCompare->coordinates & currentMetaRegion.area).area() > 0){

                    areaUnchanged = false;

                    currentMetaRegion.area |= regionToCompare->coordinates;
                    currentMetaRegion.metaOldRegions.push_back(& regionLastFrame[* lastFrameIndex]);

                    associatedMRFound.insert( &regionLastFrame[* lastFrameIndex]);

                    indicesUnhandledOld.erase(lastFrameIndex);

                }
                else{ // Only increase iterator if no Element was deleted.
                    ++lastFrameIndex;
                }
            }
        }

    }

    for(Region & region:regionLastFrame){
        Region * r = & region;
        if(associatedMRFound.count(r) == 0)

            addToOutOfSight(r);
    }

#ifdef DEBUG
    int tmp = 0;
    for(MetaRegion const & mr : metaRegions) {
        if(mr.metaNewRegions.size() != 1 || mr.metaOldRegions.size() != 1){
            cout << "MetaReg " << tmp << endl;
            cout << "size old: " << mr.metaOldRegions.size() << " size new: " << mr.metaNewRegions.size() << endl;
        }
        ++tmp;
    }
#endif
    for(Region * r : outOfSightFound) assert(outOfSightFound.count(r) <= 1);
    for(Region * r: associatedMRFound) assert(associatedMRFound.count(r) <= 1);
    return metaRegions;
}

// TODO: Would it be good to make Color a "hard" criterion and just dismiss every color combination which differs too much?
/*
 * Calculate how similiar two Regions are to each other.
 * Things taken into Consideration: Size, (Predicted-)Position, Colorscheme.
 * The bigger the value, the more the likelihood of them representing the same area.
 */

double RegionTracker::calcWeightedSimiliarity(const Region  * oldRegion, const Region *newRegion, Rect area){

    double similarityColor;

    // Color
    // CIE94 Formula. If the difference has a value greater than 100, similarityColor turns negative.
    // TODO: Think about the difference Value: Maybe make it turn negative with an even smaller Value like 20.


    similarityColor = deltaECIE94(
                oldRegion->labShirtColor[0], oldRegion->labShirtColor[1], oldRegion->labShirtColor[2],
                newRegion->labShirtColor[0], newRegion->labShirtColor[1], newRegion->labShirtColor[2]
        );

    similarityColor = (100 - similarityColor) / 100;

    double sharedArea = (oldRegion->coordinates & newRegion->coordinates).area();
    double similarityOverlap = (sharedArea / oldRegion->coordinates.area()) + (sharedArea / newRegion->coordinates.area());

    similarityColor = (similarityOverlap == 0)? 0: similarityColor;

    if(analysisData) {
        fprintf(analysisDataFile, "%.4f + %.4f = %.4f \n",
                similarityOverlap, similarityColor,
                similarityOverlap + similarityColor );
    }
    return similarityOverlap + similarityColor;
}


/*
 * Uses matchOldAndNewRegions to get a matrix which represents how the Regions correspond to each other.
 * If two matching Regions are found the correct FootballPlayer will be assigned and the old Region deleted from
 * outOfSightRegions if its needed.
 */
void RegionTracker::assignRegions( MetaRegion & metaRegion) {

    double assignmentThreshold = 1.0f;
    double minDistanceThreshold = 0.7f;

    set<int> indicesUnassignedOld;
    for (int i=0; i < metaRegion.metaOldRegions.size(); i++) indicesUnassignedOld.insert(i);

    set<int> ambiguousRegions;

    // Calculate how well every detected Regions matches against each other.
    // Indices of the first vector match the regions detected in the current Frame,
    // the indices matching the regions detected in the last Frame are saved in the tuple, along with how well they
    // match the new Region.
    vector<vector<tuple<int, double >>> matchingScores;

    for(Region * newRegion : metaRegion.metaNewRegions){
        matchingScores.emplace_back(vector<tuple<int, double>>());

        for(int index = 0; index < metaRegion.metaOldRegions.size(); ++index){
            matchingScores.back().emplace_back(tuple<int, double>(index, calcWeightedSimiliarity(
                    metaRegion.metaOldRegions[index],
                    newRegion,
                    metaRegion.area
                    )));
        }
    }

    // Sort by how well they match

    auto scoreSelector = [&](tuple<int, double> t1, tuple<int, double> t2){
        return get<1>(t1) > get<1>(t2);
    };
    for(vector<tuple<int, double>> & scores: matchingScores){

        sort(scores.begin(), scores.end(), scoreSelector);

        if(analysisData) {
            for (tuple<int, double> const &t : scores) {
                const char * playerId = metaRegion.metaOldRegions[get<0>(t)]->playerInRegion->identifier.c_str();
                fprintf(analysisDataFile,"(%s, %.4f), ", playerId, get<1>(t));
            }
            fprintf(analysisDataFile, "\n");
        }

    }

    bool ambiguous;
    int newRegionIndex = 0;
    if (metaRegion.metaOldRegions.size() > 1) {
        for (vector<tuple<int, double>> &scores : matchingScores) {

            ambiguous = false;
            if (get<1>(scores[0]) > assignmentThreshold &&
                get<1>(scores[0]) - get<1>(scores[1]) > minDistanceThreshold &&
                ambiguousRegions.count(get<0>(scores[0])) == 0) {

                // Check if no other Region matches this region
                for (vector<tuple<int, double>> &check : matchingScores) {

                    if ((&scores != &check) &&
                        get<0>(check[0]) == get<0>(scores[0]) &&
                        abs(get<1>(check[0]) - get<1>(scores[0])) > minDistanceThreshold &&
                        get<1>(check[0]) > assignmentThreshold) {


                        ambiguous = true;
                        // If the selection is ambiguous, we never want to hear from the players again. Delete the Regions from out of sight.
                        ambiguousRegions.insert(get<0>(check[0]));
                        ambiguousRegions.insert(get<0>(scores[0]));

                        indicesUnassignedOld.erase(get<0>(check[0]));
                        indicesUnassignedOld.erase(get<0>(scores[0]));

                    }

                }

                if (!ambiguous) {
                    FootballPlayer *playerInRegion = metaRegion.metaOldRegions[get<0>(scores[0])]->playerInRegion;
                    metaRegion.metaNewRegions[newRegionIndex]->playerInRegion = playerInRegion;

                    indicesUnassignedOld.erase(get<0>(scores[0]));

                    deleteFromOutOfSight(playerInRegion);

                } else {
                    if (analysisData) fprintf(analysisDataFile, "Two Regions match the same! \n");

                }
            }


            ++newRegionIndex;
        }
    }
    else{ //Only one old Region

        double maxVal = -1;
        double sndMaxVal = -2;
        int bestMatching = -1;
        double currentVal;
        newRegionIndex = 0;
        for(vector<tuple<int, double>> & scores: matchingScores){
            currentVal = get<1>(scores[0]);
            if(currentVal > assignmentThreshold){
                if(currentVal > maxVal){
                    sndMaxVal = maxVal;
                    maxVal = currentVal;
                    bestMatching = newRegionIndex;

                }
                else if(currentVal > sndMaxVal){
                    sndMaxVal = currentVal;
                }
            }
            ++newRegionIndex;
        }


        if(maxVal - sndMaxVal > minDistanceThreshold && bestMatching != -1){
            FootballPlayer * fp = metaRegion.metaOldRegions[0]->playerInRegion;

            deleteFromOutOfSight(fp);
            metaRegion.metaNewRegions[bestMatching]->playerInRegion = fp;
            indicesUnassignedOld.erase(0);

        }


    }

    // Players who were ambiguous should not be considered again
    for(int index: ambiguousRegions){

        FootballPlayer * ambiguousPlayer = metaRegion.metaOldRegions[index]->playerInRegion;

        occludedPlayers.erase(ambiguousPlayer);
        deleteFromOutOfSight(ambiguousPlayer);
        indicesUnassignedOld.erase(index);
    }


    // Add the regions who are left to outofsight Regions
    Region * unassignedRegion;
    bool behindPlayer;
    for(int index : indicesUnassignedOld){
        unassignedRegion = metaRegion.metaOldRegions[index];
        behindPlayer = false;
        for(Region * regionWithPlayer : metaRegion.metaNewRegions){
            if(regionWithPlayer->playerInRegion &&
            Region::regionsInRelativeProximity(*regionWithPlayer, *unassignedRegion, 1)){

                occludedPlayers.insert(make_pair(regionWithPlayer->playerInRegion, unassignedRegion->playerInRegion));
                behindPlayer = true;
                break;
            }
        }

        if(! behindPlayer)
            addToOutOfSight(unassignedRegion);
    }

    // Find Players who are maybe out of sight
    double colorSimiliarity;
    for(Region * r: metaRegion.metaNewRegions) {

        if(! r->playerInRegion) {


            FootballPlayer * matchingOutOfSight = nullptr;

            vector<FootballPlayer *> nearbyPlayers;


            for(Region * oofsr: outOfSightRegions){
                unsigned char * labShirtColorPlayer = oofsr->labShirtColor;
                unsigned char * labShirtColorRegion = r->labShirtColor;

                colorSimiliarity = deltaECIE94(labShirtColorPlayer[0], labShirtColorPlayer[1], labShirtColorPlayer[2],
                                                      labShirtColorRegion[0], labShirtColorRegion[1], labShirtColorRegion[2]);
                if(Region::regionsInRelativeProximity(* oofsr, *r, 2) // TODO: This caused a bug because it matched a region which was in another meta region!
                && colorSimiliarity < 30)
                    nearbyPlayers.push_back(oofsr->playerInRegion);
            }

            for(auto playerPair: occludedPlayers){

                if(Region::regionsInRelativeProximity(Region(playerPair.first->coordinates.back()), *r, 2) &&
                nearbyPlayers.size() == 0){
                    nearbyPlayers.push_back(playerPair.second);
                    occludedPlayers.erase(playerPair.first);
                    break;
                }
            }

            if(nearbyPlayers.size() == 1){
                matchingOutOfSight = nearbyPlayers[0];
            }

            if(matchingOutOfSight){
                r->playerInRegion = matchingOutOfSight;
                deleteFromOutOfSight(matchingOutOfSight);

                for(Region * oofsr: outOfSightRegions) assert(matchingOutOfSight != oofsr->playerInRegion);
            }
            else {
                r->playerInRegion = createNewFootballPlayer(r->coordinates);
            }
        }
    }

    for(Region * r: metaRegion.metaNewRegions) assert(r->playerInRegion);



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



}

/*
 * Get the selection of weights which has the highest possible Sum.
 * Constraint: There can be at most one "1" in every row and column.
 * Bruteforce solution, as the arrays are fairly small.
 */
void optimizeWeightSelection(int rows, int cols, double const *  weightMatrix, int * selected){
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
int *  MetaRegion::matchOldAndNewRegions(Mat frame, int * matching, int frameNum, RegionTracker * rt){

    unsigned long oldRegionSize = metaOldRegions.size();
    unsigned long newRegionSize = metaNewRegions.size();

    double matchingMatrix[oldRegionSize * newRegionSize];

    Region * currentOldRegion;


    if(rt->analysisData) {
        fprintf(rt->analysisDataFile, "matchOldAndNewRegion \n");
    }

    // Calculate the weighted similarities for each of the Regions in the Meta Region
    for(int rowCounter = 0; rowCounter < oldRegionSize; ++rowCounter){
        currentOldRegion = metaOldRegions[rowCounter];

        if(rt->analysisData) {
            fprintf(rt->analysisDataFile, "Player %s: \n",  currentOldRegion->playerInRegion->identifier.c_str());
        }

        for(int colCounter = 0; colCounter < metaNewRegions.size(); ++colCounter){

            matchingMatrix[(rowCounter * newRegionSize) + colCounter] =
                    rt->calcWeightedSimiliarity(currentOldRegion, metaNewRegions[colCounter], area);
        }
    }

    // Print Matrix for debugging purposes

    if(rt->analysisData) {

        for (int rowCounter = 0; rowCounter < oldRegionSize; ++rowCounter) {
            currentOldRegion = metaOldRegions[rowCounter];
            for (int colCounter = 0; colCounter < metaNewRegions.size(); ++colCounter) {

                fprintf(rt->analysisDataFile, "%.4f ",  matchingMatrix[(rowCounter * newRegionSize) + colCounter]);
            }
            fprintf(rt->analysisDataFile, "\n");
        }
    }



    // Create a matrix which shows how the Regions correspond to each other.
    optimizeWeightSelection(metaOldRegions.size(), metaNewRegions.size(), matchingMatrix, matching);
    if(rt->analysisData) {
        fprintf(rt->analysisDataFile, "Selection: \n");
        for (int rowCounter = 0; rowCounter < oldRegionSize; ++rowCounter) {
            currentOldRegion = metaOldRegions[rowCounter];
            fprintf( rt->analysisDataFile, "P %s: ", currentOldRegion->playerInRegion->identifier.c_str());
            for (int colCounter = 0; colCounter < metaNewRegions.size(); ++colCounter) {

                fprintf(rt->analysisDataFile, "%i ",  matching[(rowCounter * newRegionSize) + colCounter]);
            }
            fprintf(rt->analysisDataFile, "\n");
        }
    }
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


        for(Region * oldRegion : metaRegion.metaOldRegions) assert(oldRegion->playerInRegion);

        if(metaRegion.metaNewRegions.size() == 1 && metaRegion.metaOldRegions.size() == 1){
            metaRegion.metaNewRegions[0]->playerInRegion = metaRegion.metaOldRegions[0]->playerInRegion;
            deleteFromOutOfSight(metaRegion.metaNewRegions[0]->playerInRegion);
        }
        else if(metaRegion.metaNewRegions.empty()){
            for(Region * region: metaRegion.metaOldRegions){
                assert(region->playerInRegion);
                assert(region->coordinates.area() != 0);
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


    }

    // Try to find a matching Region for all the Regions which did not find a Partner.
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
        if (region->playerInRegion == regionPtr->playerInRegion) return;
        assert(region->playerInRegion != regionPtr->playerInRegion);
    }
    assert(regionPtr->coordinates.area() != 0);

    auto * newOutOfSight = new Region(*regionPtr);
    outOfSightRegions.push_back(newOutOfSight);
    assert(outOfSightRegions.back()->coordinates.area() != 0);
}

FootballPlayer *RegionTracker::createNewFootballPlayer(Rect const & coordinates) {
    ++objectCounter;
    FootballPlayer * fp = new FootballPlayer(coordinates, currentFrame, string(to_string(objectCounter)));
    this->footballPlayers.push_back(fp);
    return fp;
}


bool playerInRegionVector(FootballPlayer * fp, vector<Region> const & vr){

    for(Region const & r: vr) if (fp == r.playerInRegion) return true;

    return false;


}

void RegionTracker::printInfo(vector<MetaRegion> const & metaRegions) {


    FootballPlayer * fp;
    Rect  * coordinates;
    for(Region const & region:regionsNewFrame){
        fp = region.playerInRegion;
        coordinates = & fp->coordinates.back();
        fprintf(roiData, "%i;%s;%i;%i;%i;%i\n", currentFrame, fp->identifier.c_str(), coordinates->x, coordinates->y, coordinates->width, coordinates->height);
    }

    for(MetaRegion const & metaRegion: metaRegions){
        for(Region * regionInMeta : metaRegion.metaNewRegions){
            fp = regionInMeta->playerInRegion;
            if(! playerInRegionVector(fp, regionsNewFrame)){
                fprintf(roiData, "%i;%s;%i;%i;%i;%i\n", currentFrame, fp->identifier.c_str(),
                        regionInMeta->coordinates.x,
                        regionInMeta->coordinates.y,
                        regionInMeta->coordinates.width,
                        regionInMeta->coordinates.height);

            }
        }
    }
}



void RegionTracker::setAOIFile(const char *aoiFilePath) {

    if(roiData != nullptr){
        fclose(roiData);
    }

    roiData = fopen(aoiFilePath, "w");

}

RegionTracker::RegionTracker(const char *aoiFilePath, const char * videoPath) {

    if(! videoPath){ // -> videoPath is default value nullptr.
        saveVideo = false;
    }
    else{
        saveVideo = true;
        strcpy(saveVideoPath, videoPath);
    }
    roiData = fopen(aoiFilePath, "w");

    analysisData = false;
    analysisDataFile = nullptr;
}

RegionTracker::RegionTracker() {

    saveVideo = false;
    roiData = fopen("roidata.txt", "w");
    debugData = fopen("debugdata.txt", "w");
    saveVideoPath = new char[64];
    analysisData = false;
    analysisDataFile = nullptr;
}

void RegionTracker::enableVideoSave(const char *videoFilePath) {

    saveVideo = true;
    strcpy(saveVideoPath, videoFilePath);


}

void RegionTracker::setupAnalysisOutFile(const char * filename){

    analysisData = true;

    if (analysisDataFile) fclose(analysisDataFile);
    analysisDataFile = fopen(filename, "w");


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


Rect FootballPlayer::predictPosition(int frameNum) {


    if(coordinates.size() < 2)
        return Rect(coordinates.back());

    int deltaX[frames.size()];
    int deltaY[frames.size()];

    int iInc;
    for(int i = 0; i < coordinates.size(); ++i){
        iInc = i + 1;
        deltaX[i] = coordinates[coordinates.size() -  iInc].x;
        deltaY[i] = coordinates[coordinates.size() - iInc].y;
    }
    int numIterations = 0;
    for(int i = 0; i < coordinates.size() - 1; ++i){
        iInc = i + 1;

        deltaX[i] = deltaX[iInc] - deltaX[i];
        deltaY[i] = deltaY[iInc] - deltaY[i];
        numIterations = iInc;
        if(i > 0 &&  ((deltaX[i] * deltaX[i-1] < 0) || (deltaY[i] * deltaY[i-1] < 0))){ // Break if the Football Player changes directions
            break;
        }

    }

    int framesPassed = frames.back() - frames[frames.size() - (numIterations + 1)];

    double avgXChange = double(coordinates.back().x - coordinates[coordinates.size() - (numIterations + 1)].x) / double(framesPassed);
    double avgYChange = double(coordinates.back().y - coordinates[coordinates.size() - (numIterations + 1)].y) / double(framesPassed);


    return cv::Rect(
            int(coordinates.back().x + (avgXChange * ( frameNum - frames.back()))),
            int(coordinates.back().y + (avgYChange * (frameNum - frames.back()))),
            coordinates.back().width,
            coordinates.back().height
            );
}
/*
 * Calc new velocity, update coordinates and frame num.
 */
void FootballPlayer::update(Rect const &coordinates, int frame) {

    unsigned long numKnownFrames = frames.size();

    addPosition(coordinates, frame);


    auto coordinatesIterator = this->coordinates.begin();

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

    auto * bgrColorPtr = bgrShirtColorTemp.ptr<uchar>(0);
    auto * labColorPtr = labShirtColorTemp.ptr<uchar>(0);

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

    // Nan is the result of two values with minimal difference being compared. (eg (218, -122,-122) and (220, -123, -123))
    // The fault lies in deltaH, because then deltaA = deltaB = 1 and deltaC = 1.4141 = sqrt(2) which will result in a
    // negativ root being calculated because of insufficient floating Point precision.
    if(isnan(deltaE94)){
        deltaE94 = 0;
    }

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
