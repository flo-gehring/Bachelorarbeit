//
// Created by flo on 15.10.18.
//

#include "tracking.h"
#include "TrackingHelpers.h"

#include <set>

// TODO: Better Management of lost Players. Some old "residue" players cause ambigouity.

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

    pBGSubtractor->apply(frame, foregroundMask);
    for(Region & r: regionsNewFrame){
        r.createColorProfile(frame, foregroundMask);
    }

    return objectCounter;
}

/*
 * Everytime you grab a new Frame from your Videostream, pass it to this Method and
 */
bool RegionTracker::update(Mat frame) {

    matLastFrame = Mat(matCurrentFrame);

    matCurrentFrame = Mat(frame);

    pBGSubtractor->apply(frame, foregroundMask); // TODO: do i still need the bg subtractor?

    ++currentFrame;
    // Update Region Vectors, new regions from last frame are now old, get new Regions from detector
    regionLastFrame.swap(regionsNewFrame);
    regionsNewFrame.clear();

    vector<Rect> newRects = detectOnFrame(frame);

    for(Rect const & rect: newRects){

        regionsNewFrame.emplace_back(Region(rect));

        assert((rect).area() != 0);
    }

    for(Region  & newRegion: regionsNewFrame) newRegion.createColorProfile(matCurrentFrame, foregroundMask);

    vector<MetaRegion> vectorMetaRegion = calcMetaRegions();
    interpretMetaRegions(vectorMetaRegion);


    for(Region & newRegion: regionsNewFrame) newRegion.updatePlayerInRegion(currentFrame);

    // printInfo(vectorMetaRegion);

    // Delete Regions who have not been found for too long.
    /*
    for(Region * oofsR : outOfSightRegions ){
        if(currentFrame - oofsR->playerInRegion->frames.back() > 10) deleteFromOutOfSight(oofsR->playerInRegion);
    }
     */

    return false;
}


bool yAxisOverlap(Rect const & r1, Rect const & r2, int distanceThreshold){
    auto inBetween = [](int x, int y, int z, int inaccuracy){
        return (x + inaccuracy > y && x - inaccuracy < z);
    };
    auto oneSided = [inBetween](Rect const & r1, Rect const & r2, int distanceThreshold){
        return inBetween(r1.y, r2.y, r2.y + r2.height, distanceThreshold) ||
            inBetween(r1.y + r1.height,  r2.y, r2.y + r2.height, distanceThreshold);
    };
    return oneSided(r1, r2, distanceThreshold) || oneSided(r2, r1, distanceThreshold);
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
    int spaceThreshold  = 2; //
    for(int leftFace = 0; leftFace < numOfFaces; ++leftFace){
        rightFace = (leftFace + 1) % numOfFaces; //Wrap around if you reach the right most cube face.

        for(Rect  & leftRect: tmpDetected[leftFace]){
            if(leftRect.x + leftRect.width > sideLength - spaceThreshold){

                for(int  iteratorOffset = 0 ; tmpDetected[rightFace].begin() + iteratorOffset !=  tmpDetected[rightFace].end(); ){

                    Rect & rightRect = tmpDetected[rightFace][iteratorOffset];

                    if(rightRect.x - spaceThreshold <= 0 && yAxisOverlap(leftRect, rightRect, spaceThreshold)){ // Bounding Boxes probably denote the same player.


                        string debugString = "Union of Rect(%i, %i, %i %i) and Rect(%i, %i, %i %i) on Faces %i -> %i . \n";
                        if(debugData){
                            fprintf(debugData, debugString.c_str(), leftRect.x, leftRect.y, leftRect.width, leftRect.height,
                                    rightRect.x, rightRect.y,  rightRect.width, rightRect.height, leftFace, rightFace);
                        }


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
        Scalar playerColor(r.bgrShirtColor[0], r.bgrShirtColor[1], r.bgrShirtColor[2]);

        textAboveRect(frame, r.coordinates, id);
        rectangle(frame, r.coordinates, playerColor, 2);

    }

    for(MetaRegion const & mr : metaRegions){
        string players;

        if(mr.metaNewRegions.size() > 1)
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


/*
 *
 *  Methods Responsible for the actual Tracking
 *  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *
 *  calcMetaRegions: Returns a vector of spatially near Regions, from the current Frame, the last Frame and outOfSightRegions.
 *  calcWeightedSimiliarity: Gives a Number of how well two Regions and their respecting Players match each other.
 *  assignRegions: Calcs which Region in a MetaRegion is the successor of another Region. Also Marks Regions and Footballplayers as outOfSight and so on.
 *  interpetMetaRegions: Not every MetaRegions needs to go through assignRegions.
 */




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

    while(!indicesUnhandledRegions.empty()){

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
            areaUnchanged &= addRegionsToMeta(regionsNewFrame, indicesUnhandledRegions, currentMetaRegion,
                    false, associatedMRFound);
            areaUnchanged &= addRegionPtrToMeta(outOfSightRegions, indicesUnhandledOutOfSight, currentMetaRegion,
                    true, associatedMRFound);
            areaUnchanged &= addRegionsToMeta(regionLastFrame, indicesUnhandledOld, currentMetaRegion,
                    true, associatedMRFound);

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


    bool overlap = (oldRegion->coordinates & newRegion->coordinates).area() != 0;
    double similaritySize = 0;
    double similarityPosition = 0;
    double similarityHistogramm = 0;
    double similarityColor = 0;

    if(overlap){
        // Size
        similaritySize = float(oldRegion->coordinates.area()) / float(newRegion->coordinates.area());
        if(similaritySize > 1) similaritySize = 1 / similaritySize;

        // If the Player was seen in the Last frame, take the old Position, else use the  predicted Position
        // Position
        // Compare the Position of the upper left corner, hypotenuseMetaRegion is the maximum possible distance
        // and would result in a value of zero.
        Rect positionOldRegion;
        if(true /*oldRegion->playerInRegion->frames.back() == currentFrame - 1*/){
            positionOldRegion = oldRegion->coordinates;
        }
        else{
            positionOldRegion = oldRegion->playerInRegion->predictPosition(currentFrame);
#ifdef UNDEF
            cout << "Frame " << frameNum << endl;
        printf("(%i, %i, %i, %i)", positionOldRegion.x, positionOldRegion.y, positionOldRegion.width, positionOldRegion.height );
        cout << endl;
#endif
        }
        double hypotenuseMetaRegion = sqrt(pow(area.width , 2) + pow(area.height, 2));
        double lengthVector = sqrt(pow(positionOldRegion.x - newRegion->coordinates.x, 2) + pow(positionOldRegion.y - newRegion->coordinates.y, 2));
        similarityPosition = (hypotenuseMetaRegion - lengthVector) / hypotenuseMetaRegion;


        //Histogram
        // http://answers.opencv.org/question/8154/question-about-histogram-comparison-return-value/
        // Return Value of CV_COMP_CORRELL: -1 is worst, 1 is best. -> Map to  [0,1]
        Mat hist1, hist2;
        histFromRect(matCurrentFrame, oldRegion->coordinates, hist1);
        histFromRect(matCurrentFrame, newRegion->coordinates, hist2);

        similarityHistogramm = compareHist(hist1, hist2, CV_COMP_CORREL);
        similarityHistogramm = (similarityHistogramm / 2) + 0.5; // Map [-1,1] to [0,1]
        similarityHistogramm = 0;

        // Color
        // CIE94 Formula. If the difference has a value greater than 100, similarityColor turns negative.
        // TODO: Think about the difference Value: Maybe make it turn negative with an even smaller Value like 20.
        // 0x56215dc9c318


        similarityColor = deltaECIE94(
                oldRegion->labShirtColor[0], oldRegion->labShirtColor[1], oldRegion->labShirtColor[2],
                newRegion->labShirtColor[0], newRegion->labShirtColor[1], newRegion->labShirtColor[2]
        );

        similarityColor = (100 - similarityColor) / 100;

    }

    if(analysisData) {
        fprintf(analysisDataFile, "%.4f + %.4f + %.4f + %.4f = %.4f \n",
                similaritySize, similarityPosition, similarityHistogramm, similarityColor,
                similaritySize + similarityPosition + similarityHistogramm + similarityColor );
    }
    return similaritySize + similarityPosition + similarityHistogramm + similarityColor;
}

#ifdef UNDEF
double RegionTracker::calcWeightedSimiliarity2(const Region  * oldRegion, const Region *newRegion, Rect area){

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

    double similarityPosition;

    double similaritySize = double(oldRegion->coordinates.area()) / double(newRegion->coordinates.area());
    similaritySize = (similaritySize > 1)? similaritySize: 1 / similaritySize;
    similaritySize = (similarityOverlap == 0)? 0: similaritySize;

    const Rect & oldRect = oldRegion->coordinates;
    const Rect & newRect = newRegion->coordinates;

    auto pythagoras = [](double a, double b) -> double {
        return sqrt(pow(a, 2) + pow(b,2));
    };

    double maxDistance = pythagoras(area.height, area.width);

    double vectorX = oldRect.x - newRect.x;
    double vectorY = oldRect.y - newRect.y;

    similarityPosition = pythagoras(vectorX, vectorY) / maxDistance;

    if(analysisData) {
        fprintf(analysisDataFile, "%.4f + %.4f = %.4f \n",
                similarityOverlap, similarityColor,
                similarityOverlap + similarityColor );
    }

    int framesDifference = (currentFrame - oldRegion->playerInRegion->frames.back());


    return similaritySize + (similarityOverlap / framesDifference);
}
#endif

/*
 * Uses matchOldAndNewRegions to get a matrix which represents how the Regions correspond to each other.
 * If two matching Regions are found the correct FootballPlayer will be assigned and the old Region deleted from
 * outOfSightRegions if its needed.
 */
void RegionTracker::assignRegions( MetaRegion & metaRegion) {

    // double assignmentThreshold = 1.0f;
    // double minDistanceThreshold = 0.0f;

    set<int> indicesUnassignedOld;
    set<FootballPlayer *> playersAssignedInThisFrame;


    auto searchForValue = [&](unordered_map<FootballPlayer *, FootballPlayer *> occluded, FootballPlayer * searchFor) -> FootballPlayer *{

        for(auto pair: occluded){
            if(pair.second == searchFor) return pair.first;
        }
        return nullptr;

    };

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
    bool otherMatchesBetter;
    int newRegionIndex = 0;
    if (metaRegion.metaOldRegions.size() > 1) {
        for (vector<tuple<int, double>> &scores : matchingScores) {

            double scoreThisRegion = get<1>(scores[0]);
            int indexOldRegion = get<0>(scores[0]);
            ambiguous = false;
            otherMatchesBetter = false;


            Region * newRegion = metaRegion.metaNewRegions[newRegionIndex];

            if (scoreThisRegion > assignmentThreshold &&
                scoreThisRegion - get<1>(scores[1]) <= minDistanceThreshold){

                ambiguousRegions.insert(indexOldRegion);
                ambiguousRegions.insert(get<0>(scores[1]));

                indicesUnassignedOld.erase(indexOldRegion);
                indicesUnassignedOld.erase(get<0>(scores[1]));

                newRegion->playerInRegion = createAmbiguousPlayer(newRegion->coordinates);

            }
            else if(ambiguousRegions.count(get<0>(scores[0]) == 1)){

                newRegion->playerInRegion = createAmbiguousPlayer(newRegion->coordinates);
                ambiguousRegions.insert(indexOldRegion);
                indicesUnassignedOld.erase(get<0>(scores[0]));

            }
            else if (scoreThisRegion > assignmentThreshold) {
                double  scoreOtherRegion;
                // Check if no other Region matches this region
                for (vector<tuple<int, double>> &check : matchingScores) {
                    scoreOtherRegion = get<1>(check[0]);

                    int idBestMatchOther = get<0>(check[0]);
                    int idBestMatchThis = get<0>(scores[0]);

                    if ((&scores != &check) &&
                        get<0>(check[0]) == get<0>(scores[0]) &&
                            scoreOtherRegion > assignmentThreshold){

                        bool tooClose = abs(scoreOtherRegion - scoreThisRegion) < minDistanceThreshold;

                        if(! tooClose && scoreThisRegion > scoreOtherRegion){

                            // Do nothing. Assignment can continue as normal-
                        }
                        else if(! tooClose && scoreOtherRegion > scoreThisRegion){
                            otherMatchesBetter = true;
                        }
                        else{
                            ambiguous = true;
                            // If the selection is ambiguous, we never want to hear from the players again. Delete the Regions from out of sight.
                            // TODO : make function with above code
                            ambiguousRegions.insert(idBestMatchOther);
                            ambiguousRegions.insert(idBestMatchThis);

                            indicesUnassignedOld.erase(idBestMatchOther);
                            indicesUnassignedOld.erase(idBestMatchThis);
                        }

                        }
                }

                FootballPlayer *playerInRegion = metaRegion.metaOldRegions[get<0>(scores[0])]->playerInRegion;

                if(otherMatchesBetter) break;

                if (!ambiguous) {

                    if(playerInRegion->isAmbiguous) playerInRegion = createNewFootballPlayer(metaRegion.metaNewRegions[newRegionIndex]->coordinates);

                    assert(playersAssignedInThisFrame.count(playerInRegion) == 0);

                    playersAssignedInThisFrame.insert(playerInRegion);
                    metaRegion.metaNewRegions[newRegionIndex]->playerInRegion = playerInRegion;

                    indicesUnassignedOld.erase(get<0>(scores[0]));

                    deleteFromOutOfSight(playerInRegion);
                    FootballPlayer * toDelete =  searchForValue(occludedPlayers, playerInRegion);

                    if(toDelete) occludedPlayers.erase(toDelete);

                }
                else { // Players are ambiguous
                    if(playerInRegion->isAmbiguous) newRegion->playerInRegion = playerInRegion;
                    else newRegion->playerInRegion = createAmbiguousPlayer(newRegion->coordinates);

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
            FootballPlayer * keyOccluded = searchForValue(occludedPlayers, fp);
            if(keyOccluded) occludedPlayers.erase(keyOccluded);
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
    for(int index : indicesUnassignedOld){
        unassignedRegion = metaRegion.metaOldRegions[index];

        for(Region * regionWithPlayer : metaRegion.metaNewRegions){
            if(regionWithPlayer->playerInRegion &&
            Region::regionsInRelativeProximity(*regionWithPlayer, *unassignedRegion, 1)){
                occludedPlayers.insert(make_pair(regionWithPlayer->playerInRegion, unassignedRegion->playerInRegion));
                break;
            }
        }

        addToOutOfSight(unassignedRegion);
    }

    // Find Players who are maybe out of sight
    double colorSimiliarity;
    for(Region * r: metaRegion.metaNewRegions) {

        if(! r->playerInRegion) {


            FootballPlayer * matchingOutOfSight = nullptr;

            unordered_set<FootballPlayer *> nearbyPlayers;


            for(Region * oofsr: outOfSightRegions){
                unsigned char * labShirtColorPlayer = oofsr->labShirtColor;
                unsigned char * labShirtColorRegion = r->labShirtColor;

                colorSimiliarity = deltaECIE94(labShirtColorPlayer[0], labShirtColorPlayer[1], labShirtColorPlayer[2],
                                                      labShirtColorRegion[0], labShirtColorRegion[1], labShirtColorRegion[2]);
                if(Region::regionsInRelativeProximity(* oofsr, *r, 2) // TODO: This caused a bug because it matched a region which was in another meta region!
                && colorSimiliarity < 30)
                    nearbyPlayers.insert(oofsr->playerInRegion);
            }

            for(auto playerPair: occludedPlayers){

                if(Region::regionsInRelativeProximity(Region(playerPair.first->coordinates.back()), *r, 2) &&
                nearbyPlayers.empty()){
                    nearbyPlayers.insert(playerPair.second);
                    occludedPlayers.erase(playerPair.first);
                      break;
                }
            }

            if(nearbyPlayers.size() == 1){
                matchingOutOfSight = *nearbyPlayers.begin();
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
    for(Region * r1:outOfSightRegions){
        for (Region const & r2 : regionsNewFrame) assert(r1->playerInRegion != r2.playerInRegion);
    }



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

        rectangle(matCurrentFrame, metaRegion1.area, Scalar(255,0,0), 1);
    }
}


/*
 *      Management of FootballPlayer and Regions.
 *      -----------------------------------------
 */

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


/*
 *      Constructors / Destructors
 *      --------------------------
 */

RegionTracker::~RegionTracker() {
    for(FootballPlayer * footballPlayer : footballPlayers){
        delete footballPlayer;
    }
    delete[] saveVideoPath;
    for(Region *r : outOfSightRegions){
        delete r;
    }
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
    pBGSubtractor = createBackgroundSubtractorMOG2(); //MOG2 approach
}

RegionTracker::RegionTracker() {

    saveVideo = false;
    roiData = fopen("roidata.txt", "w");
    // debugData = fopen("debugdata.txt", "w");
    saveVideoPath = new char[64];
    analysisData = false;
    analysisDataFile = nullptr;
    pBGSubtractor = createBackgroundSubtractorMOG2(); //MOG2 approach

}



/*
 *      Configuration for Video and Dataoutput
 *      --------------------------------------
 */

void RegionTracker::setAOIFile(const char *aoiFilePath) {

    if(roiData != nullptr){
        fclose(roiData);
    }

    roiData = fopen(aoiFilePath, "w");

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



FootballPlayer *RegionTracker::createAmbiguousPlayer(Rect const & coordinates) {
    FootballPlayer * fp = new FootballPlayer(coordinates, currentFrame, string(to_string(-1)));
    this->footballPlayers.push_back(fp);
    fp->isAmbiguous = true;
    return fp;

}


void RegionTracker::printInfo(vector<MetaRegion> const & metaRegions) {

    if(roiData) {
        FootballPlayer *fp;
        Rect *coordinates;
        for (Region const &region:regionsNewFrame) {
            fp = region.playerInRegion;
            coordinates = &fp->coordinates.back();
            fprintf(roiData, "%i;%s;%i;%i;%i;%i\n", currentFrame, fp->identifier.c_str(), coordinates->x,
                    coordinates->y, coordinates->width, coordinates->height);
        }

        for (MetaRegion const &metaRegion: metaRegions) {
            for (Region *regionInMeta : metaRegion.metaNewRegions) {
                fp = regionInMeta->playerInRegion;
                if (!playerInRegionVector(fp, regionsNewFrame)) {
                    fprintf(roiData, "%i;%s;%i;%i;%i;%i\n", currentFrame, fp->identifier.c_str(),
                            regionInMeta->coordinates.x,
                            regionInMeta->coordinates.y,
                            regionInMeta->coordinates.width,
                            regionInMeta->coordinates.height);

                }
            }
        }
    }
}

void RegionTracker::printTrackingResults(const char * filePath) {

    FILE * outFile = fopen(filePath, "w");

    if(! outFile) {
        cerr << "Could not open file " << filePath << " to print results!" << endl;
        exit(-1);
    }

    vector<map<int, Rect>> playerFrameOccurences;
    int maxFrame = 0;
    for(FootballPlayer * fp : footballPlayers) {
        if (! fp->frames.empty() && fp->frames.back() > maxFrame) maxFrame = fp->frames.back();
        playerFrameOccurences.emplace_back(map<int, Rect>());

        for(int i = 0; i < fp->frames.size(); ++i){

            playerFrameOccurences.back()[fp->frames[i]] =  fp->coordinates[i];

        }
    }

    Rect currentCoordinates;
    int id;
    int bbLeft, bbTop, bbWidth, bbHeight;
    for(int currentFrame = 0; currentFrame <= maxFrame; ++currentFrame){
        id = 0;
        for(map<int, Rect>  & frameOccurences: playerFrameOccurences){

            auto coordinatesInFrame = frameOccurences.find(currentFrame);
            currentCoordinates = coordinatesInFrame->second;
            bbLeft = currentCoordinates.x;
            bbTop = currentCoordinates.y;
            bbWidth = currentCoordinates.width;
            bbHeight = currentCoordinates.height;

            if(coordinatesInFrame != frameOccurences.end()){
                fprintf(outFile, "%i, %i, %i, %i, %i, %i, 0, -1, -1, -1\n", coordinatesInFrame->first, id, bbLeft, bbTop, bbWidth, bbHeight);
            }
            ++id;
        }

    }
}

/*
 *
 *  Feature Extraction
 *
 */

void RegionTracker::calcOpticalFlow(Rect const &area) {
#ifdef UNDEF
    Mat inputOld = matLastFrame(area);
    Mat inputNew = matCurrentFrame(area);



    Ptr<DenseOpticalFlow> denseflow = optflow::createOptFlow_PCAFlow();

    Mat grayOld, grayNew;
    cvtColor(inputOld, grayOld, CV_BGR2GRAY);
    cvtColor(inputNew, grayNew, CV_BGR2GRAY);


    Mat ioArray;

    denseflow->calc(grayOld, grayNew, ioArray);
    /*optflow::calcOpticalFlowSF(inputOld, inputNew, ioArray,3, 2, 4, 4.1, 25.5, 18, 55.0,
            25.5, 0.35, 18, 55.0, 25.5, 10);*/

    const Size sz = ioArray.size();
    Mat flow = ioArray;

    Mat img(sz, CV_32FC3);

    for ( int i = 0; i < sz.height; ++i ){
        for ( int j = 0; j < sz.width; ++j ) {
            Vec2f point = ioArray.at<Vec2f>(i, j);
            if(point[0] + point[1] != 0) {
                img.at<Vec3f>(i,j)[0] = 100;
                img.at<Vec3f>(i,j)[1] = 100;
                img.at<Vec3f>(i,j)[2] = 100;
            }
            else {
                img.at<Vec3f>(i, j)[0] = 0;
                img.at<Vec3f>(i, j)[1] = 0;
                img.at<Vec3f>(i, j)[2] = 0;
            }
        }
    }



    cvtColor( img, img, COLOR_HSV2BGR );
    namedWindow("asdf1");
    namedWindow("asdf2");
    imshow("asdf1", inputOld);
    imshow("asdf2", inputNew);
    namedWindow("asdf3");
    moveWindow("asdf3", 100, 100);
    imshow("asdf3", img);

    cout << ioArray << endl;
    waitKey(10);

    denseflow.release();


    // cout << ioArray << endl;

#endif
}



/* ***********************
 *                       *
 *  Helper Functions     *
 *                       *
 *************************/


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



/*
 *  Calculate the perceived difference between two colors
 *  See https://en.wikipedia.org/wiki/Color_difference#CIE94 for the formula
 */
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

Mat  helperBGRKMean(Mat const &frame, int clusterCount, Mat &labels, Mat &centers) {

    Mat samples(frame.rows * frame.cols, 3, CV_32F);
    for( int y = 0; y < frame.rows; y++ ){
        for( int x = 0; x < frame.cols; x++ ) {
            for (int z = 0; z < 3; z++){
                samples.at<float>(y + x * frame.rows, z) = frame.at<Vec3b>(y, x)[z];
            }
        }
    }

    int attempts = 20;

    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );
    //    #define P2C_SHOW_KMEAN_WINDOW
// #define P2C_SHOW_KMEAN_WINDOW
        Mat new_image(frame.size(), frame.type());

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
        //
    return  new_image;

}

bool playerInRegionVector(FootballPlayer * fp, vector<Region> const & vr){

    for(Region const & r: vr) if (fp == r.playerInRegion) return true;

    return false;
}