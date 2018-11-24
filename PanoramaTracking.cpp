//
// Created by flo on 24.11.18.
//

#include "PanoramaTracking.h"

PanoramaTracking::PanoramaTracking(DetectorWrapper *detector, const char * tracker, Projector *projector) {

    this->detector = detector;
    this->trackerType = tracker;
    this->projector = projector;
    this->currentFrame = Mat();
}

void PanoramaTracking::trackVideo(const char *fileName) {

    int resolutionX = 1920;
    int resolutionY = 1080;
    Size resolution = Size(resolutionX, resolutionY);

    VideoCapture videoCapture(fileName);
    Mat resizedFrame;

    if(! videoCapture.isOpened()){
        std::cerr << "Could not open Video: " << fileName << std::endl;
        exit(-1);
    }
    videoCapture >> currentFrame;
    if(currentFrame.empty()) {
        std::cerr << "Empty video: " << fileName << std::endl;
        exit(-1);
    }


    const char * windowName = "PanoramaTracker";
    namedWindow(windowName);
    resize(currentFrame, resizedFrame, resolution);
    imshow(windowName, resizedFrame);

    while(! currentFrame.empty()){
        
        update();

        for(Rect const & r:panoramaAOI) rectangle(currentFrame, r, Scalar(0,0,255), 2);

        resize(currentFrame, resizedFrame, resolution);
        imshow(windowName, resizedFrame);
        waitKey(30);

        videoCapture >> currentFrame;

        
    }

}

bool PanoramaTracking::update() {

    panoramaAOI.clear();

    int numFrameProjections = projector->beginProjection();
    Mat projection;
    std::vector<Rect> detectedRects;
    std::vector<Ptr<Tracker>> trackerOnProjection;

    Rect2d  newPosition, panoramaPostion;
    bool objectFoundAgain;


    for (int projectionId = 0; projectionId < numFrameProjections; ++projectionId) {

        projector->project(currentFrame, projection);

        detectedRects = detector->detect(projection);
        trackerOnProjection =  projectionIdMapping[projectionId];

        // Update Tracker
        for(Ptr<Tracker> t: trackerOnProjection){
            objectFoundAgain = t->update(projection, newPosition);
            if(objectFoundAgain){
                panoramaPostion = projector->sourceCoordinates(currentFrame, newPosition, projectionId);
                panoramaAOI.emplace_back(Rect(panoramaPostion));

                // Erase the already known detections from detected rects;
                auto detected = detectedRects.begin();
                for(; detected != detectedRects.end(); ++detected){

                    if((Rect2d(*detected) & newPosition).area() > 0){
                        break;
                    }

                }
                if(detected != detectedRects.end()){
                    detectedRects.erase(detected);
                }
            }
            else{
                std::cout << "Lost sight of Object!" << std::endl;
            }



        }

        for(Rect const & newDetection : detectedRects){
            createNewTracker(projection, newDetection, projectionId);
            panoramaPostion = projector->sourceCoordinates(currentFrame, newPosition, projectionId);
            panoramaAOI.emplace_back(Rect(panoramaPostion));
        }



    }
    
    return false;
}

void PanoramaTracking::createNewTracker(Mat const &projection, Rect const &coordinates, int projectionId) {
    const char * name  = trackerType;

    Ptr<Tracker> newTracker;
    if(strcmp("KCF", name) == 0){
        newTracker = TrackerKCF::create();
    }
    else if(strcmp("Boosting", name) == 0){
        newTracker =  TrackerBoosting::create();
    }
    else if(strcmp("GOTURN", name) == 0){
        newTracker = TrackerGOTURN::create();
    }
    else if(strcmp(name ,"MedianFlow") == 0) {
        newTracker = TrackerMedianFlow::create();
    }
    else if(strcmp(name , "MIL") == 0) {
        newTracker = TrackerMIL::create();
    }
    else if(strcmp(name ,"MOSSE") == 0) {
        newTracker = TrackerMOSSE::create();
    }
    else if(strcmp(name, "TLD")==0) {
        newTracker = TrackerTLD::create();
    }
    else{
        std::cerr<< "No Matching Tracker Found" << std::endl;
        exit(-1);
    }

    newTracker->init(projection, coordinates);

    trackers.emplace_back(newTracker);
    projectionIdMapping[projectionId].emplace_back(newTracker);


}
