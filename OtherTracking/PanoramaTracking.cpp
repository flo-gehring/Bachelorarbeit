//
// Created by flo on 24.11.18.
//

#include "PanoramaTracking.h"

PanoramaTracking::PanoramaTracking(DetectorWrapper *detector, const char * tracker, Projector *projector) {

    this->detector = detector;
    this->trackerType = tracker;
    this->projector = projector;
    this->currentFrame = Mat();
    this->detectionUpdateIntervall = 30;
    this->frameCounter = 0;
    this->startTime = time(0);
}

void PanoramaTracking::trackVideo(const char *fileName, const char * videoFile) {

    Scalar textColor = Scalar(0,0,255); // Red
    double textSize = 1.5;
    int lineThickness = 1;

    VideoCapture videoCapture(fileName);
    Mat resizedFrame;
    if(! videoCapture.isOpened()){
        std::cerr << "Could not open Video: " << fileName << std::endl;
        exit(-1);
    }
    videoCapture >> currentFrame;

    startTime = time(0);

    VideoWriter videoWriter;
    if(videoFile){
        videoWriter = cv::VideoWriter(videoFile, VideoWriter::fourcc('M', 'J', 'P', 'G'),
                videoCapture.get(CAP_PROP_FPS),
               currentFrame.size(), true);
    }
    int resolutionX = 1920;
    int resolutionY = 1080;
    Size resolution = Size(resolutionX, resolutionY);




    if(currentFrame.empty()) {
        std::cerr << "Empty video: " << fileName << std::endl;
        exit(-1);
    }


    const char * windowName = "PanoramaTracker";

    if(!videoFile) {
        namedWindow(windowName);
        resize(currentFrame, resizedFrame, resolution);
        imshow(windowName, resizedFrame);
    }

    while(! currentFrame.empty()){
        
        update();

        for(std::tuple<Rect, Ptr<Tracker>> const & r :panoramaAOI){
            const Rect & rect = std::get<0>(r);
            const Ptr<Tracker> & ptrTracker = std::get<1>(r);

            std::string name =  objectIdentifier.find(ptrTracker)->second;
            rectangle(currentFrame, std::get<0>(r), Scalar(0,0,255), 2);
            cv::putText(currentFrame, name, Point(rect.x, rect.y - 10), FONT_HERSHEY_SIMPLEX, textSize , textColor, lineThickness);
        }

        double fps = (time(0) - startTime) / double(frameCounter);

        cv::putText(currentFrame, "FPS: " + std::to_string(fps), Point(10, 80), FONT_HERSHEY_SIMPLEX, 2, textColor, lineThickness);
        cv::putText(currentFrame, trackerType, Point(10, 160), FONT_HERSHEY_SIMPLEX, 2,  textColor, lineThickness);

        if(! videoFile) {
            resize(currentFrame, resizedFrame, resolution);
            imshow(windowName, resizedFrame);
            waitKey(30);
        }
        else{
            videoWriter << currentFrame;
        }
        videoCapture >> currentFrame;

    }

    videoWriter.release();

}

bool PanoramaTracking::update() {


    bool updateDetection = (frameCounter % detectionUpdateIntervall) == 0;

    panoramaAOI.clear();

    int numFrameProjections = projector->beginProjection();

    Mat projection;
    std::vector<Rect> detectedRects;
    std::vector<Ptr<Tracker>> trackerOnProjection;

    Rect2d  newPosition, panoramaPostion;
    bool objectFoundAgain;


    for (int projectionId = 0; projectionId < numFrameProjections; ++projectionId) {

        projector->project(currentFrame, projection);
        if(updateDetection)
            detectedRects = detector->detect(projection);

        trackerOnProjection =  projectionIdMapping[projectionId];

        // Update Tracker
        for(Ptr<Tracker> const & t: trackerOnProjection){
            objectFoundAgain = t->update(projection, newPosition);

            if(objectFoundAgain){
                panoramaPostion = projector->sourceCoordinates(currentFrame, newPosition, projectionId);
                panoramaAOI.emplace_back(std::make_pair(Rect(panoramaPostion), t));

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
            Ptr<Tracker> newTracker = trackers.back();
            std::string id = std::to_string(trackers.size());

            panoramaPostion = projector->sourceCoordinates(currentFrame, newPosition, projectionId);

            objectIdentifier[newTracker] = id;
            panoramaAOI.emplace_back(std::make_pair(Rect(panoramaPostion), newTracker));
        }

    }
    ++frameCounter;
    
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
