//
// Created by flo on 28.10.18.
//

#include "opencv_tracking.h"

vector<Ptr<Tracker>> trackingAlgorithms;

void trackVideo(char *videoPath, string const &  outFile,  string  const & tracker, MatDetector darknetDetector) {



    // Objects on video.
    vector<Rect> objects;
    Mat frame;
    Mat cubeFace;

    VideoCapture videoCapture(videoPath);




    Rect coordinatesOnPanorama;
    MultiTracker multiTracker;
    // Get the objects

    videoCapture >> frame;



    VideoWriter vw(outFile, VideoWriter::fourcc('M', 'J', 'P', 'G'),
                   videoCapture.get(CAP_PROP_FPS),
                   frame.size(), true);

    for(int faceId = 0; faceId < 6; ++faceId){

        createCubeMapFace(frame, cubeFace, faceId, 500, 500);
        darknetDetector.detect_and_display(cubeFace);

        for(auto  const & detectedObject : darknetDetector.found){

            mapRectangleToPanorama(frame, faceId, 500,500, detectedObject.rect, coordinatesOnPanorama);
            objects.emplace_back(Rect(coordinatesOnPanorama));
            createTrackerByName(tracker);

            multiTracker.add(trackingAlgorithms.back(), frame, coordinatesOnPanorama);

        }
    }

    int frameCounter = 0;

    time_t timeStart = time(0);
    float avgFrameRate = 0;
    float secondsPassed;

    while(! frame.empty()){

        multiTracker.update(frame);
        cout << "Updated Frame " << frameCounter << endl;
        ++frameCounter;

        secondsPassed = (time(0) - timeStart);

        if(secondsPassed != 0){
            avgFrameRate = float(frameCounter) / secondsPassed;
        }

        putText(frame,"FPS: " + to_string(avgFrameRate), Point(0,50), FONT_HERSHEY_PLAIN, 4, Scalar(0,0,255), 2);

        putText(frame, "Tracker: " + tracker, Point(0, 100), FONT_HERSHEY_PLAIN, 4, Scalar(0,0,255), 2);


        // draw the tracked object
        for(unsigned i=0;i<multiTracker.getObjects().size();i++)
            rectangle( frame, multiTracker.getObjects()[i], Scalar( 255, 0, 0 ), 2, 1 );


        vw.write(frame);
        videoCapture >> frame;



    }

}

void createTrackerByName(string name) {

    if(name == "KCF"){
        trackingAlgorithms.emplace_back(TrackerKCF::create());
    }
    if(name == "Boosting"){
        trackingAlgorithms.emplace_back(TrackerBoosting::create());
    }
    if(name == "GOTURN"){
        trackingAlgorithms.emplace_back(TrackerGOTURN::create());
    }
    if(name=="MedianFlow") {
        trackingAlgorithms.emplace_back(TrackerMedianFlow::create());
    }
    if(name == "MIL") {
        trackingAlgorithms.emplace_back(TrackerMIL::create());
    }
    if(name == "MOSSE") {
        trackingAlgorithms.emplace_back(TrackerMOSSE::create());
    }
    if(name == "TLD") {
        trackingAlgorithms.emplace_back(TrackerTLD::create());
    }


}
