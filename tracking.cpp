//
// Created by flo on 15.10.18.
//

#include "tracking.h"

CustomMultiTracker::CustomMultiTracker(){
    for(int i = 0; i < 6; ++i){
        multiTracker[i] =  * (new MultiTracker());
    }
}



void CustomMultiTracker::initialize_darknet(Mat &frame) {

    // darknetDetector.detect_and_display(frame);
    Mat face;
    int side_length = 500;

    for (int face_id = 0; face_id < 6; ++face_id){
        createCubeMapFace(frame, face, face_id, side_length, side_length);
        darknetDetector.detect_and_display(face);

       // algorithms[face_id] =  std::vector<Ptr<Tracker>>;

        for(auto it = darknetDetector.found.begin();
            it != darknetDetector.found.end(); ++it){
            algorithms[face_id].push_back(TrackerKCF::create());
            objects[face_id].push_back((*it).rect);
        }

        multiTracker[face_id].add(algorithms[face_id], frame, objects[face_id]);

    }


}

void CustomMultiTracker::update(Mat &frame) {
    Mat face;
    int side_length = 500;

    Rect2d rectOnPanorama;

    for (int face_id = 0; face_id < 6; ++face_id) {

        createCubeMapFace(frame, face, face_id, side_length, side_length);
        multiTracker[face_id].update(face);

        for (unsigned i = 0; i < multiTracker[face_id].getObjects().size(); i++) {
            mapRectangleToPanorama(frame, face_id, side_length, side_length,
                                   multiTracker[face_id].getObjects()[i],
                                   rectOnPanorama);

            rectangle(frame, rectOnPanorama, Scalar(255, 0, 0), 2, 1);

        }

    }
    //                      left  top
    //                      x     y   width height
    rectangle(frame, Rect2d(100,  0, 100, 100), Scalar(0, 255, 0), 2, 1);
}

int CustomMultiTracker::track_video_stream(char *filename) {
    VideoCapture videoCapture(filename);
    Mat frame, resized_frame;
    if (! videoCapture.isOpened()){
        return -1;
    }

    const char * windowName = "Window";

    namedWindow(windowName);
    videoCapture >> frame;

    if(frame.empty()){
        return -1;
    }
    initialize_darknet(frame);

    while(! frame.empty()){
        update(frame);
        resize(frame, resized_frame, Size(1980, 1020));
        imshow(windowName, resized_frame);

        char c = (char) waitKey(30);
        if (c == 27) break; // Press Esc to skip a current cubeface
        else if(c == 113){ // Press Q to leave Application
            return 2;
        }

        videoCapture >> frame;

    }
    return 0;




}