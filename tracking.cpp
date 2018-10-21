//
// Created by flo on 15.10.18.
//

#include "tracking.h"



/*************************
 * Custom Multi Tracker  *
 *************************/


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

    Rect2d rectOnPanorama, darknetPredRect;


    for (int face_id = 0; face_id < 6; ++face_id) {

        createCubeMapFace(frame, face, face_id, side_length, side_length);
        multiTracker[face_id].update(face);

        darknetDetector.detect_and_display(face);

        // Draw Boundaries of Cubefaces on Panorama
        Rect2d cube;
        const Rect2d asdf(0, 0, side_length, side_length);

        mapRectangleToPanorama(frame, face_id, side_length, side_length, asdf, cube);
        rectangle(frame, cube, Scalar(0,0,0), 2, 1);

        while(! darknetDetector.found.empty()){
            AbsoluteBoundingBoxes  current_prediction = darknetDetector.found.back();

            // void mapRectangleToPanorama(Mat & inFrame,  int faceId,  int width,  int height,const Rect2d & inRect, Rect2d & outRect );
            mapRectangleToPanorama(frame, face_id, side_length, side_length, current_prediction.rect, darknetPredRect);

            rectangle(frame, darknetPredRect, Scalar(0, 0, 255));

            darknetDetector.found.pop_back();


        }

        for (unsigned i = 0; i < multiTracker[face_id].getObjects().size(); i++) {
            mapRectangleToPanorama(frame, face_id, side_length, side_length,
                                   multiTracker[face_id].getObjects()[i],
                                   rectOnPanorama);

            rectangle(frame, rectOnPanorama, Scalar(255, 0, 0), 2, 1);

        }

    }

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

#ifdef UNDEF

/********************
 *                  *
 *  DarknetTracker  *
 *                  *
 ********************/

void DarknetTracker::initialize(Mat &frame) {


    Mat face;


    std::vector<TrackedObject> trackedObjects;
    allTrackedObjects.emplace_back(vector<TrackedObject>());
    vector<TrackedObject> & objectsCurrentFrame = allTrackedObjects.back();
    int objectCount = 0;

    detectObjects(frame, objectsCurrentFrame);

    for(auto it = objectsCurrentFrame.begin(); it != objectsCurrentFrame.end(); ++it){

        (* it).identifier = to_string(objectCount);
        ++objectCount;

    }

    maxNumberObjects = objectCount;
}

int DarknetTracker::detectObjects(Mat & frame, vector<TrackedObject> &objects) {
    Mat face;
    int numberDetectdObjects = 0;

    Rect2d objectInPanorama;

    for (int face_id = 0; face_id < 6; ++face_id){

        createCubeMapFace(frame, face, face_id, sideLength, sideLength);
        darknetDetector.detect_and_display(face);

        // algorithms[face_id] =  std::vector<Ptr<Tracker>>;

        for(auto it = darknetDetector.found.begin();
            it != darknetDetector.found.end(); ++it){

            objects.emplace_back(TrackedObject());

            objects.back().currentFaceId = face_id;
            objects.back().frames.push_back(0);

            objects.back().occurences.push_back(*it);
            objects.back().histogramm = calcHistForRect(face, (*it).rect);

            mapRectangleToPanorama(frame, face_id, sideLength, sideLength,
                    objects.back().occurences.back().rect, objectInPanorama);

            objects.back().occurences.back().rect = Rect(objectInPanorama);


            ++numberDetectdObjects;
        }

    }
    return numberDetectdObjects;
}

/*
 *  Draw the tracked Objects present in the frame corresponding to frameNum.
 *  No Detection will be done, so if this frame was not scanned already, nothing will be drawn.
 *
 *  Frame should be the full Equirectangular Panorama.
 */
void DarknetTracker::drawObjects(Mat & frame, int frameNum){

    Rect2d objectBox;
    AbsoluteBoundingBoxes absBB;
    int idx;
    vector<TrackedObject &> objectsInFrame = allTrackedObjects.at(frameNum);

    for (auto it = objectsInFrame.begin(); it != objectsInFrame.end(); ++it){

        // Get index of occurence for frame. We can assume that frameNum occurs in (*it).frames
        idx = ( * find((* it).frames.begin(), (*it).frames.end(), frameNum));
        absBB = (*it).occurences[idx];

        mapRectangleToPanorama(frame, (*it).currentFaceId, sideLength, sideLength, absBB.rect, objectBox);

        rectangle(frame, objectBox, Scalar(0, 0, 255), 2 ,1);
        putText(frame, (*it).identifier, Point(objectBox.x, objectBox.y),
                FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0,0 ,250), 1, CV_AA);
    }

}

Mat DarknetTracker::calcHistForRect(Mat inputImage, Rect rectangle) {

    Mat hsv, relevantRectangle;
    relevantRectangle = inputImage(rectangle);

    cvtColor(relevantRectangle, hsv, COLOR_BGR2HSV);
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
    MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};
    calcHist( &hsv, 1, channels, Mat(), // do not use mask
              hist, 2, histSize, ranges,
              true, // the histogram is uniform
              false );

    return hsv;



}

void DarknetTracker::update(Mat &frame) {
    vector<TrackedObject> objectsInFrame;
    detectObjects(frame, objectsInFrame);

    vector<TrackedObject> objectsLastFrame = allTrackedObjects.back();

    vector< TrackedObject &> intersectionsOld, intersectionsNew;

    Rect bbOldObject, bbNewObject;
    for(auto newObject = objectsInFrame.begin(); newObject != objectsInFrame.end(); ++newObject){

        intersectionsOld.clear();

        // Check with which objects in the old frame does the new Object intersect
       getIntersections(*newObject, objectsLastFrame, intersectionsOld);

        // Check for every with which the newObject intersects
        // (intersectionsOld) with how many new Object they intersect
        if(intersectionsOld.size() == 1 ){
            TrackedObject  asdf =  intersectionsOld.back();
            getIntersections(* intersectionsOld.back(), objectsInFrame, intersectionsNew);

            if(intersectionsNew.size() == 1 ){ // the new Object is the old Object. Found!
                (* newObject).identifier = intersectionsOld[0].identifier;

            }
        }
        else if(intersectionsOld.size() == 0){ // Check if new Object or just not in the last Frame

        }

        else { // New Object intersects with Multiple old Objects

        }



    }

}
void DarknetTracker::getIntersections(TrackedObject trackedObject, vector<TrackedObject> & possibleIntersections,
                                       vector<TrackedObject &> & intersections )  {

    Rect bbTrackedObject = trackedObject.occurences.back().rect;
    Rect bbToCompare;

    for (auto toCompare = possibleIntersections.begin(); toCompare != possibleIntersections.end(); ++toCompare){


        bbToCompare =  (* toCompare).occurences.back().rect;

        if( (bbToCompare & bbTrackedObject).area() > 0){ //
            intersections.push_back(* toCompare);
        }
    }
}

int DarknetTracker::track_video_stream(char *filename) {
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
    initialize(frame);
    drawObjects(frame, 0);
    resize(frame, resized_frame, Size(1980, 1020));
    imshow(windowName, resized_frame);
    waitKey();


    while(! frame.empty()){
        update(frame);

        imshow(windowName, resized_frame);

        char c = (char) waitKey();
        if (c == 27) break; // Press Esc to skip a current cubeface
        else if(c == 113){ // Press Q to leave Application
            return 2;
        }

        videoCapture >> frame;

    }
    return 0;
}

#endif
