//
// Created by flo on 15.10.18.
//

#include "tracking.h"


void CustomMultiTracker::initialize_darknet(Mat &frame) {

    darknetDetector.detect_and_display(frame);




    for(auto it = darknetDetector.found.begin();
    it != darknetDetector.found.end(); ++it){
        algorithms.push_back(TrackerKCF::create());
        objects.push_back((*it).rect);
    }
    multiTracker.add(algorithms, frame, objects);

}

void CustomMultiTracker::update(Mat &frame) {
    multiTracker.update(frame);
    for(unsigned i=0;i<multiTracker.getObjects().size();i++)
        rectangle( frame, multiTracker.getObjects()[i], Scalar( 255, 0, 0 ), 2, 1 );

}