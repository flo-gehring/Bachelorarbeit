//
// Created by flo on 24.11.18.
//

#include "PanoramaTrackingImplementations.h"
#include "cubetransform.h"


int CubeMapProjector::beginProjection() {
    currentProjectionId = 0;
    return 6;
}

int CubeMapProjector::project(Mat const &input, Mat & output) {

    project(input, currentProjectionId, output);

    return ++currentProjectionId;
}

void CubeMapProjector::project(Mat const &input, int projectionId, Mat &output) {

    createCubeMapFace(input, output, projectionId,projectionWidth, projectionHeight);

}

Rect CubeMapProjector::sourceCoordinates(Mat const &input, Rect const &coordinates, int projectionNumber) {
    Rect toReturn;
    mapRectangleToPanorama(input, projectionNumber, projectionWidth, projectionHeight, coordinates, toReturn);
    return toReturn;
}

CubeMapProjector::CubeMapProjector() {
    projectionWidth = 500;
    projectionHeight = 500;
}

std::vector<Rect> YOLOWrapper::detect(Mat const &input) {
    matDetector.detect_and_display(input);
    std::vector<Rect> toReturn;

    for(AbsoluteBoundingBoxes const & abb: matDetector.found){
        toReturn.emplace_back(Rect(abb.rect));
    }
    return toReturn;
}


