//
// Created by flo on 24.11.18.
//

#include "PanoramaTrackingImplementations.h"
#include "../cubetransform.h"


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
    projectionWidth = 608;
    projectionHeight = 608;
}

std::vector<Rect> YOLOWrapper::detect(Mat const &input) {
    matDetector.detect_and_display(input);
    std::vector<Rect> toReturn;

    for(AbsoluteBoundingBoxes const & abb: matDetector.found){
        toReturn.emplace_back(Rect(abb.rect));
    }
    return toReturn;
}


AOIFileDetectorWrapper::AOIFileDetectorWrapper(const char *aoiFile) {

    detector.loadAOI(aoiFile);
}

std::vector<Rect> AOIFileDetectorWrapper::detect(Mat const &input) {
    detector.detect_and_display(input);
    std::vector<Rect> toReturn;
    for (AbsoluteBoundingBoxes const & abb: detector.found){
        toReturn.emplace_back(Rect(abb.rect));
    }
    return toReturn;
}

SectionProjector::SectionProjector(Size const & inputSize, int projectionWidth , int projectionHeight) {

    projectionsInWidth = int(ceil(inputSize.width / double(projectionWidth)));
    projectionsInHeight = int(ceil(inputSize.height / double(projectionHeight)));

    numberProjections = projectionsInHeight * projectionsInWidth;
    currentProjectionId = 0;

    Rect section;
    int xCoordinate, yCoordinate, rectHeight, rectWidth;
    for(int widthOffset = 0; widthOffset < projectionsInWidth; ++widthOffset){

        for(int heightOffset = 0; heightOffset < projectionsInHeight; ++heightOffset){

            xCoordinate = widthOffset * projectionWidth;
            yCoordinate = heightOffset * projectionHeight;

            if(xCoordinate + projectionWidth <= inputSize.width){
                rectWidth = projectionWidth;
            }
            else{
                rectWidth = inputSize.width - xCoordinate;
            }


            if(yCoordinate + projectionHeight <= inputSize.height) {
                rectHeight = projectionHeight;
            }
            else{
                rectHeight = inputSize.height - yCoordinate;
            }

            projectedSections.emplace_back(Rect(xCoordinate, yCoordinate, rectWidth, rectHeight));

        }

    }
}

int SectionProjector::beginProjection() {
    currentProjectionId = 0;
    return numberProjections;
}

int SectionProjector::project(Mat const &input, Mat &output) {

    output  = input(projectedSections[currentProjectionId]);

    ++currentProjectionId;
    return currentProjectionId;
}

void SectionProjector::project(Mat const &input, int projectionId, Mat &output) {

    output = input(projectedSections[projectionId]);
}

Rect SectionProjector::sourceCoordinates(Mat const &input, Rect const &coordinates, int projectionNumber) {

    const Rect & projectionRect = projectedSections[projectionNumber];
    return cv::Rect(projectionRect.x + coordinates.x,
            projectionRect.y + coordinates.y,
            coordinates.width,
            coordinates.height);
}

int EquatorLine::beginProjection() {
    currentProjectionId  = 0;
    return numberProjections;
}

EquatorLine::EquatorLine(Size const &inputSize, int projectionWidth, int projectionHeight) {

    projectionsInHeight = 1;

    numberProjections = 0;

    int currentTopLeftX = 0;

    int sectionWidth;
    int topY = (inputSize.height / 2) - (projectionHeight / 2);
    while(currentTopLeftX < inputSize.width){

        sectionWidth = (currentTopLeftX + projectionWidth) >= inputSize.width ? inputSize.width - currentTopLeftX : projectionWidth;


        projectedSections.emplace_back(Rect(currentTopLeftX, topY, sectionWidth, projectionHeight));

        currentTopLeftX += projectionWidth;
        ++numberProjections;
    }

}

int EquatorLine::project(Mat const &input, Mat &output) {

    currentProjectionId = currentProjectionId % numberProjections;

    output = input(projectedSections[currentProjectionId]);


    ++currentProjectionId;
    return currentProjectionId;
}

void EquatorLine::project(Mat const &input, int projectionId, Mat &output) {

    output = input(projectedSections[projectionId]);

}

Rect EquatorLine::sourceCoordinates(Mat const &input, Rect const &coordinates, int projectionNumber) {
    const Rect & projectionRect = projectedSections[projectionNumber];
    return cv::Rect(projectionRect.x + coordinates.x,
                    projectionRect.y + coordinates.y,
                    coordinates.width,
                    coordinates.height);
}

CleanCubeMap::CleanCubeMap() {
    currentProjectionId = 0;

    maxProjection = 4;
}

int CleanCubeMap::beginProjection() {

    currentProjectionId = 0;
    return maxProjection;
}

int CleanCubeMap::project(Mat const &input, Mat &output) {

    getCubeSide(input, output, 608, currentProjectionId);
    ++currentProjectionId;
    return currentProjectionId;
}

void CleanCubeMap::project(Mat const &input, int projectionId, Mat &output) {
    getCubeSide(input, output, 608, projectionId);
}

Rect CleanCubeMap::sourceCoordinates(Mat const &input, Rect const &coordinates, int projectionNumber) {
    Size inputSize = input.size();
    float xtl, ytl, xbr, ybr;
    panoramaCoords(inputSize, projectionNumber, coordinates.x, coordinates.y, 608,  &xtl, &ytl);
    panoramaCoords(inputSize, projectionNumber, coordinates.x + coordinates.width, coordinates.y + coordinates.height,
            608,  &xbr, &ybr);


    return cv::Rect(int(xtl), int(ytl),int(xbr - xtl), int(ybr - ytl));
}
