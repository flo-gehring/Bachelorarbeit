//
// Created by flo on 24.11.18.
//

#ifndef PANORAMA2CUBEMAP_PANORAMATRACKINGIMPLEMENTATIONS_H
#define PANORAMA2CUBEMAP_PANORAMATRACKINGIMPLEMENTATIONS_H

#include "PanoramaTracking.h"
#include "detect.h"

class CubeMapProjector : public Projector{
public:

    CubeMapProjector();

    int beginProjection() override;

    int project(Mat const &input, Mat &output) override;

    void project(Mat const &input, int projectionId, Mat &output) override;

    Rect sourceCoordinates(Mat const &input, Rect const &coordinates, int projectionNumber) override;

};

class YOLOWrapper : public DetectorWrapper{
public:
    YOLOWrapper() = default;
    std::vector<Rect> detect(Mat const &input) override;
private:
    MatDetector matDetector;

};

#endif //PANORAMA2CUBEMAP_PANORAMATRACKINGIMPLEMENTATIONS_H
