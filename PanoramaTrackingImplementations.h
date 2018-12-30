//
// Created by flo on 24.11.18.
//

#ifndef PANORAMA2CUBEMAP_PANORAMATRACKINGIMPLEMENTATIONS_H
#define PANORAMA2CUBEMAP_PANORAMATRACKINGIMPLEMENTATIONS_H

#include "PanoramaTracking.h"
#include "detect.h"


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class EquatorLine : public Projector{
public:
    EquatorLine() = default;
    void setUp(Size const & inputSize, int projectionWidth = 512, int projectionHeight = 512);
    EquatorLine(Size const & inputSize, int projectionWidth = 512, int projectionHeight = 512);
    int beginProjection() override;

    int project(Mat const &input, Mat &output) override;

    void project(Mat const &input, int projectionId, Mat &output) override;

    Rect sourceCoordinates(Mat const &input, Rect const &coordinates, int projectionNumber) override;

    int numberProjections;
    std::vector<Rect> projectedSections;
    int projectionsInHeight;
    int projectionsInWidth;

};

class SectionProjector : public Projector{
public:
    SectionProjector(Size const & inputSize, int projectionWidth = 512, int projectionHeight = 512);

    int beginProjection() override;

    int project(Mat const &input, Mat &output) override;

    void project(Mat const &input, int projectionId, Mat &output) override;

    Rect sourceCoordinates(Mat const &input, Rect const &coordinates, int projectionNumber) override;


    cv::Size inputSize;

    int numberProjections;
    std::vector<Rect> projectedSections;
    int projectionsInHeight;
    int projectionsInWidth;


};

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


class AOIFileDetectorWrapper: public DetectorWrapper{
public:
    explicit AOIFileDetectorWrapper(const char * aoiFile);
    std::vector<Rect> detect(Mat const & input) override;
    DetectionFromFile detector;



};

class MaskRCNN : public DetectorWrapper{
public:
    explicit MaskRCNN();
    std::vector<Rect> detect(Mat const & input) override;

    dnn::Net net; // The neural network doing all the work.
    std::vector<std::tuple<Rect, Mat>> boundingBoxAndMask;

    std::vector<std::tuple<Rect, Mat>> detectWithMask(Mat const & input);
};

#endif //PANORAMA2CUBEMAP_PANORAMATRACKINGIMPLEMENTATIONS_H
