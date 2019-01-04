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

    setUp(inputSize, projectionWidth, projectionHeight);

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

    assert(projectionRect.x + coordinates.x + coordinates.width <= input.cols && projectionRect.y + coordinates.y + coordinates.height <= input.cols);

    return cv::Rect(projectionRect.x + coordinates.x,
                    projectionRect.y + coordinates.y,
                    coordinates.width,
                    coordinates.height);
}

void EquatorLine::setUp(Size const &inputSize, int projectionWidth, int projectionHeight) {
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

std::vector<std::tuple<Rect, Mat>> MaskRCNN::detectWithMask(Mat const &input) {

    boundingBoxAndMask.clear();
    float confThreshold = 0.4;
    float maskThreshold = 0.8;
    Mat blob;
    // Create a 4D blob from a frame.
    dnn::blobFromImage(input, blob, 1.0, Size(input.cols, input.rows), Scalar(), true, false);

    //Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output from the output layers
    std::vector<String> outNames(2);
    outNames[0] = "detection_out_final";
    outNames[1] = "detection_masks";
    std::vector<Mat> outs;
    net.forward(outs, outNames);

    Mat outDetections = outs[0];
    Mat outMasks = outs[1];


    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];
    const int numClasses = outMasks.size[1];


    std::vector<Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> indices;

    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < numDetections; ++i) {
        float score = outDetections.at<float>(i, 2);
        int classId = static_cast<int>(outDetections.at<float>(i, 1));

        if (score > confThreshold && classId == 0) { // Only Detect Persons
            // Extract the bounding box
            int left = static_cast<int>(input.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(input.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(input.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(input.rows * outDetections.at<float>(i, 6));

            left = max(0, min(left, input.cols - 1));
            top = max(0, min(top, input.rows - 1));
            right = max(0, min(right, input.cols - 1));
            bottom = max(0, min(bottom, input.rows - 1));
            Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

            assert(bottom <= input.rows &&  right < input.cols );

            // Extract the mask for the object
            // Mask is always 15 * 15, so we need to resize it to the boxes width and  height.
            Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));

            resize(objectMask, objectMask, Size(box.width, box.height));
            Mat mask = (objectMask > maskThreshold);

            bboxes.push_back(box);
            scores.push_back(score);


            boundingBoxAndMask.emplace_back(std::make_pair(box, mask));

            // Draw bounding box, colorize and show the mask on the image

        }
    }

    dnn::NMSBoxes(bboxes, scores, confThreshold, 0.1, indices);


    std::vector<std::tuple<Rect, Mat>> passedNMS;

    for(int i: indices){
        passedNMS.emplace_back(boundingBoxAndMask[i]);
    }

    boundingBoxAndMask.swap(passedNMS);
    return boundingBoxAndMask;
}

std::vector<Rect> MaskRCNN::detect(Mat const &input) {

    detectWithMask(input);
    std::vector<Rect> toReturn;

    for(std::tuple<Rect, Mat> const & tuple : boundingBoxAndMask) toReturn.emplace_back(Rect(std::get<0>(tuple)));

    return toReturn;
}

MaskRCNN::MaskRCNN() {
    std::string directory = "./mask_rcnn_inception_v2_coco_2018_01_28/";

    // Load names of classes
    std::string classesFile = directory + "ms_coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;

    std::vector<std::string> classes;

    // Give the configuration and weight files for the model
    String textGraph = directory + "mask_rcnn_inception_v2_coco_2018__01_28.pbtxt";
    String modelWeights = directory + "frozen_inference_graph.pb";

    // Load the network
    net = dnn::readNetFromTensorflow(modelWeights, textGraph);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

}
