//
// Created by flo on 30.12.18.
//

#include "testMaskRCNN.h"

#include "../RegionTracker/tracking.h"


    vector<string> classes;
    vector<Scalar> colors;
    float confThreshold, maskThreshold;

    vector<Rect> detectionsOnFrame;
    vector<int> detectionsOnProjectionId;
    vector<float> confScores;

void detectOnVideo(const char *filePath, Projector *projector, FILE * outfile, int skipToFrame) {

        classes.clear();
        colors.clear();

        colors = {Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 255, 255), Scalar(255, 102, 255)};
        confThreshold = 0.5;
        maskThreshold = 0.8;

        char *windowName = "MaskRCNN";
        namedWindow(windowName);

        VideoCapture videoCapture(filePath);

        Mat frame, blob;
        videoCapture >> frame;

        std::string directory = "./Detectors/mask_rcnn_inception_v2_coco_2018_01_28/";

        // Load names of classes
        std::string classesFile = directory + "ms_coco.names";
        std::ifstream ifs(classesFile.c_str());
        std::string line;

        vector<string> classes;

        // Give the configuration and weight files for the model
        String textGraph = directory + "mask_rcnn_inception_v2_coco_2018__01_28.pbtxt";
        String modelWeights = directory + "frozen_inference_graph.pb";

    // Load the network
        Net net = readNetFromTensorflow(modelWeights, textGraph);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        while (getline(ifs, line)) classes.push_back(line);

        int frameCounter = 0;

        for(; frameCounter < skipToFrame  && ! frame.empty(); ++frameCounter){
            videoCapture >> frame;
        }

        while (!frame.empty()) {

            detectionsOnFrame.clear();
            detectionsOnProjectionId.clear();
            confScores.clear();

            Mat projectedImage;
            int numOfProjections = projector->beginProjection();
            for (int projectionIndex = 0; projectionIndex < numOfProjections; ++projectionIndex) {

                projector->project(frame, projectedImage);
                // Create a 4D blob from a frame.
                blobFromImage(projectedImage, blob, 1.0, Size(projectedImage.cols, projectedImage.rows), Scalar(), true, false);

                //Sets the input to the network
                net.setInput(blob);

                // Runs the forward pass to get output from the output layers
                std::vector<String> outNames(2);
                outNames[0] = "detection_out_final";
                outNames[1] = "detection_masks";
                vector<Mat> outs;
                net.forward(outs, outNames);

                // Extract the bounding box and mask for each of the detected objects
                postprocess(projectedImage, outs);

                cout << "Processed " << projectionIndex + 1 << " projections" << endl;



                // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
                vector<double> layersTimes;
                double freq = getTickFrequency() / 1000;
                double t = net.getPerfProfile(layersTimes) / freq;
                string label = format("Mask-RCNN : Inference time for a frame : %.2f ms", t);
                putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

                // Write the frame with the detection boxes
                Mat detectedFrame;
                projectedImage.convertTo(detectedFrame, CV_8U);

                imshow(windowName, projectedImage);
                waitKey(30);

            }

            if(outfile){
                int numOfProjections = projector->beginProjection();
                int detectionIndex = 0;

                vector<Rect> projectedDetections;

                for(int  projectionIndex = 0; projectionIndex < numOfProjections; ++projectionIndex){

                    for(int i = 0; i < detectionsOnProjectionId[projectionIndex]; ++i){

                        projectedDetections.emplace_back(
                                projector->sourceCoordinates(frame, detectionsOnFrame[detectionIndex], projectionIndex)
                                );
                        ++detectionIndex;
                    }

                }

                Rect currentBB;
                string listOfDetections = "";
                for(int i = 0; i < confScores.size(); ++i){
                        // Generate List of detections

                        currentBB = projectedDetections[i];

                        listOfDetections.append(
                                std::string("{ \"x\": " )+  to_string(currentBB.x) +  std::string(", \"y\":" ) + to_string(currentBB.y) + std::string(", \"width\": " )+
                                to_string(currentBB.width)+ std::string( ", \"height\": ") + to_string( currentBB.height) + + ", \"confidence\": " + to_string(confScores[i]) +
                                std::string( "},\n")
                        );
                    }
                    listOfDetections = listOfDetections.substr(0, listOfDetections.length() - 2);


                    fprintf(outfile,
                            "{ \"frame\":%i, \n \"detections\": [ %s] }, ", frameCounter, listOfDetections.c_str());


                // printDetectionsToFile(outfile, frameCounter, projectedDetections);

            }
            videoCapture >> frame;
            ++frameCounter;



        }

    }

// For each frame, extract the bounding box and mask for each detected object
    void postprocess(Mat &frame, const vector<Mat> &outs) {
        Mat outDetections = outs[0];
        Mat outMasks = outs[1];

        // Output size of masks is NxCxHxW where
        // N - number of detected boxes
        // C - number of classes (excluding background)
        // HxW - segmentation shape
        const int numDetections = outDetections.size[2];
        const int numClasses = outMasks.size[1];

        detectionsOnProjectionId.push_back(0);

        outDetections = outDetections.reshape(1, outDetections.total() / 7);
        for (int i = 0; i < numDetections; ++i) {
            float score = outDetections.at<float>(i, 2);
            if (score > 0) {
                // Extract the bounding box
                int classId = static_cast<int>(outDetections.at<float>(i, 1));
                int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
                int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
                int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
                int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

                left = max(0, min(left, frame.cols - 1));
                top = max(0, min(top, frame.rows - 1));
                right = max(0, min(right, frame.cols - 1));
                bottom = max(0, min(bottom, frame.rows - 1));
                Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

                if(classId == 0){
                    detectionsOnFrame.emplace_back(Rect(box));
                    detectionsOnProjectionId.back() = detectionsOnProjectionId.back() + 1;
                    confScores.push_back(score);

                    // Extract the mask for the object
                    Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));

                    // Draw bounding box, colorize and show the mask on the image
                    drawBox(frame, classId, score, box, objectMask);
                }

            }
        }
    }

// Draw the predicted bounding box, colorize and show the mask on the image
    void drawBox(Mat &frame, int classId, float conf, Rect box, Mat &objectMask) {
        //Draw a rectangle displaying the bounding box
        rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);

        //Get the label for the class name and its confidence
        string label = format("%.2f", conf);
        if (!classes.empty()) {
            CV_Assert(classId < (int) classes.size());
            label = classes[classId] + ":" + label;
        }

        //Display the label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        box.y = max(box.y, labelSize.height);
        rectangle(frame, Point(box.x, box.y - round(1.5 * labelSize.height)),
                  Point(box.x + round(1.5 * labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
        putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

        Scalar color = colors[classId % colors.size()];

        // Resize the mask, threshold, color and apply it on the image
        resize(objectMask, objectMask, Size(box.width, box.height));
        Mat mask = (objectMask > maskThreshold);
        Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
        coloredRoi.convertTo(coloredRoi, CV_8UC3);

        // Draw the contours on the image
        vector<Mat> contours;
        Mat hierarchy;
        mask.convertTo(mask, CV_8U);
        findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        drawContours(coloredRoi, contours, -1, color, 1, LINE_8, hierarchy, 100);
        coloredRoi.copyTo(frame(box), mask);

    }
