#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <math.h>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

#ifndef OPENCV
#define OPENCV
#endif

#include "/home/flo/Workspace/darknet/include/darknet.h"

#include "detect.cpp"

// Can't use namespace std because of naming conflicts
using namespace cv;
using namespace dnn;
inline void createCubeMapFace(const Mat &in, Mat &face,
                              int faceId = 0, const int width = -1,
                              const int height = -1);

// Functions from Example
void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

std::vector<String> getOutputsNames(const Net& net);
/*
 * See https://stackoverflow.com/questions/29678510/convert-21-equirectangular-panorama-to-cube-map
 * for how i got to this solution.
*/


float confThreshold = 0.5;
float nmsThreshold = 0.5;
std::vector<String> classesVec;

int main(int argc, char *argv[]) {
    std::cout << getBuildInformation() << std::endl;

    Net net = readNetFromDarknet("/home/flo/Workspace/darknet/cfg/yolov3.cfg",
            "/home/flo/Workspace/darknet/yolov3.weights");
    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
      //https://pjreddie.com/darknet/yolo/" << endl;
        exit(-1);
    }
    if (argc != 2)
    {
        std::cout << "Not enough parameters" << std::endl;
        return -1;
    }
    std::stringstream conv;
    const std::string video_path = argv[1];

    int frameNum = -1;          // Frame counter
    int inpWidth = 416;
    int inpHeight = inpWidth;


    const char* WIN_VID = "Video";

    // Windows
    namedWindow(WIN_VID, WINDOW_AUTOSIZE);

    // cout << video_capture.get(CAP_PROP_FORMAT);

    Mat frameReference;
    Mat_<Vec3b> resized_frame(Size(2000, 1000), Vec3b(255,0,0));
    Mat inputBlob, detectionMat;


    std::ifstream classNamesFile( "/home/flo/Workspace/darknet/data/coco.names");
    if (classNamesFile.is_open())
    {
        std::string className = "";
        while (std::getline(classNamesFile, className))
            classesVec.push_back(className);
    }


    // Get First Frame, next at the end of the for loop.

    for (char face_id = 0; face_id < 6;  ++face_id) {

        VideoCapture video_capture(video_path);

        if (!video_capture.isOpened())
        {
            std::cout  << "Could not open reference " << video_path << std::endl;
            return -1;
        }
        video_capture >> frameReference;
        waitKey(30);


        for (;;) //Show the image captured in the window and repeat
        {


            if (frameReference.empty()) {
                std::cout << "Face " << int(face_id) << "shown" << std::endl;
                break;
            }
            ++frameNum;

            createCubeMapFace(frameReference, resized_frame, face_id, 416, 416);

            inputBlob = blobFromImage(resized_frame, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
            net.setInput(inputBlob, "data");
            std::vector<Mat> outs;
            net.forward(outs, getOutputsNames(net));
            postprocess(resized_frame, outs, net);

            // Put efficiency information.
            std::vector<double> layersTimes;
            double freq = getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            std::string label = format("Inference time: %.2f ms", t);
            putText(resized_frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

            imshow(WIN_VID, resized_frame);

            char c = (char) waitKey(20);
            if (c == 27) break;

            // Get next Frame
            video_capture >> frameReference;

        }
    }
    return 0;
}

void outImgToXYZ(int i, int j, char face, int edge,
        double & x, double & y, double & z) {
    double a,b;

    a = 2.0 * i / edge;
    b = 2.0 * j / edge;

    switch (face){
        case 0: // back
            x = -1;
            y = 1 - a;
            z = 3 -b;
            break;
        case 1:// left
            x = a - 3;
            y = -1;
            z = 3 - b;
            break;
        case 2: // front
            x = 1;
            y = a -5;
            z = 3 -b;
            break;
        case 3: // right
            x = 7 -a;
            y = 1;
            z = 3 -b;
            break;
        case 4: // top
            x = b -1;
            y =  a - 5;
            z = 1;
            break;
        case 5: // bottom
            x = 5 -b;
            y = a -5;
            z = -1;
            break;
        default:
            std::cerr << "Error in outImgToXYZ, there are only 6 faces to a cube (0 to 5) ";
            exit(-1);

    }


}

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() == 1);
        float* data = (float*)outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > confThreshold)
            {
                int left = (int)data[i + 3];
                int top = (int)data[i + 4];
                int right = (int)data[i + 5];
                int bottom = (int)data[i + 6];
                int width = right - left + 1;
                int height = bottom - top + 1;
                classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }
    else if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() == 1);
        float* data = (float*)outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > confThreshold)
            {
                int left = (int)(data[i + 3] * frame.cols);
                int top = (int)(data[i + 4] * frame.rows);
                int right = (int)(data[i + 5] * frame.cols);
                int bottom = (int)(data[i + 6] * frame.rows);
                int width = right - left + 1;
                int height = bottom - top + 1;
                classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classesVec.empty())
    {
        CV_Assert(classId < (int)classesVec.size());
        label = classesVec[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

float faceTransform[6][2] =
        {
                {0, 0},
                {M_PI / 2, 0},
                {M_PI, 0},
                {-M_PI / 2, 0},
                {0, -M_PI / 2},
                {0, M_PI / 2}
        };

inline void createCubeMapFace(const Mat &in, Mat &face,
                              int faceId, const int width,
                              const int height) {

    float inWidth = in.cols;
    float inHeight = in.rows;

    // Allocate map
    Mat mapx(height, width, CV_32F);
    Mat mapy(height, width, CV_32F);

    // Calculate adjacent (ak) and opposite (an) of the
    // triangle that is spanned from the sphere center
    //to our cube face.
    const float an = sin(M_PI / 4);
    const float ak = cos(M_PI / 4);

    const float ftu = faceTransform[faceId][0];
    const float ftv = faceTransform[faceId][1];

    // For each point in the target image,
    // calculate the corresponding source coordinates.
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {

            // Map face pixel coordinates to [-1, 1] on plane
            float nx = (float)y / (float)height - 0.5f;
            float ny = (float)x / (float)width - 0.5f;

            nx *= 2;
            ny *= 2;

            // Map [-1, 1] plane coords to [-an, an]
            // thats the coordinates in respect to a unit sphere
            // that contains our box.
            nx *= an;
            ny *= an;

            float u, v;

            // Project from plane to sphere surface.
            if(ftv == 0) {
                // Center faces
                u = atan2(nx, ak); //
                v = atan2(ny * cos(u), ak);
                u += ftu;
            } else if(ftv > 0) {
                // Bottom face
                float d = sqrt(nx * nx + ny * ny);
                v = M_PI / 2 - atan2(d, ak);
                u = atan2(ny, nx);
            } else {
                // Top face
                float d = sqrt(nx * nx + ny * ny);
                v = -M_PI / 2 + atan2(d, ak);
                u = atan2(-ny, nx);
            }

            // Map from angular coordinates to [-1, 1], respectively.
            u = u / (M_PI);
            v = v / (M_PI / 2);

            // Warp around, if our coordinates are out of bounds.
            while (v < -1) {
                v += 2;
                u += 1;
            }
            while (v > 1) {
                v -= 2;
                u += 1;
            }

            while(u < -1) {
                u += 2;
            }
            while(u > 1) {
                u -= 2;
            }

            // Map from [-1, 1] to in texture space
            u = u / 2.0f + 0.5f;
            v = v / 2.0f + 0.5f;

            u = u * (inWidth - 1);
            v = v * (inHeight - 1);

            // Save the result for this pixel in map
            mapx.at<float>(x, y) = u;
            mapy.at<float>(x, y) = v;
        }
    }

    // Recreate output image if it has wrong size or type.
    if(face.cols != width || face.rows != height ||
       face.type() != in.type()) {
        face = Mat(width, height, in.type());
    }

    // Do actual resampling using OpenCV's remap
    remap(in, face, mapx, mapy,
          INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
}
std::vector<String> getOutputsNames(const Net& net)
{
    static std::vector<String> names;
    if (names.empty())
    {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}