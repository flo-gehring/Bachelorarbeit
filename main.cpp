#include <iostream> // for standard I/O
#include <string>   // for strings



#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include "cubetransform.h"
#include "opencv_detect.h"
#include "detect.h"

using namespace cv;
using namespace dnn;
using namespace std;

void show_on_cubefaces(YOLODetector yoloD, char * video_path);
void save_video_projection(YOLODetector yoloDetector, char* inPath, char* outPath);
void darknet_predictions(char* video_path);

int main(int argc, char *argv[]) {
    std::cout << getBuildInformation() << std::endl;



    if (!(argc == 6 || argc == 7))
    {
        cout << "Wrong number of parameters." << endl;
        cout << "Usage: ./Panorama2Cube show  <yolov3.cfg file> <yolov3.weight file> <coco.names file> <videofile>" << std::endl;
        cout << "./Panorama2Cube save <yolov3.cfg file> <yolov3.weight file> <coco.names file> <videofile> <outfile>" << std::endl;
        return -1;
    }
    stringstream conv;
    char * video_path = argv[5];

    YOLODetector yoloDetector(argv[2], argv[3], argv[4]);

    if(strcmp(argv[1], "save") == 0){
        char * outfile = argv[6];
        save_video_projection(yoloDetector, video_path, outfile);
    }
    else if(strcmp(argv[1], "show") == 0){
        show_on_cubefaces(yoloDetector, video_path);
    }
    else if(strcmp(argv[1], "darknet") == 0){
        darknet_predictions(video_path);
    }

    return 0;
}

void darknet_predictions(char * video_path){

    MatDetector matDetector;
    VideoCapture video(video_path);
    Mat frameReference, resizedFrame;
    int frameNum = 0;

    const char * WIN_VID = "Darknet Detection";
    namedWindow(WIN_VID, WINDOW_AUTOSIZE);

    for (char face_id = 0; face_id < 6;  ++face_id) {

        VideoCapture video_capture(video_path);

        if (!video_capture.isOpened())
        {
            std::cout  << "Could not open reference " << video_path << std::endl;
            return;
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

            createCubeMapFace(frameReference, resizedFrame, face_id, 500, 500);

            matDetector.detect_and_display(resizedFrame);

            while(! matDetector.found.empty()){
                AbsoluteBoundingBoxes current_box = matDetector.found.back();

                rectangle(resizedFrame,
                          current_box.rect,
                          Scalar(0,0,255));
                matDetector.found.pop_back();

            }
            imshow(WIN_VID, resizedFrame);
            char c = (char) waitKey(20);
            if (c == 27) break;

            // Get next Frame
            video_capture >> frameReference;

        }
    }
    return;
}

void save_video_projection(YOLODetector yoloDetector, char* inPath, char* outPath){
    Mat er_projection, resized_er;
    Mat_<Vec3b> cube_face(Size(2000, 1000), Vec3b(255,0,0));


    // show_on_cubefaces(yoloD, argv[1]);

    VideoCapture video_capture(inPath);
    if(! video_capture.isOpened()){
        cout  << "Could not open reference " << inPath << endl;
        return;
    }


    video_capture >> er_projection;

    VideoWriter vw(outPath, VideoWriter::fourcc('M', 'J', 'P', 'G'),
                   video_capture.get(CAP_PROP_FPS),
                   er_projection.size(), true);
    waitKey(30);

    float left, top, right, bottom = 0;
    float * left_ptr = &left;
    float * top_ptr = &top;
    float * right_ptr = & right;
    float  * bottom_ptr = & bottom;

    int frame_counter = 0;
    while(! er_projection.empty()){

        for(short face_id = 0; face_id < 6; ++face_id){
            createCubeMapFace(er_projection, cube_face, face_id, 416, 416);
            yoloDetector.detect(cube_face);

            while(! yoloDetector.predictions.empty()){
                prediction  current_prediction = yoloDetector.predictions.back();

                getPanoramaCoords(er_projection, face_id,  416, 416,
                                  current_prediction.top,  current_prediction.left,
                                  left_ptr, top_ptr);

                getPanoramaCoords(er_projection, face_id,  416, 416,
                                  current_prediction.bottom, current_prediction.right,
                                  right_ptr, bottom_ptr);

                rectangle(er_projection, Point(int(* left_ptr), int(* top_ptr)),
                          Point(int(* right_ptr), int(* bottom_ptr)), Scalar(0, 0, 255));

                yoloDetector.predictions.pop_back();


            }

            cout << "Face Side " << face_id << " done." << endl;


            char c = waitKey(30);
            if(c == 27) return;

        }
        ++ frame_counter;
        cout << "Frame " << frame_counter <<  " of " << video_capture.get(CAP_PROP_FRAME_COUNT) << " done." << endl;
        vw.write(er_projection);
        video_capture >> er_projection;

    }
}

void show_on_cubefaces(YOLODetector yoloD, char* video_path){
    const char* WIN_VID = "Video";

    // Windows
    namedWindow(WIN_VID, WINDOW_AUTOSIZE);

    // cout << video_capture.get(CAP_PROP_FORMAT);

    Mat frameReference;
    Mat_<Vec3b> resized_frame(Size(2000, 1000), Vec3b(255,0,0));




    // Get First Frame, next at the end of the for loop.
    for (char face_id = 0; face_id < 6;  ++face_id) {

        VideoCapture video_capture(video_path);

        if (!video_capture.isOpened())
        {
            cout  << "Could not open reference " << video_path << endl;
            return;
        }
        video_capture >> frameReference;
        waitKey(30);

        for (;;) //Show the image captured in the window and repeat
        {




            if (frameReference.empty()) {
                cout << "Face " << int(face_id) << "shown" << endl;
                break;
            }

            createCubeMapFace(frameReference, resized_frame, face_id, 416, 416);
            yoloD.detect(resized_frame);

            imshow(WIN_VID, resized_frame);

            char c = (char) waitKey(20);
            if (c == 27) break; // Press Esc to skip a current cubeface
            else if(c == 113){ // Press Q to leave Application
                return;
            }


            // Get next Frame
            video_capture >> frameReference;

        }
    }
}

inline void createCubeMapFace(const Mat &in, Mat &face,
                              int faceId, const int width,
                              const int height) {


    // Allocate map
    Mat mapx(height, width, CV_32F);
    Mat mapy(height, width, CV_32F);

    float u = 0;
    float v = 0;
    float * u_ptr = &u;
    float * v_ptr = &v;

    // For each point in the target image,
    // calculate the corresponding source coordinates.


    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {

            getPanoramaCoords(in, faceId, width,height, x,  y, u_ptr,  v_ptr);
            // Save the result for this pixel in map
            mapx.at<float>(x, y) = *u_ptr;
            mapy.at<float>(x, y) = *v_ptr;
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

inline void getPanoramaCoords(const Mat & in, int faceId, const int width, const int height,
                                     int x, int y,
                                     float * u_ptr, float* v_ptr){

    // Calculate adjacent (ak) and opposite (an) of the
    // triangle that is spanned from the sphere center
    //to our cube face.
    const float an = sin(M_PI / 4);
    const float ak = cos(M_PI / 4);

    const float ftu = faceTransform[faceId][0];
    const float ftv = faceTransform[faceId][1];

    float u,v;

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

    u = u * (in.cols - 1);
    v = v * (in.rows - 1);

    *u_ptr = u;
    *v_ptr = v;
}
