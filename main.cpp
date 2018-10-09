#include <iostream> // for standard I/O
#include <string>   // for strings



#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O


#include "cubetransform.h"
#include "opencv_detect.h"

using namespace cv;
using namespace dnn;
using namespace std;

int main(int argc, char *argv[]) {
    // std::cout << getBuildInformation() << std::endl;


    if (argc != 5)
    {
        cout << "Wrong number of parameters." << endl;
        cout << "Usage: ./Panorama2Cube <videofile> <yolov3.cfg file> <yolov3.weight file> <coco.names file>" << std::endl;
        return -1;
    }
    stringstream conv;
    const string video_path = argv[1];

    YOLODetector yoloD(argv[2], argv[3], argv[4]);


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
            return -1;
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
            yoloD.detect_and_display(resized_frame);

            imshow(WIN_VID, resized_frame);

            char c = (char) waitKey(20);
            if (c == 27) break; // Press Esc to skip a current cubeface
            else if(c == 113){ // Press Q to leave Application
                return 0;
            }


            // Get next Frame
            video_capture >> frameReference;

        }
    }
    return 0;
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
