#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <math.h>

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


#define PI 3.14159

using namespace cv;
inline void createCubeMapFace(const Mat &in, Mat &face,
                              int faceId = 0, const int width = -1,
                              const int height = -1);



/*
 * See https://stackoverflow.com/questions/29678510/convert-21-equirectangular-panorama-to-cube-map
 * for how i got to this solution.
*/


int main(int argc, char *argv[]) {

    MatDetector matDetector;
    if (argc != 2)
    {
        std::cout << "Not enough parameters" << std::endl;
        return -1;
    }
    std::stringstream conv;
    const std::string video_path = argv[1];

    int frameNum = -1;          // Frame counter


    const char* WIN_VID = "Video";

    // Windows
    namedWindow(WIN_VID, WINDOW_AUTOSIZE);

    // cout << video_capture.get(CAP_PROP_FORMAT);

    Mat frameReference;
    Mat_<Vec3b> resized_frame(Size(2000, 1000), Vec3b(255,0,0));

    // Get First Frame, next at the end of the for loop.

    for (char face_id = 0; face_id < 6;  ++face_id) {

        VideoCapture video_capture(video_path);

        VideoCapture *cap_ptr = &video_capture;
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

            createCubeMapFace(frameReference, resized_frame, face_id, 500, 500);

            matDetector.detect_and_display(resized_frame);

            while(! matDetector.found.empty()){
                AbsoluteBoundingBoxes current_box = matDetector.found.top();

                rectangle(resized_frame,
                        Point2d(
                        current_box.left,
                        current_box.top),
                        Point2d(
                                current_box.right,
                                current_box.bottom
                                ),
                                Scalar(0,0,255));
                matDetector.found.pop();

            }
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
                u = atan2(nx, ak);
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
