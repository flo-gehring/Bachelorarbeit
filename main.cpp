#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <math.h>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

#include "VideoFrameTransform.h"


#define PI 3.14159

using namespace std;
using namespace cv;
double getPSNR ( const Mat& I1, const Mat& I2);
Scalar getMSSIM( const Mat& I1, const Mat& I2);

/*
 * See https://stackoverflow.com/questions/29678510/convert-21-equirectangular-panorama-to-cube-map
 * for how i got to this solution.
*/

/*
 * Get x, y, z coords from out image pixel cords.
 * i, j are pixel coords
 * face is face number
 * edge is edge length.
 * x, y, z are coordinates in 3D Room
 */
void outImgToXYZ(int i, int j, char face, int edge, double & x, double & y, double & z);

/*
 * Convert an Equirectangular Panorama Image into an Cubemap
 */
void convertBack(Mat imgIn, Mat imgOut);

/*
 * Clips input to at least min or at most max
 */
int clip(int in, int min, int max);

int main(int argc, char *argv[])
{

    // cout << getBuildInformation() ;
    if (argc != 2)
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }
    stringstream conv;
    const string video_path = argv[1];

    int frameNum = -1;          // Frame counter
    VideoCapture video_capture(video_path);
    if (!video_capture.isOpened())
    {
        cout  << "Could not open reference " << video_path << endl;
        return -1;
    }

    const char* WIN_VID = "Video";

    // Windows
    namedWindow(WIN_VID, WINDOW_AUTOSIZE);

    Size video_refrence = Size((int) video_capture.get(CAP_PROP_FRAME_WIDTH),
                               (int) video_capture.get(CAP_PROP_FRAME_HEIGHT));

    // cout << video_capture.get(CAP_PROP_FORMAT);

    cout << "Reference frame resolution: Width=" << video_refrence.width << "  Height=" << video_refrence.height
         << " of nr#: " << video_capture.get(CAP_PROP_FRAME_COUNT) << endl;

    Mat frameReference;
    double psnrV;
    Scalar mssimV;

    Mat_<Vec3b> resized_frame(Size(2000, 1000), Vec3b(255,0,0));

    // Get First Frame, next at the end of the for loop.
    video_capture >> frameReference;
    waitKey(30);
    CV_Assert(frameReference.depth() == CV_8U);
    cout << "Type: " << frameReference.type() << endl;
    cout << "Video Frame Is Continous: " << frameReference.isContinuous() << endl;
    cout << "New Frame is Continous: " << resized_frame.isContinuous() << endl;
    CV_Assert(frameReference.isContinuous());
    cout << "Mat size rows" << frameReference.rows  << " \n Cols " << frameReference.cols << endl;

    cout << "Frame Depth" << frameReference.depth() << endl;
    cout << "Cannel " << frameReference.channels() << endl;


    for(;;) //Show the image captured in the window and repeat
    {


        convertBack(frameReference, resized_frame);

        if (frameReference.empty())
        {
            cout << " < < <  Game over!  > > > ";
            break;
        }
        ++frameNum;
        std::cout << "Transform frame " << frameNum << std::endl;
        // vft.transformPlane(frameReference,resized_frame, 500, 500, 1,1 );
        imshow(WIN_VID, resized_frame);

        char c = (char)waitKey(30);
        if (c == 27) break;

        // Get next Frame
        video_capture >> frameReference;

    }
    return 0;
}

void outImgToXYZ(int i, int j, char face, int edge, double & x, double & y, double & z) {
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
            cerr << "Error in outImgToXYZ, there are only 6 faces to a cube (0 to 5) ";
            exit(-1);

    }


}

int clip(int in, int min, int max){
    if (in < max &&  in > min) return in;
    else if(in < min) return min;
    return max;
}

void convertBack(Mat imgIn, Mat imgOut){

    // Image Sizes
    Size imgInSize = imgIn.size();
    Size imgOutSize = imgOut.size();

    int edge = imgInSize.width / 4;
    char face, face2;

    int iterate_from, iterate_to;

    double x; // 3D coords calculated by outImgToXYZ
    double y;
    double z;

    double theta, phi; // Radian degrees
    double uf, vf; // approximated out pixel coordinates
    int ui, vi; // Pixels to the bottom left
    int u2, v2; // Pixels to the rop right
    int mu, nu; // Fraction of way across pixels

    uchar A, B, C, D; // Pixel Values

    // Iterator for the image we want to have

    for(int i = 0;  i < imgOutSize.width; ++i){
        face = char(i / edge); // 0 - back, 1 - left 2 - front, 3 - right
        if(face==2){
            iterate_from = 0;
            iterate_to = 3 * edge;
        }
        else {
            iterate_from = edge;
            iterate_to = 2 * edge;
        }

        for (int j = iterate_from;  j  < iterate_to; ++j) {
            if(j < edge){
                face2 = 4;
            }
            else if(j >= 2 * edge){
                face2 = 5;
            }
            else{
                face2 = face;
            } // End of if

            outImgToXYZ(i, j, face2, edge, x, y, z);

            theta = atan( y / x); // check if between pi and -pi
            assert(theta <= 3.2 && theta >= -3.2);

            double r = hypot(x, y);

            phi = atan(z / r);
            assert(phi >= (- PI /2) && phi <= (PI / 2));

            // Source img coords
            uf = ( 2.0*edge*(theta + PI)/PI );
            vf = ( 2.0*edge * (PI/2 - phi)/PI);

            ui = int(floor(uf));
            vi = int(floor(vf));
            u2 = ui + 1;
            v2 = vi + 1;
            mu = uf - ui;
            nu = vf - vi;



            //Pixel values of four Corners
            A  = imgIn.at<uchar>(ui % imgInSize.height, clip(vi, 0, imgInSize.width -1));
            /*
            B  = imgIn.at<Vec3b>(u2 % imgInSize.width, clip(vi, 0, imgInSize.height -1));
            C  = imgIn.at<Vec3b>(ui % imgInSize.width, clip(v2, 0, imgInSize.height -1));
            D  = imgIn.at<Vec3b>(u2 % imgInSize.width, clip(v2, 0, imgInSize.height -1));
             */
            // TODO: Interpolate
            Vec3b outPix = A;

            imgOut.at<Vec3b>(i, j) = outPix; // invalid write

        }

    }



}
