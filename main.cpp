#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <math.h>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O



#define PI 3.14159
using namespace std;
using namespace cv;
inline void createCubeMapFace(const Mat &in, Mat &face,
                              int faceId = 0, const int width = -1,
                              const int height = -1);


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
            ++frameNum;
            createCubeMapFace(frameReference, resized_frame, face_id, 500, 500);
            imshow(WIN_VID, resized_frame);
            char c = (char) waitKey(20);
            if (c == 27) break;

            // Get next Frame
            video_capture >> frameReference;

        }
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

void translateToPixelCoords(const Mat &, int pos, int &x, int &y){

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
    MatIterator_<Vec3b> it, end;
    int iteration_counter = 0;
    for( it = imgOut.begin<Vec3b>(), end = imgOut.end<Vec3b>(); it != end; ++it) {
        cout << "row: " << iteration_counter;
        /*
        (*it)[0] = table[(*it)[0]];
        (*it)[1] = table[(*it)[1]];
        (*it)[2] = table[(*it)[2]];
         */
    }

    int nRows = imgOut.rows;
    int nCols = imgOut.cols * imgOut.channels();

    if(imgOut.isContinuous()){
        nCols *= nRows;
        nRows = 1;
    }

    for(int i = 0;  i < nRows; ++i){
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
