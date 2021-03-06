//
// Created by flo on 09.10.18.
//

#include <math.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "cubetransform.h"

/* See https://stackoverflow.com/questions/29678510/convert-21-equirectangular-panorama-to-cube-map  for how i got to this solution.
*/



using namespace cv;

float faceTransform[6][2] = {
        {0, 0},
        {M_PI / 2, 0},
        {M_PI, 0},
        {-M_PI / 2, 0},
        {0, -M_PI / 2},
        {0, M_PI / 2}
};


void createCubeMapFace(Mat const &in, Mat &face,
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

void mapRectangleToPanorama(Mat const & inFrame,  int faceId,  int width,  int height, const Rect2d &  inRect, Rect & outRect ){

    float top_adr = 0;
    float left_adr = 0;
    float right_adr = 0;
    float bottom_adr = 0;

    float * top_ptr = & top_adr;
    float * left_ptr =  & left_adr;
    float * right_ptr = & right_adr;
    float * bottom_ptr = & bottom_adr;

    double top, left, right, bottom;

    top = inRect.y;
    left = inRect.x;
    right = left + inRect.width;
    bottom = top + inRect.height;


    getPanoramaCoords(inFrame, faceId, width, height, top, left, left_ptr,  top_ptr);
    getPanoramaCoords(inFrame, faceId, width, height, bottom, right, right_ptr,  bottom_ptr);

    Rect r(Point(* left_ptr, *top_ptr),
             Point( * right_ptr, * bottom_ptr));


    outRect.x = r.x;
    outRect.y = r.y;
    outRect.width = r.width;
    outRect.height = r.height;

}


/**
 *
 * @param matSize
 * @param face
 * @param i
 * @param j
 * @param edgeLength
 * @param u
 * @param v
 */
inline void panoramaCoords(Size matSize, int face, int i, int j, int edgeLength, float * u, float * v){

    float x;
    float  y;
    float  z;

    int height = matSize.height;
    int width = matSize.width;

    Mat xMap(edgeLength, edgeLength, CV_32F);
    Mat yMap(edgeLength, edgeLength, CV_32F);

    float phi, r, theta;


    getWorldCoords(i, j, face, edgeLength, &x, &y, &z);

    phi = atan2(y, x);
    r = hypot(x,y);
    theta =  M_PI_2 - atan2(z, r);

    // uf = (phi + M_PI) / M_PI * height;
    *u  = ((phi + M_PI) / (M_PI * 2)) * width;
    *v  = (theta) / ( M_PI) * height;
}

/**
 * Get the Coordinates in a cartesian coordinate System  for the Position (i,j) on
 * Cubeface face.
 * @param i
 * @param j
 * @param face
 * @param edge
 * @param x
 * @param y
 * @param z
 */
inline void getWorldCoords(int i, int j, int face, int edge, float * x, float * y, float * z){

    float a = (2 * float(i) / float(edge)) -1;

    float b =  (2 * float(j) / float(edge)) -1;


    if(face == 0){
        *x  = -1;
        *y =  -a;
        *z = -b;
    }
    else if(face == 1){
        *x = a;
        *y = -1.0f;
        *z = -b;
    }
    else if(face == 2){
        *x = 1.0f;
        *y = a;
        *z = -b;
    }
    else if (face == 3){
        * x = -a;
        * y = 1.0f;
        * z = -b;

    } else if (face==4) { // top
        *x =  b;
        *y = a;
        *z = 1.0f;
    } else if (face==5) { // bottom
        *x = -b;
        *y= a;
        *z = -1.0f;
    }
}

float get_max(float x, float y){
    if(x > y) return x;
    else return y;
}
float get_min(float x, float y){
    if(x < y) return x;
    return y;

}
void getCubeSide(Mat const &  imgIn, Mat & out, int edgeLength, int faceSide){

    Mat xMap(edgeLength, edgeLength, CV_32F);
    Mat yMap(edgeLength, edgeLength, CV_32F);

    float uf, vf;

    for(int i = 0; i < edgeLength; ++i){
        for (int j = 0; j < edgeLength; ++j){


            panoramaCoords(imgIn.size(), faceSide, i, j, edgeLength, &uf, &vf);

            xMap.at<float>(j, i) = uf;
            yMap.at<float>(j, i) = vf;


        }
    }

    remap(imgIn, out, xMap, yMap,
          INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
}









void getPanoramaCoords( Mat const & in, int faceId,  int width,  int height,
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


void cubeNet( Mat const & panorama, Mat & cubeNet){

    CV_Assert(panorama.depth() == CV_8U);



    int sideLength = 500;
    Mat face;


    const int type = panorama.type();

    Mat face_sides[6];
    for(int s = 0; s < 6; ++s) createCubeMapFace(panorama, face_sides[s],  s, sideLength, sideLength);


    int channels = face_sides[0].channels();

    int nRows = face_sides[0].rows;
    int nCols = face_sides[0].cols * channels;

    if (face.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    std::cout << "nRows "<<  nRows << std::endl;
    cubeNet.create(nRows , face_sides[0].cols *6, type);
    int i,j;
    uchar* p_cubeNet;
    uchar* p_face;


    for( i = 0; i < nRows; ++i)
    {
        p_cubeNet = cubeNet.ptr<uchar>(i);
        for(int s = 0; s < 6; ++s) {
            p_face = face_sides[s].ptr<uchar>(i);

            for (j = 0; j < nCols; ++j) {

                p_cubeNet[j + (nCols * s)] = p_face[j];

            }
        }
    }
}
