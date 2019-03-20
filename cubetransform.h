//
// Created by flo on 09.10.18.
//
#include <math.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

#ifndef PANORAMA2CUBEMAP_CUBETRANSFORM_H

#define PANORAMA2CUBEMAP_CUBETRANSFORM_H

extern float faceTransform[6][2];

void getPanoramaCoords( Mat const & in, int faceId,  int width,  int height,
                  int x, int y,
                  float * u_ptr, float* v_ptr);

void createCubeMapFace(Mat const &in, Mat &face,
                              int faceId, const int width,
                              const int height);

void mapRectangleToPanorama(Mat const & inFrame,  int faceId,  int width,  int height,const Rect2d & inRect, Rect & outRect );


void cubeNet(Mat const & panorama, Mat& cubemap);


// CLeaner Projection, no dirty tricks. This is as it's explained in the Thesis.
void panoramaCoords(Size matSize, int face, int i, int j, int edgeLength, float * u, float * v);
void getWorldCoords(int i, int j, int face, int edge, float * x, float * y, float * z) ;
void getCubeSide(Mat const & imgIn, Mat & out, int edgeLength, int faceSide);

#endif //PANORAMA2CUBEMAP_CUBETRANSFORM_H
