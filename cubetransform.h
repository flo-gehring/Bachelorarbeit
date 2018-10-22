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

void getPanoramaCoords( Mat & in, int faceId,  int width,  int height,
                  int x, int y,
                  float * u_ptr, float* v_ptr);

void createCubeMapFace(Mat  &in, Mat &face,
                              int faceId, const int width,
                              const int height);

void mapRectangleToPanorama(Mat & inFrame,  int faceId,  int width,  int height,const Rect2d & inRect, Rect & outRect );


void cubeNet(Mat & panorama, Mat& cubemap);


#endif //PANORAMA2CUBEMAP_CUBETRANSFORM_H
