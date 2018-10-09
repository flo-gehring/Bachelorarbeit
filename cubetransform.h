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

extern inline void createCubeMapFace(const Mat &in, Mat &face,
                              int faceId, const int width,
                              const int height);

#endif //PANORAMA2CUBEMAP_CUBETRANSFORM_H
