//
// Created by flo on 09.10.18.
//

#include <math.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

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


