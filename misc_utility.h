//
// Created by flo on 19.03.19.
//

#ifndef PANORAMA2CUBEMAP_MISC_UTILITY_H
#define PANORAMA2CUBEMAP_MISC_UTILITY_H

#include "OtherTracking/PanoramaTracking.h"
#include "Detectors/opencv_detect.h"
#include "Detectors/detect.h"
#include <string>
#include "cubetransform.h"


using namespace std;


void darknet_on_cubenet(char * video_path);
void show_on_cubefaces(YOLODetector yoloD, char * video_path);
void save_video_projection(YOLODetector yoloDetector, char* inPath, char* outPath);
void darknet_predictions(char* video_path);

void createDetectionSourceFile(const char * videoPath, FILE * outfile, Projector * projector, MatDetector * darknetDetector );

void createImageDir(const char * videoPath , Projector * projector, string videoName, string projectorName);



#endif //PANORAMA2CUBEMAP_MISC_UTILITY_H
