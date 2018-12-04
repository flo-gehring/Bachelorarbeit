#include <iostream> // for standard I/O
#include <string>   // for strings


/* TODO Since Merging / Splitting does not work reliably, i should probably, when two regions are merging, just put all the
    players into the outOfSightBox (and therefor let this be a FootballPlayer vector and not a Region vector.)
 */
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include "cubetransform.h"
#include "opencv_detect.h"
#include "detect.h"
#include "tracking.h"
#include "opencv_tracking.h"

#include "PanoramaTracking.h"
#include "PanoramaTrackingImplementations.h"


using namespace cv;
using namespace dnn;
using namespace std;


void darknet_on_cubenet(char * video_path);
void show_on_cubefaces(YOLODetector yoloD, char * video_path);
void save_video_projection(YOLODetector yoloDetector, char* inPath, char* outPath);
void darknet_predictions(char* video_path);

int main(int argc, char *argv[]) {
    if (!(argc == 6 || argc == 7))
    {
        cout << "Wrong number of parameters." << endl;
        cout << "Usage: ./Panorama2Cube show  <yolov3.cfg file> <yolov3.weight file> <coco.names file> <videofile>" << std::endl;
        cout << "./Panorama2Cube save <yolov3.cfg file> <yolov3.weight file> <coco.names file> <videofile> <outfile>" << std::endl;
        return -1;
    }
    stringstream conv;
    char * video_path = argv[5];


    RegionTracker rt;

    rt.trackVideo(video_path);

    /*
    string prefix = "/home/flo/Videos/";

    string videonames[] = {
            "Video2.mp4",
            "TS_10_5_t01.mp4",
            "TS_10_5.mp4"
    };

    RegionTracker rt;

    for(string const & s : videonames){
        rt.enableVideoSave(("tracked_" + s).c_str());
        rt.setAOIFile(("aoi_" + s + ".csv").c_str());
        rt.trackVideo((prefix+s).c_str());
    }
     */


    VideoCapture vc(video_path);

    string trackers[] = {
            //"GOTURN",
            "KCF",
            "Boosting",
            "MedianFlow",
            "MIL",
            "TLD",
            "MOSSE"

    };
    int numTrackers = sizeof(trackers) / sizeof(string);
    string tracker;

/*
    for(int i = 0; i < numTrackers; ++i) {
        tracker = trackers[i];
        tracker.append("_fps.mp4");
        cout << "Beginn tracking with "<< tracker << endl;
        trackVideo(video_path, tracker , trackers[i], matDetector);
    }
    */



/*

    YOLODetector yoloDetector(argv[2], argv[3], argv[4]);

    if(strcmp(argv[1], "save") == 0){
        char * outfile = argv[6];
        save_video_projection(yoloDetector, video_path, outfile);
    }
    else if(strcmp(argv[1], "show") == 0){
        show_on_cubefaces(yoloDetector, video_path);
    }
    else if(strcmp(argv[1], "darknet") == 0){
        darknet_predictions(video_path);
    }
*/

    return 0;
}


/*
 * Führt Detection auf dem Kompletten Video als Cubenet dargestellt aus.
 * Funktioniert gar nicht gut.
 */
void darknet_on_cubenet(char * video_path){

    MatDetector matDetector;
    VideoCapture video(video_path);
    Mat frameReference, resizedFrame;
    int frameNum = 0;

    const char * WIN_VID = "Darknet Detection";
    namedWindow(WIN_VID, WINDOW_AUTOSIZE);



        VideoCapture video_capture(video_path);

        if (!video_capture.isOpened())
        {
            std::cout  << "Could not open reference " << video_path << std::endl;
            return;
        }
        video_capture >> frameReference;
        waitKey(30);


        for (;;) //Show the image captured in the window and repeat
        {


            if (frameReference.empty()) {

                break;
            }


           cubeNet(frameReference, resizedFrame);

            matDetector.detect_and_display(resizedFrame);

            while(! matDetector.found.empty()){
                AbsoluteBoundingBoxes current_box = matDetector.found.back();

                rectangle(resizedFrame,
                          current_box.rect,
                          Scalar(0,0,255));
                matDetector.found.pop_back();

            }
            imshow(WIN_VID, resizedFrame);
            char c = (char) waitKey(20);
            if (c == 27) break;

            // Get next Frame
            video_capture >> frameReference;

        }


}

void darknet_predictions(char * video_path){

    MatDetector matDetector;
    VideoCapture video(video_path);
    Mat frameReference, resizedFrame;
    int frameNum = 0;

    const char * WIN_VID = "Darknet Detection";
    namedWindow(WIN_VID, WINDOW_AUTOSIZE);

    for (char face_id = 0; face_id < 6;  ++face_id) {

        VideoCapture video_capture(video_path);

        if (!video_capture.isOpened())
        {
            std::cout  << "Could not open reference " << video_path << std::endl;
            return;
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

            createCubeMapFace(frameReference, resizedFrame, face_id, 500, 500);

            matDetector.detect_and_display(resizedFrame);

            while(! matDetector.found.empty()){
                AbsoluteBoundingBoxes current_box = matDetector.found.back();

                rectangle(resizedFrame,
                          current_box.rect,
                          Scalar(0,0,255));
                matDetector.found.pop_back();

            }
            imshow(WIN_VID, resizedFrame);
            char c = (char) waitKey(20);
            if (c == 27) break;

            // Get next Frame
            video_capture >> frameReference;

        }
    }
    return;
}

void save_video_projection(YOLODetector yoloDetector, char* inPath, char* outPath){
    Mat er_projection, resized_er;
    Mat_<Vec3b> cube_face(Size(2000, 1000), Vec3b(255,0,0));


    // show_on_cubefaces(yoloD, argv[1]);

    VideoCapture video_capture(inPath);
    if(! video_capture.isOpened()){
        cout  << "Could not open reference " << inPath << endl;
        return;
    }


    video_capture >> er_projection;

    VideoWriter vw(outPath, VideoWriter::fourcc('M', 'J', 'P', 'G'),
                   video_capture.get(CAP_PROP_FPS),
                   er_projection.size(), true);
    waitKey(30);

    float left, top, right, bottom = 0;
    float * left_ptr = &left;
    float * top_ptr = &top;
    float * right_ptr = & right;
    float  * bottom_ptr = & bottom;

    int frame_counter = 0;
    while(! er_projection.empty()){

        for(short face_id = 0; face_id < 6; ++face_id){
            createCubeMapFace(er_projection, cube_face, face_id, 416, 416);
            yoloDetector.detect(cube_face);

            while(! yoloDetector.predictions.empty()){
                prediction  current_prediction = yoloDetector.predictions.back();

                getPanoramaCoords(er_projection, face_id,  416, 416,
                                  current_prediction.top,  current_prediction.left,
                                  left_ptr, top_ptr);

                getPanoramaCoords(er_projection, face_id,  416, 416,
                                  current_prediction.bottom, current_prediction.right,
                                  right_ptr, bottom_ptr);

                rectangle(er_projection, Point(int(* left_ptr), int(* top_ptr)),
                          Point(int(* right_ptr), int(* bottom_ptr)), Scalar(0, 0, 255));

                yoloDetector.predictions.pop_back();


            }

            cout << "Face Side " << face_id << " done." << endl;


            char c = waitKey(30);
            if(c == 27) return;

        }
        ++ frame_counter;
        cout << "Frame " << frame_counter <<  " of " << video_capture.get(CAP_PROP_FRAME_COUNT) << " done." << endl;
        vw.write(er_projection);
        video_capture >> er_projection;

    }
}

void show_on_cubefaces(YOLODetector yoloD, char* video_path){
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
            return;
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
            yoloD.detect(resized_frame);

            imshow(WIN_VID, resized_frame);

            char c = (char) waitKey(20);
            if (c == 27) break; // Press Esc to skip a current cubeface
            else if(c == 113){ // Press Q to leave Application
                return;
            }


            // Get next Frame
            video_capture >> frameReference;

        }
    }
}

