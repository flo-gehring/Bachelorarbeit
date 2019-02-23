#include <iostream> // for standard I/O
#include <string>   // for strings


#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include "cubetransform.h"
#include "Detectors/opencv_detect.h"
#include "Detectors/detect.h"
#include "RegionTracker/tracking.h"
#include "OtherTracking/opencv_tracking.h"
#include "Detectors/testMaskRCNN.h"

#include "OtherTracking/PanoramaTracking.h"
#include "OtherTracking/PanoramaTrackingImplementations.h"

using namespace cv;
using namespace dnn;
using namespace std;

void darknet_on_cubenet(char * video_path);
void show_on_cubefaces(YOLODetector yoloD, char * video_path);
void save_video_projection(YOLODetector yoloDetector, char* inPath, char* outPath);
void darknet_predictions(char* video_path);

void createDetectionSourceFile(const char * videoPath, FILE * outfile, Projector * projector, MatDetector * darknetDetector );

void createImageDir(const char * videoPath , Projector * projector, string videoName, string projectorName){
    VideoCapture videoCapture(videoPath);
    Mat currentFrame;

    videoCapture >> currentFrame;
    Mat projectedFrame;

    string dirName = videoName + "/"+ projectorName;

    system(("mkdir " + videoName).c_str());
    system(("mkdir " + dirName).c_str());



    int frameCounter = -1;
    while(! currentFrame.empty()){
        ++frameCounter;
        int numOfProjections = projector->beginProjection();

        string frameDir = dirName + "/" + to_string(frameCounter);

        system(("mkdir " + frameDir).c_str());

        for(int projectionIndex = 0; projectionIndex < numOfProjections; ++projectionIndex){
            projector->project(currentFrame, projectedFrame);
            cout << frameDir << endl;

            projector->project(currentFrame, projectedFrame);
            imwrite((frameDir + "/" + to_string(projectionIndex) + ".jpeg").c_str(), projectedFrame);

        }
        videoCapture >> currentFrame;
        cout << frameCounter << flush;

    }
}


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


    string prefix = "/home/flo/Videos/";

    string videonames[] = {
            "/home/flo/Videos/Video2.mp4",
           "/home/flo/Videos/TS_10_5.mp4",
           "/home/flo/Videos/TS_10_5_t01.mp4"
    };

    string vid[] = {

            // "TS_10_5",
            "TS_10_5_t01",
            "Video2"
    };

    Projector * projectors[] = {
            new EquatorLine(cv::Size(3840, 1920)),
            new CubeMapProjector()

    };
    string projectorNames[] = {
            "equator_line",
            "cubemap"

    };

    createImageDir(videonames[2].c_str() , projectors[1], "TS_10_5_t01", "cubemap");
    return 0;





    FILE * detectionOutFile;
    string filename;
    MatDetector darknetDetector;

    MatDetector * ptr_darknetDetector = & darknetDetector;
    for(string const & videoname : vid){
        filename = prefix + videoname + ".mp4";

        for(int i = 1; i >= 0; --i){
            cout << videoname <<  endl << "\t" << filename << endl << "\t" << "StartTime: " << time(0) << endl;
            cout << "\t Projector: " << projectorNames[i] << endl;
            detectionOutFile = fopen(("darknet-yolo_" + videoname + "_"+ projectorNames[i] + ".json").c_str(), "w");


            createDetectionSourceFile(filename.c_str(), detectionOutFile, projectors[i],  ptr_darknetDetector);

            fclose(detectionOutFile);

            cout  << endl << "\t finished: " << time(0) << endl;

        }
    }

    for(Projector * p:projectors) delete p;




    /*

    VideoCapture vc("/home/flo/Videos/TS_10_5.mp4");
    Mat testFrame;
    vc >> testFrame;
    EquatorLine sp(testFrame.size(), 1500, 750);
    vc.release();

    detectOnVideo("/home/flo/Videos/TS_10_5.mp4", & sp);


    // AOIFileDetectorWrapper yoloWrapper("data/AOI/neu_aoi_TS_10_5.data");
    YOLOWrapper yoloWrapper;
    CubeMapProjector cubeMapProjector;

    PanoramaTracking panoramaTracking(&yoloWrapper, "Boosting", &sp);

    panoramaTracking.trackVideo("/home/flo/Videos/TS_10_5.mp4");

    string AOIFiles[] = {
            //"data/AOI/aoi_from_vid.data",
            "data/AOI/aoi_TS_10_5_lang.data",
            "data/AOI/aoi_TS_10_5_t01.data"

    };


    */

    CubeMapProjector  cubeface = CubeMapProjector();

    for(string const & s : videonames){

        size_t fromSubstring  = s.find_last_of('/');
        size_t toSubstring = s.find_last_of('.');
        string name = s.substr(fromSubstring, toSubstring + 1);
        FILE * detectionFile = fopen((name + ".json").c_str(), "w");

        detectOnVideo(s.c_str(), &cubeface, detectionFile);
    }

    return 0;

    RegionTracker rt;
    rt.assignmentThreshold = 2.5;
    rt.minDistanceThreshold = 0.3;
    int i = 0;




    for(string const & s : videonames){


       // rt.setAOIFile(("hmm_" + s + ".csv").c_str());


       cout << "opening " << s << endl;
        VideoCapture vc(s.c_str());

        if(vc.isOpened()) {
            cout << "opened.. " << endl;
        }
        else{
            cout << "Failure" << endl;
        }

        int currentFrame = 0;
        Mat frame;
        vc >> frame;
        rt.debugData = nullptr;

        FILE * outfile = fopen((s.substr(0, s.length() - 4) + ".json").c_str(), "w");
        while(! frame.empty()){
            vector<Rect> dets = rt.detectOnFrame(frame);
            printDetectionsToFile(outfile, currentFrame, dets);
            ++currentFrame;
            // imshow("frame", frame);
            vc >> frame;
        }
        fclose(outfile);




       //rt.darknetDetector.loadAOI("data/AOI/neu_aoi_TS_10_5.data");

       // rt.trackVideo((prefix+s).c_str());
       // cout << s << " finished.." << endl;
       // rt.printTrackingResults("trackingResult.txt");
    }



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
}

void createDetectionSourceFile(const char * videoPath, FILE * outfile, Projector * projector, MatDetector * darknetDetector){

    VideoCapture videoCapture(videoPath);
    Mat currentFrame;

    videoCapture >> currentFrame;
    Mat projectedFrame;

    vector<Rect> boundingBoxes;
    vector<float> confScores;

    Rect sourceCoords;
    int frameCounter = -1;
    fprintf(outfile, "[ \n");
    while(! currentFrame.empty()){

        int numOfProjections = projector->beginProjection();
        boundingBoxes.clear();
        confScores.clear();

        for(int projectionIndex = 0; projectionIndex < numOfProjections; ++projectionIndex){
            projector->project(currentFrame, projectedFrame);

            darknetDetector->detect_and_display(projectedFrame);

            for(AbsoluteBoundingBoxes const & abb: darknetDetector->found) {
                sourceCoords = projector->sourceCoordinates(currentFrame, abb.rect, projectionIndex);
                boundingBoxes.emplace_back(Rect(sourceCoords));
                confScores.emplace_back(abb.prob);
            }
        }
        frameCounter++;

        Rect currentBB;

        string listOfDetections = "";
        for(int i = 0; i < confScores.size(); ++i){
            // Generate List of detections

             currentBB = boundingBoxes[i];

            listOfDetections.append(
                        std::string("{ \"x\": " )+  to_string(currentBB.x) +  std::string(", \"y\":" ) + to_string(currentBB.y) + std::string(", \"width\": " )+
                        to_string(currentBB.width)+ std::string( ", \"height\": ") + to_string( currentBB.height) + + ", \"confidence\": " + to_string(confScores[i]) +
                        std::string( "},\n")
                );
            }
        listOfDetections = listOfDetections.substr(0, listOfDetections.length() - 2);


        fprintf(outfile,
                    "{ \"frame\":%i, \n \"detections\": [ %s] }, ", frameCounter, listOfDetections.c_str());
        videoCapture >> currentFrame;
        cout << frameCounter << " ";
        cout << flush;
        }

    fprintf(outfile, "]");


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

