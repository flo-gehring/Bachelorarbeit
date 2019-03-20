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

#include "misc_utility.h"

using namespace cv;
using namespace dnn;
using namespace std;



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
            "/home/flo/Videos/TS_10_5_t01.mp4",
           "/home/flo/Videos/TS_10_5.mp4"

    };

    VideoCapture t(videonames[2]);

    Mat t_frame, projected_t_frame, same_video_excerpt;

    t >> t_frame;

    CubeMapProjector cmp;

    string vid[] = {
            "TS_10_5",
            "Video2",

            "TS_10_5_t01",

    };

    Projector * projectors[] = {
            new EquatorLine(cv::Size(3840, 1920), 3840, 1920),
            new EquatorLine(cv::Size(3840, 1920)),
            new CleanCubeMap()

    };
    string projectorNames[] = {
            "whole_frame",
            "equator_line",
            "cubemap"

    };
    // MatDetector darknetDetector;


    FILE * detectionOutFile;
    string filename;


    YOLOWrapper yw;

    MatDetector * ptr_darknetDetector = &(yw.matDetector);

    FILE * outFile;
    std::string outFileName;
    PanoramaTracking * pt;
    for(const char * name: TRACKER_NAMES){
        if(!name) break;

        cout << name << endl;
        outFileName = string(name) + "_video2";
        cout << outFileName << endl;
        outFile = fopen((outFileName + ".txt").c_str(), "w");
        pt = new PanoramaTracking(&yw, name, projectors[2]);
        pt->trackingResult = outFile;
        pt->trackVideo(videonames[0].c_str(), (outFileName + ".mp4").c_str());


        fflush(outFile);
        fclose(outFile);

        cout << "Finished: "  << name << endl;
        delete pt;

    }

    cout << "Finished " << endl;






    for(string const & videoname : vid){
        filename = prefix + videoname + ".mp4";

        for(int i = 2; i >= 2; --i){
            cout << videoname <<  endl << "\t" << filename << endl << "\t" << "StartTime: " << time(0) << endl;

            string name = std::string( videoname +"_"  + projectorNames[i] + ".json");
            const char  * detFile = name.c_str();
            cout << detFile << endl;

            detectionOutFile = fopen(detFile , "w");


            createDetectionSourceFile(filename.c_str(), detectionOutFile, projectors[i], ptr_darknetDetector);
            // fclose(detectionOutFile);
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
