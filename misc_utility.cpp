//
// Created by flo on 19.03.19.
//

#include "misc_utility.h"


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


/**
 * Use this function to create Detections with the YOLO Detector.
 * @param videoPath
 * @param outfile
 * @param projector
 * @param darknetDetector
 */
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
    fflush(outfile);
    fclose(outfile);


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
