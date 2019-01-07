//
// Created by flo on 05.01.19.
//

#include "TrackingHelpers.h"

/****************************************
 *                                      *
 *      Football Player Class Methods   *
 *                                      *
 ****************************************/

FootballPlayer::FootballPlayer(Rect coordinate, int frame, string const & identifier) {

    coordinates.emplace_back(coordinate);
    frames.emplace_back(frame);
    this->identifier = string(identifier);
    hist = Mat();
    x_vel = 0;
    y_vel = 0;
    isAmbiguous = false;

}

void FootballPlayer::addPosition(Rect coordinates, int frame) {
    if(frames.back() != frame) {
        this->coordinates.emplace_back(Rect(coordinates));
        this->frames.push_back(frame);
    }
}


Rect FootballPlayer::predictPosition(int frameNum) {

    if(coordinates.size() < 2)
        return Rect(coordinates.back());

    int deltaX[frames.size()];
    int deltaY[frames.size()];

    int iInc;
    for(int i = 0; i < coordinates.size(); ++i){
        iInc = i + 1;
        deltaX[i] = coordinates[coordinates.size() -  iInc].x;
        deltaY[i] = coordinates[coordinates.size() - iInc].y;
    }
    int numIterations = 0;
    for(int i = 0; i < coordinates.size() - 1; ++i){
        iInc = i + 1;

        deltaX[i] = deltaX[iInc] - deltaX[i];
        deltaY[i] = deltaY[iInc] - deltaY[i];
        numIterations = iInc;
        if(i > 0 &&  ((deltaX[i] * deltaX[i-1] < 0) || (deltaY[i] * deltaY[i-1] < 0))){ // Break if the Football Player changes directions
            break;
        }

    }

    int framesPassed = frames.back() - frames[frames.size() - (numIterations + 1)];

    double avgXChange = double(coordinates.back().x - coordinates[coordinates.size() - (numIterations + 1)].x) / double(framesPassed);
    double avgYChange = double(coordinates.back().y - coordinates[coordinates.size() - (numIterations + 1)].y) / double(framesPassed);


    return cv::Rect(
            int(coordinates.back().x + (avgXChange * ( frameNum - frames.back()))),
            int(coordinates.back().y + (avgYChange * (frameNum - frames.back()))),
            coordinates.back().width,
            coordinates.back().height
    );
}
/*
 * Calc new velocity, update coordinates and frame num.
 */
void FootballPlayer::update(Rect const &coordinates, int frame) {

    unsigned long numKnownFrames = frames.size();

    addPosition(coordinates, frame);


    auto coordinatesIterator = this->coordinates.begin();

    Rect & lastPosition = * coordinatesIterator;

    double deltaX, deltaY;
    while(coordinatesIterator != this->coordinates.end()){
        deltaX = lastPosition.x - coordinatesIterator->x;
        deltaY = lastPosition.y - coordinatesIterator->y;

        x_vel += deltaX;
        y_vel += deltaY;

        lastPosition = * coordinatesIterator;
        ++coordinatesIterator;

    }

    x_vel = x_vel / numKnownFrames;
    y_vel = y_vel / numKnownFrames;
}



/********************************************
 *                                          *
 *          Region Class Methods            *
 *          ^^^^^^^^^^^^^^^^^^^^            *
 ********************************************/

/*
 * Adds the Position of the Region and the current Frame to the tracked objects in the Region.
 */
void Region::updatePlayerInRegion(int frameNum) {

    playerInRegion->update(coordinates, frameNum);

}


Region::Region(const Rect &coordinates, FootballPlayer *  ptrPlayer) {
    this->coordinates = coordinates;
    playerInRegion = ptrPlayer;
}

Region::Region(Rect coordinates) {
    this->coordinates = coordinates;
    playerInRegion = nullptr;
}


bool Region::regionsIntersect(const Region &r1, const Region &r2){
    return ((r1.coordinates & r2.coordinates).area() > 0);

}

void estimatePosition(Region const & r1, int framesPassed, Rect & r){
    int pixelPerFrame = 3;
    Rect rect1 = Rect(r1.coordinates);
    int x1, y1,width1 ,height1;
    x1 = rect1.x - (framesPassed * pixelPerFrame);
    y1 = rect1.y - (framesPassed * pixelPerFrame);
    width1 = rect1.width + (2 * pixelPerFrame * framesPassed);
    height1 = rect1.height + (2 * pixelPerFrame * framesPassed);

    r.x = x1;
    r.y = y1;
    r.width = width1;
    r.height = height1;
}

bool Region::regionsInRelativeProximity(Region const &r1, Region const &r2, int framesPassed) {

    Rect estimate1, estimate2;
    estimatePosition(r1, framesPassed, estimate1);
    estimatePosition(r2, framesPassed, estimate2);

    return (estimate1 & estimate2).area() > 0;
}

Region::Region(Region const &r1) {
    coordinates = Rect(r1.coordinates);
    playerInRegion = r1.playerInRegion;
    for(unsigned char i = 0; i < 3; ++i){
        labShirtColor[i] = r1.labShirtColor[i];
        bgrShirtColor[i] = r1.bgrShirtColor[i];
    }

}


Mat Region::getLabColors(Mat const &frame, int colorCount) {
    Mat regionImgReference, regionImgCopy;

    regionImgReference = frame(coordinates);
    regionImgReference.copyTo(regionImgCopy);

    // Copy all the Pixel values to the samples Mat
    int clusterCount = colorCount;
    Mat labels, centers;

    Mat kMeanImage;
    helperBGRKMean(regionImgCopy, colorCount, labels, centers);

    int  clusterColorCount[clusterCount];
    for(int i = 0; i < clusterCount; ++i) *(clusterColorCount + i) = 0;


    for(int x = 0; x < labels.rows; ++x){
        (* (clusterColorCount + labels.at<int>(x, 0)))++;
    }

    Mat rgbColorClusters(Size(clusterCount,1), CV_8UC3);
    for(int i = 0; i < clusterCount; ++i) {
        float blue = centers.at<float>(i, 0);
        float green = centers.at<float>(i, 1);
        float red = centers.at<float>(i, 2);
        /*
        printf("Cluster %i: %i with (B,G,R) %f, %f, %f \n",
               i, *(clusterColorCount + i), blue, green,
               red);*/
        rgbColorClusters.at<Vec3b>(0, i)[0] = red;
        rgbColorClusters.at<Vec3b>(0, i)[1] = green;
        rgbColorClusters.at<Vec3b>(0, i)[2] = blue;

    }
    Mat labColorCluster;
    cvtColor(rgbColorClusters, labColorCluster, COLOR_BGR2Lab);

// #define P2C_SHOW_KMEAN_WINDOW
#ifdef P2C_SHOW_KMEAN_WINDOW
    namedWindow("KMEAN");
        Mat new_image( regionImgCopy.size(), regionImgCopy.type() );

        cout << labColorCluster << endl;
        for( int y = 0; y < regionImgCopy.rows; y++ ) {
            for (int x = 0; x < regionImgCopy.cols; x++) {
                // Save the index of the Cluster the current pixel is in in cluster_idx
                // (Labels is (imgCopy.rows * imgCopy.cols) long and saves the cluster every pixel is in).
                // centers saves the rgb values the center of every cluster.
                int cluster_idx = labels.at<int>(y + x * regionImgCopy.rows, 0);
                new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
                new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
                new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
            }

        }
        putText(new_image,  playerInRegion->identifier, Point(0,coordinates.height), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255), 2);
        imshow("KMEAN", new_image);

        while(true){
            char c = waitKey(30);
            if (c == 32) break;
        }
#endif


    return Mat(labColorCluster);
}

/*
 * Try to determine the color of the shirt.
 * Perform the k-mean algorithm and check which colors correspond to the pixels in the foreground the most
 */
Mat Region::getShirtColor(Mat const &frameFull, Mat const & foregroundMask) {

    // Estimation:
    // Crop the picture head and legs from picture
    int croppedYSource = coordinates.height / 6;
    int croppedLength = coordinates.height / 2;
    Rect croppedRect = Rect(coordinates.x,
                            coordinates.y + croppedYSource,
                            coordinates.width,
                            croppedLength);




    Mat regionImgReference, regionImgCopy;

    Mat foregroundReference = foregroundMask(croppedRect);
    regionImgReference = frameFull(croppedRect);

    regionImgReference.copyTo(regionImgCopy);
    Mat frame = regionImgReference;

    const int colorCount  = 2;
    Mat labels, centers;

    Mat returnKmean = helperBGRKMean(frame, colorCount, labels, centers);

    Mat grayCenters(Size(1, colorCount), CV_8UC3);
    for(int cluster_idx = 0; cluster_idx < colorCount; ++cluster_idx) {
        grayCenters.at<Vec3b>(0, cluster_idx)[0] = centers.at<float>(cluster_idx, 0);
        grayCenters.at<Vec3b>(0, cluster_idx)[1] = centers.at<float>(cluster_idx, 1);
        grayCenters.at<Vec3b>(0, cluster_idx)[2] = centers.at<float>(cluster_idx, 2);
    }

    cvtColor(grayCenters, grayCenters, COLOR_BGR2GRAY);

    long weight[colorCount];
    int numColorAppearances[colorCount];

    // Prepare Arrays
    for(int index = 0; index < colorCount; ++index){
        weight[index] = 0;
        numColorAppearances[index] = 0;
    }

    /*
    double yFactor, xFactor; // Range [1,2]
    int clusterId;
    float frameRows = frame.rows;
    float frameCols = frame.cols;
    for(int y = 0; y < frame.rows; ++y){
        for(int x = 0; x < frame.cols; ++x){
             clusterId = labels.at<int>(y + x * frame.rows, 0);
             if(foregroundReference.at<char>(x,y) != 0) weight[clusterId]++;
             ++numColorAppearances[clusterId];
        }
    }
     */

    Mat image = Mat(returnKmean);
    //Prepare the image for findContours
    cvtColor(image, image, CV_BGR2GRAY);

    // Adapt the threshold so the contours will always be found

    uchar * rowPtr = grayCenters.ptr<uchar>(0);
    uchar val1 = rowPtr[0];
    uchar val2;
    if(! grayCenters.isContinuous()){
        rowPtr = grayCenters.ptr<uchar>(1);
        val2 = rowPtr[0];
    }
    else{
        val2 = rowPtr[1];
    }

    cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));

    Mat contoursImageInverted(image.size(), CV_8UC3, Scalar(0,0,0));

    int threshold = (val1 + val2) / 2;

    cv::threshold(image, contourImage, threshold, 255, CV_THRESH_BINARY);
    cv::threshold(image, contoursImageInverted, threshold, 255, CV_THRESH_BINARY_INV);

    // Mask the Picture
    Mat maskInverted;
    regionImgCopy.copyTo(maskInverted, contoursImageInverted);

    Mat masked;
    regionImgReference.copyTo(masked, contourImage);

#ifdef undef
    imshow("kmean", returnKmean);
    cvMoveWindow("kmean", 100, 100);
    imshow("Masked", masked);
    cvMoveWindow("Masked", 100, 200);
    imshow("MaskedInv", maskInverted);
    cvMoveWindow("MaskedInv", 150, 200);

    cv::imshow("Converted KMean", image);
    cvMoveWindow("Converted KMean", 0, 0);
    cv::imshow("Contours", contourImage);
    cvMoveWindow("Contours", 200, 0);

    imshow("inverted", contoursImageInverted);
    cvMoveWindow("inverted", 100, 100);

    namedWindow("Player");
    imshow("Player", regionImgReference);
    cvMoveWindow("Player", 200, 200);

#endif

    vector<Point> nonZeroMask;
    vector<Point> invertedNonZeroMask;


    cv::findNonZero(contourImage, nonZeroMask);

    Mat invertedImage;
    invertedImage = image.clone();

    cv::findNonZero(contoursImageInverted, invertedNonZeroMask);

    Mat nonZeroPixels(Size(1, nonZeroMask.size()), CV_8UC3);

    nonZeroPixels.zeros(Size(1, nonZeroMask.size()), CV_8UC3);

    Mat invertedNonZeroPixels(Size(1, invertedNonZeroMask.size()), CV_8UC3);

    int column = 0;
    for(Point const & point:nonZeroMask){
        nonZeroPixels.at<Vec3b>(0, column) = regionImgCopy.at<Vec3b>(point);
        ++column;
    }

    column = 0;
    for(Point const & point: invertedNonZeroMask){
        invertedNonZeroPixels.at<Vec3b>(0, column) = regionImgCopy.at<Vec3b>(point);
        ++column;
    }



    Mat newCenters, newLabels, newCentersInverted;


    helperBGRKMean(nonZeroPixels, 1, newLabels, newCenters);
    helperBGRKMean(invertedNonZeroPixels, 1, newLabels, newCentersInverted);

    Mat colorValuesNewCenter(Size(1,1), CV_8UC3);
    Mat colorValuesNewCenterInverted(Size(1,1), CV_8UC3);
    uchar * ncPtr = colorValuesNewCenter.ptr(0);
    ncPtr[0] = newCenters.at<float>(0, 0);
    ncPtr[1] = newCenters.at<float>(0, 1);
    ncPtr[2] = newCenters.at<float>(0, 2);

    uchar * ncIPtr = colorValuesNewCenterInverted.ptr(0);
    ncIPtr[0] = newCentersInverted.at<float>(0, 0);
    ncIPtr[1] = newCentersInverted.at<float>(0, 1);
    ncIPtr[2] = newCentersInverted.at<float>(0, 2);

    /*
    cout << "New Centers:" << endl << newCenters << endl;
    cout << "Inverted: " << endl << newCentersInverted << endl;
    */


    // If newCentersInverted is close to red, the player has a red shirt.
    Mat labColorCenter(Size(1,1), CV_8UC3);
    Mat hsvColorCenterInverted(Size(1,1), CV_8UC3);


    cvtColor(colorValuesNewCenterInverted, labColorCenter, CV_BGR2Lab);
    cvtColor(colorValuesNewCenterInverted, hsvColorCenterInverted, CV_BGR2HSV);

    auto *  hsvPtr = hsvColorCenterInverted.ptr<uchar>(0);
    uchar hue = hsvPtr[0];



    auto * cPtr = labColorCenter.ptr<uchar>(0);


    Mat examplebgr(Size(1,1), CV_8UC3);
    auto * exampleptr = examplebgr.ptr<uchar>(0);
    exampleptr[0] = 17;
    exampleptr[1] = 36;
    exampleptr[2] = 123;


    cvtColor(examplebgr, examplebgr, CV_BGR2Lab);

    resize(colorValuesNewCenter, colorValuesNewCenter, Size(30, 30));
    resize(colorValuesNewCenterInverted, colorValuesNewCenterInverted, Size(30, 30));

#ifdef undef
    imshow("Color", colorValuesNewCenter);
    imshow("Color Inverted", colorValuesNewCenterInverted);
    waitKey(0);
#endif

    double distanceToRed = deltaECIE94(cPtr[0], cPtr[1], cPtr[2],
                                       exampleptr[0], exampleptr[1], exampleptr[2]);

    // Use the Color Center with the higher red value;

    Mat colorValues(1,1,CV_8UC3);
    uchar * cvPtr = colorValues.ptr(0);
    if( hue < 15 || hue > 150){
        cvPtr[0] = 0;
        cvPtr[1] = 0;
        cvPtr[2] = 255;


    }
    else{
        cvPtr[0] = 255;
        cvPtr[1] = 255;
        cvPtr[2] = 255;

    }


    /*
    cvPtr[0] = centers.at<float>(colorIndex, 0);
    cvPtr[1] = centers.at<float>(colorIndex, 1);
    cvPtr[2] = centers.at<float>(colorIndex, 2);
     */

    return colorValues;
}

/*
 * Fills the Paramters bgrShirtColor and labShirtColor.
 */
void Region::createColorProfile(Mat const &frame, Mat const & foregroundMask) {
    Mat bgrShirtColorTemp = getShirtColor(frame, foregroundMask);
    Mat labShirtColorTemp;
    cvtColor(bgrShirtColorTemp, labShirtColorTemp, CV_BGR2Lab);

    auto * bgrColorPtr = bgrShirtColorTemp.ptr<uchar>(0);
    auto * labColorPtr = labShirtColorTemp.ptr<uchar>(0);

    for(int i = 0; i < 3; ++i){
        labShirtColor[i] = labColorPtr[i];
        bgrShirtColor[i] = bgrColorPtr[i];
    }
}