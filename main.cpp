#include<stdio.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<fstream>
#include<sstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "opencv2/photo.hpp"
#include "algorithm"
#include <string>
#include <vector>
#include "opencv2/video/background_segm.hpp"
//opencvblobslib library cloned from http://github.com/OpenCVBlobsLib/opencvblobslib
#include "opencvblobslib/blob.h"
#include "opencvblobslib/BlobResult.h"
using namespace cv;
using namespace std;
Ptr<BackgroundSubtractorMOG2> bg;
int Keyboard;
int main()
{
    bg = createBackgroundSubtractorMOG2();
    Mat frame, roi;
    Mat back;
    Mat fore;
    VideoCapture capture("doo.mp4");
    bg->setNMixtures(3);
    bg->setDetectShadows(true);
    bg->setShadowValue(0);
    bg->setShadowThreshold(0.5);
    vector<vector<Point> > contours;
    //namedWindow("Frame", WINDOW_AUTOSIZE);
    //namedWindow("Foreground", WINDOW_AUTOSIZE);
    while ((char)Keyboard!='q' && (char)Keyboard!=27)
    {
        capture >> frame;
        roi = frame(Rect(46,25,144,84) );
        bg->apply(roi, fore);
        bg->getBackgroundImage(back);

        erode(fore,fore,Mat());
        dilate(fore,fore,Mat());
        //findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        //drawContours(roi,contours,-1,Scalar(0,0,255),2);
        //roi = frame(Rect(46,25,144,84) );
        //imshow("Frame",roi);
        imshow("Foreground",fore);
        //imshow("Background",back);
        Keyboard=waitKey(1);
    }
    return 0;
}
