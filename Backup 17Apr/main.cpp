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
int Keyboard, TotalBlobs=0;
Point pt1 = Point(0,170), pt2 = Point(320,170);
//Some serious stuff down here.
int main()
{
  bg = createBackgroundSubtractorMOG2();
  Mat frame, roi;
  Mat back;
  Mat fore;
  VideoCapture capture("dooagain.mp4");
  bg->setNMixtures(3);
  bg->setDetectShadows(true);
  bg->setShadowValue(0);
  bg->setShadowThreshold(0.5);
  vector<vector<Point> > contours;
  namedWindow("Frame Drawing",WINDOW_AUTOSIZE);
  namedWindow("Frame", WINDOW_AUTOSIZE);
  Point temp;
  while ((char)Keyboard!='q' && (char)Keyboard!=27)
  {
    capture >> frame;
    CBlobResult blobs;
    roi = frame(Rect(76,199,113,40) );
    bg->apply(frame, fore);
    bg->getBackgroundImage(back);
    erode(fore,fore,Mat());
    dilate(fore,fore,Mat());
    findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    int i=0;
    for( ; i < contours.size(); i++ )
    {
      approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true ); //replace area with i.
      boundRect[i] = boundingRect( Mat(contours_poly[i]) );
      minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }
    Mat drawing = Mat::zeros( frame.size(), CV_8UC3 );
    for(i = 0; i< contours.size(); i++ )
    {
      rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), Scalar(0,0,255), 2, 8, 0 );
      temp = boundRect[i].br();
    }
    cout<<temp.y<<endl;
    if(temp.y<175 && temp.y>165){
    line(frame, pt1, pt2, Scalar(0,255,0), 3, 8, 0);
    }
    else{
    line(frame, pt1, pt2, Scalar(0,0,255), 3, 8, 0);
    }
    imshow("Frame Drawing",drawing);
    imshow("Frame",frame);
    Keyboard=waitKey(5);
  }
  return 0;
}
