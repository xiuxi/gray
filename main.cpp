#include<iostream>
#include "algorithm"
#include <vector>
//opencv lib
#include<opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/video.hpp>
#include "opencv2/photo.hpp"
#include "opencv2/video/background_segm.hpp"
//own header file
#include "KFTracker.h"

using namespace cv;
using namespace std;
Ptr<BackgroundSubtractorMOG2> bg;
int Keyboard;
Point pt1 = Point(0,170), pt2 = Point(320,170);
//Some serious stuff is in the air.
//KFTracker implementation start.
KFTracker::KFTracker(){
  KF = KalmanFilter(2, 2, 0);
  state = Mat(2, 1, CV_32F);
  processNoise = Mat(2, 1, CV_32F);
  measurement = Mat::zeros(2, 1, CV_32F);
  init = true;
}
KFTracker::~KFTracker(){}
void KFTracker::initializeKF(int x, int y){
  KF.statePre.at<float>(0) = x;
  KF.statePre.at<float>(1) = y;
  KF.transitionMatrix = (Mat_<float>(2, 2) << 1,0,0,1 );

  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
  setIdentity(KF.errorCovPost, Scalar::all(1));

  points_vector.clear();
  kalman_vector.clear();
  init = false;
}
void KFTracker::track(int x, int y){
  if ( init )
      initializeKF(x, y);

  Mat prediction = KF.predict();
  Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

  measurement.at<float>(0) = x;
  measurement.at<float>(1) = y;

  Point measPt(measurement.at<float>(0), measurement.at<float>(1));
  points_vector.push_back(measPt);

  Mat estimated = KF.correct(measurement);
  Point statePt( estimated.at<float>(0), estimated.at<float>(1) );
  kalman_vector.push_back(statePt);
}
void KFTracker::draw(Mat img){
  for (int i=0; i<kalman_vector.size()-1; i++) {
      line(img, kalman_vector[i], kalman_vector[i+1], Scalar(0,255,0), 1);
  }
}
//end
int main()
{
  KFTracker KFObj;
  bg = createBackgroundSubtractorMOG2();
  Mat frame, back, fore;
  VideoCapture capture("dooagain.mp4");
  bg->setNMixtures(3);
  bg->setDetectShadows(true);
  bg->setShadowValue(0);
  bg->setShadowThreshold(0.5);
  vector<vector<Point> > contours;
  namedWindow("Frame", WINDOW_AUTOSIZE);
  int xLoc, yLoc;
  Point Px;
  while ((char)Keyboard!='q' && (char)Keyboard!=27)
  {
    capture >> frame;
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
    for(i = 0; i< contours.size(); i++ )
    {
      rectangle( frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,0,255), 2, 8, 0 );
      Px=boundRect[i].br();
      xLoc=(int)Px.x;
      yLoc=(int)Px.y;
      KFObj.track(xLoc, yLoc);
      KFObj.draw(frame);
    }
    line(frame, pt1, pt2, Scalar(0,255,0), 3, 8, 0);
    imshow("Frame",frame);
    Keyboard=waitKey(5);
  }
  return 0;
}
