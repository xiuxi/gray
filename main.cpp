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
int Keyboard, colorCount=0, GlobalContourSize=0;
Point pt1 = Point(1,152), pt2 = Point(319,152);
//KFTracker implementation starts here.
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
  setIdentity(KF.processNoiseCov, Scalar::all(1e-3));
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));
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
      if(colorCount%2==0)
          line(img, kalman_vector[i], kalman_vector[i+1], Scalar(0,255,0), 1);
      else
          line(img, kalman_vector[i], kalman_vector[i+1], Scalar(0,255,255), 1);
  }
}
void KFTracker::clearPoints(){
  init=true;
    colorCount++;
}
int pedInCount, pedOutCount, stateIn=0, stateOut=0;
bool InCounted, OutCounted;
//KFTracker ends here
void Inflow(int yLoc);
void Outflow(int yLoc);
int main()
{
  pedInCount=0; pedOutCount=0;
  KFTracker KFObj;
  bg = createBackgroundSubtractorMOG2();
  Mat frame, back, fore;
  int xLoc, yLoc, yLocca, prevpedInCount=0, prevpedOutCount=0, i, CSize;
  Point Px1, Px2, point;
  VideoCapture capture("dooagain.mp4");
  bg->setNMixtures(5);
  bg->setDetectShadows(true);
  bg->setShadowValue(0);
  bg->setShadowThreshold(0.6);
  vector<vector<Point> > contours;
  namedWindow("Frame", WINDOW_AUTOSIZE);
  int con_size=0;
  while ((char)Keyboard!='q' && (char)Keyboard!=27)
  {
    capture >> frame;
    bg->apply(frame, fore);
    bg->getBackgroundImage(back);
    erode(fore,fore,Mat());
    dilate(fore,fore,Mat());
    dilate(fore,fore,Mat());
    findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    i=0, CSize=0;
    for( ; i < contours.size(); i++ )
    {
      approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
      boundRect[i] = boundingRect( Mat(contours_poly[i]) );
      minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
      CSize++;
      //cout<<endl<<(int)center[i].y<<" - ";
    }
    if(CSize!=0){
      for(i = 0; i< contours.size(); i++ )
       {
        if(contourArea(contours[i])>350){
        rectangle( frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,255), 2, 8, 0 );
        Px2 = boundRect[i].br();
        xLoc=Px2.x;
        yLocca=Px2.y;
        yLoc=center[i].y;
        KFObj.track(xLoc, yLocca);
        GlobalContourSize=contours.size();
        KFObj.draw(frame);
        }
       }
    }
    else
    {
      KFObj.clearPoints();
    }
    //cout<<yLoc<<endl;
    //up to down
    Inflow(yLoc);
    Outflow(yLoc);
    if(prevpedOutCount!=pedOutCount || prevpedInCount!=pedInCount)
    //cout<<endl<<"*"<<yLoc<<"*"<<endl;

//down to up
    line(frame, pt1, pt2, Scalar(0,255,0), 2, 8, 0);
    imshow("Frame",frame);
    //if(prevpedOutCount!=pedOutCount || prevpedInCount!=pedInCount)

     prevpedInCount=pedInCount;
     prevpedOutCount=pedOutCount;
    Keyboard=waitKey(1);
  }
  cout<<"Up Count: "<<pedInCount<<endl<<"Down Count: "<<pedOutCount<<endl;
  return 0;
}
void Inflow(int yLoc){
  if(stateIn==0){
    if((yLoc>171 && yLoc<176) || (yLoc>176 && yLoc<180) ){
      //cout<<yLoc<<"-";
      stateIn=1;
    }
  }
  if(stateIn==1 && (yLoc>155 && yLoc<160)){
    stateIn=2;
  }
  if(stateIn==2){
    //cout<<endl<<contours.size();
    if(GlobalContourSize>=1 && GlobalContourSize<3)
    pedInCount+=GlobalContourSize;
    else
    pedInCount+=2;
    //InCounted=true;
    stateIn=0;
  }
  //else InCounted=false;
  // if(!InCounted){
  //   if(stateIn==0){
  //     if(yLoc>151 && yLoc<161){
  //       stateIn=1;
  //     }
  //   }
  //   if(stateIn==1 && (yLoc>140 && yLoc<150)){
  //     stateIn=2;
  //   }
  //   if(stateIn==2){
  //     //cout<<endl<<contours.size();
  //     pedInCount++;
  //     InCounted=false;
  //     //InCounted=true;
  //     stateIn=0;
  //   }
  // }
}
void Outflow(int yLoc){
  if(stateOut==0){
     if((yLoc>120 && yLoc<125) || (yLoc>117 && yLoc<121)){
       stateOut=1;
       //cout<<yLoc<<"-";
     }
   }
   if(stateOut==1 && (yLoc>129 && yLoc<133)){
          stateOut=2;
   }
   if(stateOut==2){
     //cout<<endl<<contours.size();
     if(GlobalContourSize>=1 && GlobalContourSize<3)
     pedOutCount+=GlobalContourSize;
     else
     pedOutCount+=2;
     //OutCounted=true;
     stateOut=0;
   }
   //else OutCounted=false;
  //  if(!OutCounted){
  //    if(stateOut==0){
  //       if(yLoc>135 && yLoc<145){
  //         stateOut=1;
  //       }
  //     }
  //     if(stateOut==1 && (yLoc>155 && yLoc<165)){
  //       stateOut=2;
  //     }
  //     if(stateOut==2){
  //       //cout<<endl<<contours.size();
  //       pedOutCount++;
  //       OutCounted=true;
  //       stateOut=0;
  //     }
  //  }
}
