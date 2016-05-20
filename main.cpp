#include<iostream>
#include<stdlib.h>
#include "algorithm"
#include <vector>
#include<ctime>
#include<opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/video.hpp>
#include "opencv2/photo.hpp"
#include "opencv2/video/background_segm.hpp"
#include "KFTracker.h"

using namespace cv;
using namespace std;
Ptr<BackgroundSubtractorMOG2> bg;
int Keyboard, colorCount=0, GlobalContourSize=0;
int GlobalContourArea=0;
int currentState=0;
int GlobalCounter=0;
int regionOneCeil = 110, regionTwoCeil=130, regionTwoFloor=150;
class mycounter{
public:
  mycounter();
  int CSize, CArea;
  int updateCounter(){
    if(CSize==1){
      if(CArea>900 && CArea<1600){
        GlobalCounter+=2;
      }
      else if(CArea>1600){
        GlobalCounter+=3;
      }
    }
    else{
      GlobalCounter+=CSize;
    }
  }
};
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
  setIdentity(KF.processNoiseCov, Scalar::all(1e-3));//3
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));//2
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
void KFTracker::clearPoints(){
  init=true;
  colorCount++;
}
int pedInCount, pedOutCount, stateIn=0, stateOut=0;
time_t start_s, stop_s;
void Inflow(int yLoc);
void Outflow(int yLoc);
void get_count(int yLoc);
int CSizeup=0, CSizedown=0;
int CAreaup=0, CAreadown=0;
int main()
{
  mycounter cObj;
  KFTracker KFObj;
  bg = createBackgroundSubtractorMOG2();
  Mat frame, back, fore, foreup, foredown;
  Mat up, down;
  int xLoc, yLoc, i, cSize;
  VideoCapture capture("s1.mp4");
  bg->setNMixtures(5);
  bg->setDetectShadows(true);
  bg->setShadowValue(0);
  bg->setShadowThreshold(0.6);
  vector<vector<Point> > contours;
  vector<vector<Point> > contoursup;
  vector<vector<Point> > contoursdown;
  namedWindow("Frame", WINDOW_AUTOSIZE);
  int con_size=0;
  while ((char)Keyboard!='q' && (char)Keyboard!=27)
  {
    capture >> frame;
    if(!capture.read(frame)){
      exit(0);
    }
    up = frame(Rect(1,110,318,10));
    down = frame(Rect(1,140,318,10));
    bg->apply(frame, fore);
    bg->getBackgroundImage(back);
    erode(fore,fore,Mat());
    dilate(fore,fore,Mat());
    dilate(fore,fore,Mat());
    findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    // Start
    // Creating an entire set for the segmented region
    bg->apply(up, foreuo);
    bg->apply(down, foredown);
    erode(foreup,foreup,Mat());
    erode(foredown,foredown,Mat());
    dilate(foreup,foreup,Mat());
    dilate(foreup,foreup,Mat());
    dilate(foredown,foredown,Mat());
    dilate(foredown,foredown,Mat());
    findContours(foreup,contoursup,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(foredown,contoursdown,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    // end
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<Point2f>center(contours.size());
    vector<float>radius(contours.size());
    i=0, CSize=0;
    for( ; i < contours.size(); i++ )
    {
      approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
      boundRect[i] = boundingRect( Mat(contours_poly[i]) );
      minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
      CSize++;
    }
    if(CSize!=0){
      for(i = 0; i< contours.size(); i++ )
      {
        if(contourArea(contours[i])>350){
          GlobalContourArea = contourArea(contours[i]);
          rectangle( frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,255), 2, 8, 0 );
          xLoc=center[i].x;
          yLoc=center[i].y;
          KFObj.track(xLoc, yLoc);
          GlobalContourSize=contours.size();
          KFObj.draw(frame);
        }
      }
    }
    else
    {
      cout<<".";
      KFObj.clearPoints();
    }
    Inflow(yLoc);
    Outflow(yLoc);
    get_count(yLoc);
    imshow("Frame",frame); //output
    std::cout << "Size:" << CSize << " " << "Area:" << GlobalContourArea << std::endl;
    Keyboard=waitKey(30);
  }
  return 0;
}
void Inflow(int yLoc){
  if(stateIn==0){
    if((yLoc>171 && yLoc<176) || (yLoc>176 && yLoc<180) ){
      stateIn=1;
    }
  }
  if(stateIn==1 && (yLoc>155 && yLoc<160)){

    stateIn=2;
  }
  if(stateIn==2){
      pedInCount+=GlobalContourSize;
      stateIn=0;
  }
}
void Outflow(int yLoc){
  if(stateOut==0){
    if((yLoc>155 && yLoc<160) || (yLoc>160 && yLoc<155)){
      stateOut=1;
    }
  }
  if(stateOut==1 && (yLoc>171 && yLoc<176)){
    stateOut=2;
  }
  if(stateOut==2){
        pedOutCount+=GlobalContourSize;
    stateOut=0;
  }
}
void get_count(int yLoc){
  if(currentState==0){
    if((yLoc > regionOneCeil) && (yLoc < regionTwoCeil)){//Object in Region one
      currentState=1;
      start_s=time(0);

    }
  }
  if(currentState==1){
    stop_s=time(0);
    if((yLoc > regionTwoCeil) && (yLoc < regionTwoFloor)){//Object in Region two
      if((int)stop_s-start_s < 2){
        currentState=2;
      }
      else
        currentState=0;
    }
  }
  if(currentState==2){
    cout<<currentState<<endl;
    currentState=0;
  }

}
