  #include<iostream>
  #include<stdlib.h>
  #include "algorithm"
  #include <vector>
  #include<ctime>
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
  int Keyboard, GlobalContourSize=0;
  int GlobalContourArea=0;
  int GlobalCounter=0;
  int currentStateUp=0, currentStateDown=0;
  time_t start_s_up, stop_s_up;
  time_t start_s_down, stop_s_down;
  Point pt1Up = Point(1,135), pt2Up = Point(319,135);
  Point pt1Down = Point(1,185), pt2Down = Point(319,185);
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
  }
  //void Inflow(int yLoc);
  //void Outflow(int yLoc);
  void updateCounter(){
    if(GlobalContourSize==1){
      if(GlobalContourArea<750){
        GlobalCounter++;
      }
      else{
        GlobalCounter+=2;
      }
    }
    else{
      GlobalCounter+=GlobalContourSize;
    }
    cout<<GlobalCounter<<endl;
  }
  void mycounterUp(int yLoc){
    //-------------- Counting Object Moving Down
    if(currentStateUp==0){
      if(yLoc>120 && yLoc<130){
        currentStateUp=1;
        start_s_up=time(0);
      }
    }
    if(currentStateUp==1){
      stop_s_up=time(0);
      if((int)stop_s_up - start_s_up > 1){
        currentStateUp=0;
      }
      if(yLoc>132 && yLoc<142){
        if((int)stop_s_up - start_s_up < 2){
          currentStateUp=2;
        }
        else{
          currentStateUp=0;
        }
      }
    }
    if(currentStateUp==2){
      updateCounter();
      currentStateUp=0;
    }
}
void mycounterDown(int yLoc){
    //-------------- Counting Object Moving Up
    //cout<<yLoc<<" - "<<currentStateDown<<endl;
    if(currentStateDown==0){
      if(yLoc>198 && yLoc<208){
        currentStateDown=1;
        start_s_down=time(0);
      }
    }
    if(currentStateDown==1){
      stop_s_down=time(0);
      if((int)stop_s_down - start_s_down > 1){
        currentStateDown=0;
      }
      if(yLoc>186 && yLoc<196){
        if((int)stop_s_down - start_s_down < 2){
          currentStateDown=2;
        }
        else{
          currentStateDown=0;
        }
      }
    }
    if(currentStateDown==2){
      updateCounter();
      currentStateDown=0;
    }
  }
  int main()
  {
    KFTracker KFObj;
    bg = createBackgroundSubtractorMOG2();
    Mat frame, back, fore;
    int xLoc, yLoc, prevpedInCount=0, prevpedOutCount=0, i, CSize;
    Point Px1, Px2, point;
    VideoCapture capture("dooagain.mp4");//input
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
      if(!capture.read(frame)){
        exit(0);
      }
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

      for(i=0 ; i < contours.size(); i++ )
      {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
      }
      if(contours.size()!=0){
        for(i = 0; i< contours.size(); i++ )
        {
          if(contourArea(contours[i])>350){
            GlobalContourSize=contours.size();
            GlobalContourArea = contourArea(contours[i]);
            rectangle( frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,255), 2, 8, 0 );
            Px2 = boundRect[i].br();
            xLoc=center[i].x;
            yLoc=center[i].y;
            KFObj.track(xLoc, yLoc);
            KFObj.draw(frame);
          }
        }
      }
      else
      {
        yLoc=0;
        KFObj.clearPoints();
      }
      mycounterUp(yLoc);
      mycounterDown(yLoc);
      line(frame, pt1Up, pt2Up, Scalar(0,255,0), 2, 8, 0);
      line(frame, pt1Down, pt2Down, Scalar(0,255,0), 2, 8, 0);
      imshow("Frame",frame); //output
      Keyboard=waitKey(30);
    }
    return 0;
  }
