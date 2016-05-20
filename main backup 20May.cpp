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
int Keyboard, colorCount=0, GlobalContourSize=0;
int frameno;
bool danger;
int GlobalContourArea=0;
Point pt1 = Point(1,160), pt2 = Point(319,160);
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
void KFTracker::draw(Mat img, bool value){
  for (int i=0; i<kalman_vector.size()-1; i++) {
    if(value==true){
          line(img, kalman_vector[i], kalman_vector[i+1], Scalar(0,0,255), 3);
    }
    line(img, kalman_vector[i], kalman_vector[i+1], Scalar(0,255,0), 1);
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
//void coolflow(int yLoc);
int main()
{
  int start_s=clock();
  pedInCount=0; pedOutCount=0;
  KFTracker KFObj;
  bg = createBackgroundSubtractorMOG2();
  Mat frame, back, fore;
  int xLoc, yLoc, xLocca, yLocca, prevpedInCount=0, prevpedOutCount=0, i, CSize;
  int pedSouth=0, pedNorth=0, pedWest=0, pedEast=0;
  int stateNorth=0, stateSouth=0, stateWest=0, stateEast=0;
  Mat roiNorth, roiSouth, roiWest, roiEast;
  Point Px1, Px2, point;
  VideoCapture capture("s1.mp4");//input
  frameno = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
  //capture.set(CV_CAP_PROP_FPS, 5.0);
  //cout<<"Frame no: "<<frameno<<endl;
  bg->setNMixtures(5);
  bg->setDetectShadows(true);
  bg->setShadowValue(0);
  bg->setShadowThreshold(0.6);
  vector<vector<Point> > contours;
  vector<vector<Point> > contoursroiNorth;
  vector<vector<Point> > contoursroiSouth;
  vector<vector<Point> > contoursroiWest;
  vector<vector<Point> > contoursroiEast;
  namedWindow("Frame", WINDOW_AUTOSIZE);
  int con_size=0;
  while ((char)Keyboard!='q' && (char)Keyboard!=27)
  {
    capture >> frame;
    if(!capture.read(frame)){
      int stop_s=clock();
      //cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << endl;
    exit(0);
    }
    //imwrite("./outputFrameBG.jpg",frame);
    // cvtColor(frame, gray_image, CV_BGR2GRAY);
    // threshold(gray_image, binary_image, 190, 255, CV_THRESH_BINARY);
    bg->apply(frame, fore);
    //erode(binary_image,binary_image,Mat());
    //dilate(binary_image,binary_image,Mat());
    bg->getBackgroundImage(back);
    erode(fore,fore,Mat());
    dilate(fore,fore,Mat());
    dilate(fore,fore,Mat());
    roiNorth = fore(Rect(40,0,240,40));
    roiSouth = fore(Rect(40,195,240,40));
    roiWest = fore(Rect(0,40,40,155));
    roiEast = fore(Rect(275,40,40,155));

    findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(fore,contoursroiNorth,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(fore,contoursroiSouth,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(fore,contoursroiWest,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(fore,contoursroiEast,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

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
          GlobalContourArea = contourArea(contours[i]);
          rectangle( frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,255), 2, 8, 0 );
          Px2 = boundRect[i].br();
          xLoc=center[i].x;
          yLoc=center[i].y;
          xLocca=xLoc;
          yLocca=yLoc;
          KFObj.track(xLocca, yLocca);
          GlobalContourSize=contours.size();
          if(Px2.x<75 || Px2.x>270)
            danger=true;
          else
            danger=false;
          KFObj.draw(frame, danger);
        }
      }
    }
    else
    {
      KFObj.clearPoints();
    }
    //cout<<xLoc<<endl;
    //up to down
    Inflow(yLoc);
    Outflow(yLoc);
    //------------------------------------------------------------------------------------------
    if(stateSouth==0){
      if(yLoc>220 && yLoc<230){
        stateSouth=1;
      }
    }
    if(stateSouth==1 && (yLoc>200 && yLoc<210)){

      stateSouth=2;
    }
    if(stateSouth==2){
        pedSouth++;
      stateSouth=0;
    }
    //------------------------------------------------------------------------------------------
    if(stateNorth==0){
      if(yLoc>27 && yLoc<37){
        stateNorth=1;
      }
    }
    if(stateNorth==1 && (yLoc>7 && yLoc<17)){
      stateNorth=2;
    }
    if(stateNorth==2){
        pedNorth++;
      stateNorth=0;
    }
    //------------------------------------------------------------------------------------------
    if(stateWest==0){
      if(xLoc>7 && xLoc<17){
        stateWest=1;
      }
    }
    if(stateWest==1 && (xLoc>27 && xLoc<37)){

      stateWest=2;
    }
    if(stateWest==2){
        pedWest++;
      stateWest=0;
    }
    //------------------------------------------------------------------------------------------
    if(stateEast==0){
      if(xLoc>305 && xLoc<315){
        stateEast=1;
      }
    }
    if(stateEast==1 && (xLoc>285 && xLoc<295)){

      stateEast=2;
    }
    if(stateEast==2){
        pedEast++;
      stateEast=0;
    }
    //------------------------------------------------------------------------------------------


    //coolflow(yLoc);
    //if(prevpedOutCount!=pedOutCount || prevpedInCount!=pedInCount)
    //cout<<endl<<"*"<<yLoc<<"*"<<endl;

    //down to up
    //line(frame, pt1, pt2, Scalar(0,255,0), 2, 8, 0);
    stringstream ss;
    rectangle(frame, cv::Point(0, 214), cv::Point(65,239),
              cv::Scalar(255,255,255), -1);
    ss << pedNorth+pedSouth+pedEast+pedWest;
    string frameNumberString = ss.str();
    putText(frame, frameNumberString.c_str(), cv::Point(10, 225),
            FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
    //imwrite("./outputFastMesConv.jpg",frame);
    //imwrite("./outputdanger.jpg",frame);
    imshow("Frame",frame); //output
    //imshow("Frame",fore); //output
    //if(prevpedOutCount!=pedOutCount || prevpedInCount!=pedInCount)
       //cout<<pedInCount+pedOutCount<<endl;
    //cout<<pedNorth+pedSouth+pedEast+pedWest<<endl;
    prevpedInCount=pedInCount;
    prevpedOutCount=pedOutCount;
    frameno--;
    Keyboard=waitKey(1);
  }
   //cout<<pedInCount+pedOutCount<<endl;
  //cout<<"Up Count: "<<pedInCount<<endl<<"Down Count: "<<pedOutCount<<endl;
  //cout<<pedInCount+pedOutCount<<endl;
  return 0;
}
// void coolflow(int yLoc){
//     if
// }
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
    //cout<<endl<<GlobalContourSize<<endl;
    //if(GlobalContourSize>=1 && GlobalContourSize<3)
    // if(GlobalContourArea>650)
    //     pedInCount+=2;
    // else
      pedInCount+=GlobalContourSize;
    //else
    //pedInCount+=2;
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
    if((yLoc>155 && yLoc<160) || (yLoc>160 && yLoc<155)){
      stateOut=1;
      //cout<<yLoc<<"-";
    }
  }
  if(stateOut==1 && (yLoc>171 && yLoc<176)){
    stateOut=2;
  }
  if(stateOut==2){
    //cout<<endl<<GlobalContourSize<<endl;
    //if(GlobalContourSize>=1 && GlobalContourSize<3)
    // if(GlobalContourArea>650)
    //     pedOutCount+=2;
    // else
        pedOutCount+=GlobalContourSize;
    //else
    //pedOutCount+=2;
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
