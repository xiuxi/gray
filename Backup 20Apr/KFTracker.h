#ifndef KFTRACKER_H
#define KFTRACKER_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;

class KFTracker{
public:
  KalmanFilter KF;
  Mat state; /* (phi, delta_phi) */
  Mat processNoise;
  Mat measurement;

  vector<Point>points_vector, kalman_vector;
  bool init;

  KFTracker();
  virtual ~KFTracker();
  void track(int x, int y);
  void initializeKF(int x, int y);
  void draw(Mat img);
protected:
private:
};
#endif
