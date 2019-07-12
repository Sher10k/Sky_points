#ifndef CAMCALIBRATION_H
#define CAMCALIBRATION_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

class CalibrationCamera
{
private:
    
public:
    
    cv::VideoCapture CAP;
    cv::Matx33d cameraMatrix;           //= Matx33d( 10,     0,  FRAME_WIDTH/2, 0,     10,  FRAME_HEIGHT/2, 0,     0,  1);
    cv::Matx<double, 1, 5> distCoeffs;  //= Matx<double, 1, 5>(0.0, 0.0, 0.0, 0.0, 0.0);  // (k1, k2, p1, p2, k3)
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    int width_frame, height_frame;
    
    CalibrationCamera(cv::VideoCapture *);
    void setParam(cv::VideoCapture *);
    void printParam();
    void calibratCamera(int, int, int);
    
};

#endif // CAMCALIBRATION_H
