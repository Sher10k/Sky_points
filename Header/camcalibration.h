#ifndef CAMCALIBRATION_H
#define CAMCALIBRATION_H

//#include <iostream>
#include <opencv2/core.hpp>
//#include <opencv2/core/mat.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


class CalibrationCamera {
private:
    Matx33d cameraMatrix;           //= Matx33d( 10,     0,  FRAME_WIDTH/2, 0,     10,  FRAME_HEIGHT/2, 0,     0,  1);
    Matx<double, 1, 5> distCoeffs;  //= Matx<double, 1, 5>(0.0, 0.0, 0.0, 0.0, 0.0);  // (k1, k2, p1, p2, k3)
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    unsigned int width_frame, height_frame;
    
public:
    CalibrationCamera(Matx33d, Matx<double, 1, 5>, vector<Mat>, vector<Mat>, unsigned int, unsigned int);
    void setParam(Matx33d, Matx<double, 1, 5>, vector<Mat>, vector<Mat>, unsigned int, unsigned int);
    void printParam();
};


#endif // CAMCALIBRATION_H
