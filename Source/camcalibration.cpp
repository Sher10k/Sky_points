#include "Header/camcalibration.h"

#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

CalibrationCamera::CalibrationCamera(Matx33d data_cameraMatrix,
                                     Matx<double, 1, 5> data_distCoeffs,
                                     vector<Mat> data_rvecs,
                                     vector<Mat> data_tvecs,
                                     unsigned int data_width,
                                     unsigned int data_height)
{
    setParam(data_cameraMatrix, data_distCoeffs, data_rvecs, data_tvecs, data_width, data_height);
}

void CalibrationCamera::setParam(Matx33d data_cameraMatrix, 
                                 Matx<double, 1, 5> data_distCoeffs, 
                                 vector<Mat> data_rvecs, 
                                 vector<Mat> data_tvecs,
                                 unsigned int data_width,
                                 unsigned int data_height)
{
    //cameraMatrix = Matx33d( 10,     0,  data_width/2, 0,     10,  data_height/2, 0,     0,  1);
    //distCoeffs = Matx<double, 1, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cameraMatrix(i, j) = data_cameraMatrix(i, j);
    
    for (int j = 0; j < 5; j++)
        distCoeffs(0, j) = data_distCoeffs(0, j);
    
    rvecs = data_rvecs;
    tvecs = data_tvecs;
    width_frame = data_width;
    height_frame = data_height;
}

void CalibrationCamera::printParam()
{
    cout << "cameraMatrix = " << endl;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            cout << " " << cameraMatrix(i, j) << "\t\t";
        cout << endl;
    }
    
    cout << "distCoeffs = " << endl;
    for (int j = 0; j < 5; j++)
        cout << " " << distCoeffs(0, j) << "\t";
    cout << endl;
    
    cout << "rvecs = " << endl;
    for (unsigned long i = 0; i < rvecs.size(); i++)
    {
        cout << i << " = [  ";
        for (int j = 0; j < rvecs[i].rows; j++)
            cout << rvecs[i].at<double>(j) << "   ";
        cout << " ]" << endl;
    }
    
    cout << "tvecs = " << endl;
    for (unsigned long i = 0; i < tvecs.size(); i++)
    {
        cout << i << " = [  ";
        for (int j = 0; j < tvecs[i].rows; j++)
            cout << tvecs[i].at<double>(j) << "   ";
        cout << " ]" << endl;
    }
}




