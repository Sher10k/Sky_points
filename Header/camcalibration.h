#ifndef CAMCALIBRATION_H
#define CAMCALIBRATION_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>


using namespace std;
using namespace cv;

enum {  CHESS_BOARD     = 0,    // Calibration with chessboard
        CHARUCO_BOARD   = 1     // Calibration with ChArUco Boards
     };

class CalibrationCamera
{
private:
    
public:
    
    VideoCapture CAP;
    Matx33d cameraMatrix;           //= Matx33d( 10,     0,  FRAME_WIDTH/2, 0,     10,  FRAME_HEIGHT/2, 0,     0,  1);
    Matx<double, 1, 5> distCoeffs;  //= Matx<double, 1, 5>(0.0, 0.0, 0.0, 0.0, 0.0);  // (k1, k2, p1, p2, k3)
    vector<Mat> rvecs, tvecs;
    int calibrationFlags;
    int width_frame, height_frame;
    
    CalibrationCamera(VideoCapture *);
    void setParam(VideoCapture *);
    void printParam();
    void Read_from_file(int);
    void calibrCameraChess( int numCornersHor,                 // Кол-во углов по вертикале и горизонтале для метода ChArUco и на 1 меньше чем кол-во 
                            int numCornersVer,                 // квадратов по вертикале и горизонтале, для метода калибровки по chessboard
                            unsigned int nFrames);             // Number of calibration frames
    void calibrCameraChArUco( int numCellX,                // Кол-во углов по вертикале и горизонтале для метода ChArUco и на 1 меньше чем кол-во 
                              int numCellY,                 // квадратов по вертикале и горизонтале, для метода калибровки по chessboard
                              float squareLength,
                              float markerLength,
                              int dictionaryId,
                              unsigned int nFrames);     // Number of calibration frames
    
};

#endif // CAMCALIBRATION_H
