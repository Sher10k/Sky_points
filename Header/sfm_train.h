#ifndef SFM_TRAIN_H
#define SFM_TRAIN_H

//#define CERES_FOUND true
#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <iostream>
#include <opencv2/core.hpp>
//#include <opencv2/core/mat.hpp>
//#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/sfm.hpp>
//#include <opencv2/sfm/simple_pipeline.hpp>
#include <opencv2/sfm/triangulation.hpp>

//#include <vector>


using namespace std;
using namespace cv;
using namespace cv::sfm;
using namespace cv::xfeatures2d;

class SFM_Reconstruction 
{
private:
    
    unsigned long Match_find_SIFT(vector<cv::KeyPoint>,
                                  vector<cv::KeyPoint>,
                                  cv::Mat,
                                  cv::Mat,
                                  vector<cv::KeyPoint> *,
                                  vector<cv::KeyPoint> *,
                                  vector<cv::DMatch> *);
    
public:
    
        // Camera 
    cv::VideoCapture CAPsfm;
    int width_frame, height_frame;
    cv::Mat frame1, frame2, frame4;
    
        // Keypoints
    //unsigned long key_num = 10;  // Num keypoint
    //Ptr<FeatureDetector> detector = ORB::create(static_cast<int>(key_num), 1.2f, 8, 31, 0, 4, ORB::HARRIS_SCORE, 31);   // HARRIS_SCORE, FAST_SCORE
    //Ptr<SURF> detectorSURF = cv::xfeatures2d::SURF::create(static_cast<double>(key_num), 4, 3, true, false);   // cv::xfeatures2d::
    //Ptr<SURF> detectorSURF = cv::xfeatures2d::SURF::create(100);
    cv::Ptr<cv::xfeatures2d::SIFT> detectorSIFT = cv::xfeatures2d::SIFT::create();       // 0, 4, 0.04, 10, 1.6
    vector<cv::KeyPoint> keypoints1_SIFT, keypoints2_SIFT;         // Key points
    cv::Mat descriptors1_SIFT, descriptors2_SIFT;                       // Descriptors key points
    unsigned long numKeypoints;                                     // Number key points
    vector<cv::DMatch> good_matches;                                    // Good matches between frames
    vector<cv::KeyPoint> good_points1, good_points2;                    // Good points satisfying the threshold
    vector<cv::Point2f> points1, points2;                           // KeyPoints -> Points(x, y)
    vector <cv::Mat> pointsMass;                                    // Array points both frames
    
        // Fundamental matrix
    cv::Mat F = cv::Mat( 3, 3, CV_32FC1 );
    
        // Projection matrices for each camera
    cv::Mat Pt1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Pt2 = cv::Mat::eye(3, 4, CV_64F);
    vector<cv::Mat> Proj;             // Matx34d  // Vector of projection matrices for each camera
//    Ps[0] = cv::Mat(3, 4, CV_64F);
//    Ps[1] = cv::Mat(3, 4, CV_64F);
    
        // 3D points
    cv::Mat points3D;
    
    
    SFM_Reconstruction(cv::VideoCapture *);
    void setParam(cv::VideoCapture *);
    void f1Tof2();
    void detectKeypoints(cv::Mat *);
    void goodClear();
    void matchKeypoints();
    void fundametalMat();
    void projectionsMat();
    void triangulationPoints();
    
};

#endif // SFM_TRAIN_H
