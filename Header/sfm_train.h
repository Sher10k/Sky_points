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
//#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class SFM_Reconstruction 
{
private:
    
    unsigned long Match_find_SIFT(vector<KeyPoint>,
                                  vector<KeyPoint>,
                                  Mat,
                                  Mat,
                                  vector<KeyPoint> *,
                                  vector<KeyPoint> *,
                                  vector<DMatch> *);
    
public:
    
    VideoCapture CAPsfm;
    int width_frame, height_frame;
    Mat frame1, frame2, frame4;
        // Keypoints
    Ptr<SIFT> detectorSIFT = cv::xfeatures2d::SIFT::create();   // 0, 4, 0.04, 10, 1.6
    std::vector<KeyPoint> keypoints1_SIFT, keypoints2_SIFT;
    Mat descriptors1_SIFT, descriptors2_SIFT;
    unsigned long numKeypoints;
    vector<DMatch> good_matches;
    vector<KeyPoint> good_points1, good_points2;
    
    
    SFM_Reconstruction(VideoCapture *);
    void setParam(VideoCapture *);
    void f1Tof2();
    void detectKeypoints(Mat *);
    void goodClear();
    void matchKeypoints();
    
};

#endif // SFM_TRAIN_H
