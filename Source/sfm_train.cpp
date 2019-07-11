
#include "Header/sfm_train.h"

using namespace std;
using namespace cv;

SFM_Reconstruction::SFM_Reconstruction(VideoCapture *data_CAP)
{
    SFM_Reconstruction::setParam(data_CAP);
}

void SFM_Reconstruction::setParam(VideoCapture *data_CAP)
{
    CAPsfm = *data_CAP;
    Mat frame;
    CAPsfm.read(frame);
    width_frame = frame.cols;
    height_frame = frame.rows;
    frame1 = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);
    frame2 = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);
    frame4 = Mat::zeros(Size(2 * width_frame, height_frame), CV_8UC3);
    numKeypoints = 0;
}

void SFM_Reconstruction::f1Tof2()
{
    frame1.copyTo( frame2 );
    keypoints2_SIFT = keypoints1_SIFT;
    descriptors2_SIFT = descriptors1_SIFT;
}

void SFM_Reconstruction::detectKeypoints(Mat *frame)
{
    frame1 = *frame;
    detectorSIFT->detectAndCompute(frame1, noArray(), keypoints1_SIFT, descriptors1_SIFT);
}

void SFM_Reconstruction::matchKeypoints()
{
    if ((keypoints1_SIFT.size() != 0) && (keypoints2_SIFT.size() != 0))
    {
        numKeypoints = Match_find_SIFT(keypoints1_SIFT,
                                       keypoints2_SIFT,
                                       descriptors1_SIFT,
                                       descriptors2_SIFT,
                                       &good_points1,
                                       &good_points2,
                                       &good_matches);
    }
}

void SFM_Reconstruction::goodClear()
{
    good_matches.clear();
    good_points1.clear();
    good_points2.clear();
}

// SIFT Find matches between points
unsigned long SFM_Reconstruction::Match_find_SIFT(vector<KeyPoint> kpf1,
                              vector<KeyPoint> kpf2,
                              Mat dpf1,
                              Mat dpf2,
                              vector<KeyPoint> *gp1,
                              vector<KeyPoint> *gp2,
                              vector<DMatch> *gm)
{
    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    vector<DMatch> matches;
    DMatch dist1, dist2;
 
    matcher->match(dpf1, dpf2, matches, noArray());     // Matches key points of the frame with key points of the frame2

    // Sort match
    unsigned long temp = 0;
    unsigned long kn = static_cast<unsigned long>(matches.size());
    for (unsigned long i = 0; i < kn - 1; i++) { //key_num - 1
        dist1 = matches[i];
        for (unsigned long j = i + 1; j < kn; j++) {
            dist2 = matches[j];
            if (dist2.distance < dist1.distance) {
                dist1 = matches[j];
                temp = j;
            }
            if (j == kn - 1) {
                matches[temp] = matches[i];
                matches[i] = dist1;
                break;
            }
        }
    }
    //-- Вычисление максимального и минимального расстояния среди всех дескрипторов в пространстве признаков
    float max_dist = 0, min_dist = 100;
    for(size_t i = 0; i < matches.size(); i++ ) {
        float dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    cout << "-- Max dist : " << max_dist << endl;
    cout << "-- Min dist : " << min_dist << endl;
    // Selection of key points on both frames satisfying the threshold
    temp = 0;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance <  10 * min_dist) {   // threshold
            int new_i = static_cast<int>( gp1->size() );
            gp1->push_back( kpf1[ static_cast<unsigned long>( matches[i].queryIdx ) ] );    // queryIdx
            gp2->push_back( kpf2[ static_cast<unsigned long>( matches[i].trainIdx ) ] );    // trainIdx
            gm->push_back( DMatch(new_i, new_i, 0) );
            temp++;
        }
    }
    cout << "-- Temp : " << temp << endl << endl;
    return temp;
}
