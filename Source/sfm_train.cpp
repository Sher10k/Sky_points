
#include "Header/sfm_train.h"

SFM_Reconstruction::SFM_Reconstruction(cv::VideoCapture *data_CAP)
{
    SFM_Reconstruction::setParam(data_CAP);
}

void SFM_Reconstruction::setParam(cv::VideoCapture *data_CAP)
{
    CAPsfm = *data_CAP;
    cv::Mat frame;
    CAPsfm.read(frame);
    width_frame = frame.cols;
    height_frame = frame.rows;
    frame1 = cv::Mat::zeros(cv::Size(width_frame, height_frame), CV_8UC3);
    frame2 = cv::Mat::zeros(cv::Size(width_frame, height_frame), CV_8UC3);
    frame4 = cv::Mat::zeros(cv::Size(2 * width_frame, height_frame), CV_8UC3);
    numKeypoints = 0;
}

void SFM_Reconstruction::f1Tof2()
{
    frame1.copyTo( frame2 );
    keypoints2_SIFT = keypoints1_SIFT;
    descriptors2_SIFT = descriptors1_SIFT;
}

void SFM_Reconstruction::detectKeypoints(cv::Mat *frame)
{
    frame1 = *frame;
    detectorSIFT->detectAndCompute(frame1, cv::noArray(), keypoints1_SIFT, descriptors1_SIFT);
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
        points1.clear();
        points2.clear();
        points1.resize(numKeypoints);
        points2.resize(numKeypoints);
        cv::KeyPoint::convert(good_points1, points1);     // Convert from KeyPoint to Point2f
        cv::KeyPoint::convert(good_points2, points2);
        
        pointsMass.clear();
        pointsMass.resize(2);
        pointsMass[0] = cv::Mat(2, static_cast<int>(numKeypoints), CV_64F);
        pointsMass[1] = cv::Mat(2, static_cast<int>(numKeypoints), CV_64F);
        for (int i = 0; i < static_cast<int>(numKeypoints); i++) {         // Unioning the key points in new variable
            pointsMass[0].at<float>(0, i) = points1[static_cast<unsigned long>(i)].x;
            pointsMass[0].at<float>(1, i) = points1[static_cast<unsigned long>(i)].y;
            pointsMass[1].at<float>(0, i) = points2[static_cast<unsigned long>(i)].x;
            pointsMass[1].at<float>(1, i) = points2[static_cast<unsigned long>(i)].y;
            /*cout    << "points2frame [0](0, " << i << ") = " << points2frame[0].at<float>(0, i) << endl
                    << "points2frame [0](1, " << i << ") = " << points2frame[0].at<float>(1, i) << endl
                    << "points2frame [1](0, " << i << ") = " << points2frame[1].at<float>(0, i) << endl
                    << "points2frame [1](1, " << i << ") = " << points2frame[1].at<float>(1, i) << endl
                    << endl;*/
        }
    }
}

void SFM_Reconstruction::goodClear()
{
    good_matches.clear();
    good_points1.clear();
    good_points2.clear();
}

// SIFT Find matches between points
unsigned long SFM_Reconstruction::Match_find_SIFT(vector<cv::KeyPoint> kpf1,
                                                  vector<cv::KeyPoint> kpf2,
                                                  cv::Mat dpf1,
                                                  cv::Mat dpf2,
                                                  vector<cv::KeyPoint> *gp1,
                                                  vector<cv::KeyPoint> *gp2,
                                                  vector<cv::DMatch> *gm)
{
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();
    vector<cv::DMatch> matches;
    cv::DMatch dist1, dist2;
 
    matcher->match(dpf1, dpf2, matches, cv::noArray());     // Matches key points of the frame with key points of the frame2

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
            gm->push_back( cv::DMatch(new_i, new_i, 0) );
            temp++;
        }
    }
    cout << "-- Temp : " << temp << endl << endl;
    return temp;
}

void SFM_Reconstruction::fundametalMat()
{
    if (numKeypoints > 7 )
    {
        F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 1.0, 0.99, cv::noArray());
        
    } else {
        CV_Error(cv::Error::StsOutOfRange, "Not enough keypoints");
    }
}


void SFM_Reconstruction::projectionsMat()
{
    if ( !F.empty() )
    {
        Proj.clear();
        
        cv::sfm::projectionsFromFundamental(F, Pt1, Pt2);
        
        Proj.push_back(Pt1);
        Proj.push_back(Pt2);
        
    } else {
        CV_Error(cv::Error::StsOutOfRange, "Fundamental matrix is empty");
    }
}

void SFM_Reconstruction::triangulationPoints()
{
    cv::sfm::triangulatePoints(pointsMass, Proj, points3D);
    
    for (int i = 0; i < points3D.cols; i++){
        for (int j = 0; j < points3D.rows; j++){
            cout << " " << points3D.at<double>(i, j) << " ";
        }
        cout << endl;
    }
    cv::FileStorage poins3D;      // Вывод в файл 3д точек
    poins3D.open("poins3D_XYZ.txt", cv::FileStorage::WRITE);
    poins3D << "points3D" << points3D;
    poins3D.release();
    cout    << "points3D.cols = " << points3D.cols << endl;
    cout << " --- 3D points written into file: poins3D_XYZ.txt" << endl << endl;
}

