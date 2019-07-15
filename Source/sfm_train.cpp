
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

void SFM_Reconstruction::goodClear()
{
    good_matches.clear();
    good_points1.clear();
    good_points2.clear();
}

void SFM_Reconstruction::matchKeypoints()
{
    goodClear();
    
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
        
    }
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
    /*for (size_t i = 0; i < matches.size(); i++)
    {
        for (size_t j = 0; j < matches.size(); j++) 
        {
            if (matches[i].distance < 0.1f * matches[j].distance) 
            {
                gm->push_back(matches[i]);
                gp1->push_back( kpf1[ static_cast<unsigned long>( matches[i].queryIdx ) ] );    // queryIdx
                gp2->push_back( kpf2[ static_cast<unsigned long>( matches[i].trainIdx ) ] );    // trainIdx
                temp ++;
            }
        }
    }*/
    
    cout << "-- Temp : " << temp << endl << endl;
    return temp;
}

void SFM_Reconstruction::homo_fundam_Mat(Matx33d K_1)  //--------------------------------------------- DEBUG -----!!!!!!!!!!!
{
    if (numKeypoints > 7 )
    {
        cout << "numKeypoints before homo_mask = " << numKeypoints << endl;
        numKeypoints = 0;
        retval = findHomography(points1, points2, RANSAC, 100, homo_mask);
        F = findFundamentalMat(points1, points2, FM_RANSAC, 1.0, 0.99, Fundam_mask);
        for (int i = 0; i < homo_mask.rows; i++)
        {
            if  (homo_mask.at<uchar>(i) == 0)
            {
                points1.erase(points1.begin() + i);
                points2.erase(points2.begin() + i);
            } 
            else numKeypoints++;
        }
        
        convertPointsToHomogeneous(points1, points1H);
        convertPointsToHomogeneous(points2, points2H);
        /*float X = 0, Y = 0, W = 0;
        for (int n = 0; n < points1H.rows; n++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    X += static_cast<float>(K_1(i ,j)) * points1H.at<float>(n, 0);
                }
            }
        }*/
        
        cout << "numKeypoints after homo_mask = " << numKeypoints << endl;
        FileStorage fundam;
        fundam.open("Fundamental_matrix.txt", FileStorage::WRITE);
        fundam << "homography_matrix" << retval;
        fundam << "homography_mask" << homo_mask;
        fundam << "fundamental_matrix" << F;
        fundam << "fundamental_mask" << Fundam_mask;
        fundam << "K-1" << K_1;
        fundam << "points1H" << points1H;
        fundam << "points2H" << points2H;
        fundam.release();
        cout << " --- Fundamental matrix written into file: Fundamental_matrix.txt" << endl << endl;
        
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
    
    cv::sfm::triangulatePoints(pointsMass, Proj, points3D);
    
    /*for (int i = 0; i < points3D.cols; i++){
        for (int j = 0; j < points3D.rows; j++){
            cout << " " << points3D.at<double>(i, j) << " ";
        }
        cout << endl;
    }*/
    cv::FileStorage poins3D;      // Вывод в файл 3д точек
    poins3D.open("poins3D_XYZ.txt", cv::FileStorage::WRITE);
    poins3D << "points3D" << points3D;
    poins3D.release();
    cout    << "points3D.cols = " << points3D.cols << endl;
    cout << " --- 3D points written into file: poins3D_XYZ.txt" << endl << endl;
}

void SFM_Reconstruction::opticalFlow(Mat *f, Mat *fc, int win, int vecS)
{
    img2Original = *f;
    flow = Mat(img2Original.cols, img2Original.rows, CV_32FC2);
    cvtColor(*f, frameGREY, COLOR_BGR2GRAY);
    cvtColor(*fc, frameCacheGREY, COLOR_BGR2GRAY);
    calcOpticalFlowFarneback(frameGREY, frameCacheGREY, flow, 0.9, 1, 12, 2, 8, 1.7, 0);//OPTFLOW_FARNEBACK_GAUSSIAN
    
    //cv::normalize(flow, flow, 1, 0, NORM_L2, -1, noArray());
    
    for (int y = 0; y < img2Original.rows; y += win) {
        for (int x = 0; x < img2Original.cols; x += win) {
            // get the flow from y, x position * 3 for better visibility
            const Point2f flowatxy = flow.at<Point2f>(y, x) * vecS;
            // draw line at flow direction
            line(img2Original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 200, 0));
            // draw initial point
            circle(img2Original, Point(x, y), 1, Scalar(0, 0, 0), -1);
        }
    }
}
