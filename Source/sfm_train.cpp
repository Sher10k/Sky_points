
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

void SFM_Reconstruction::Reconstruction3D(Mat *data_frame1, Mat *data_frame2, Matx33d K)
{
    points3D *= 0;
//--- STEP 1 --- Detection and calculation
    keypoints1_SIFT.clear();
    keypoints2_SIFT.clear();
    descriptors1_SIFT *= 0;
    descriptors2_SIFT *= 0;
    if (!data_frame1->empty()) {
        data_frame1->copyTo(frame1);
        detectorSIFT->detectAndCompute(frame1, cv::noArray(), keypoints1_SIFT, descriptors1_SIFT);
    }
    if (!data_frame2->empty()) {
        data_frame2->copyTo(frame2);
        detectorSIFT->detectAndCompute(frame2, cv::noArray(), keypoints2_SIFT, descriptors2_SIFT);
    }
//--- STEP 2 --- Matcher keypoints
    good_matches.clear();
    good_points1.clear();
    good_points2.clear();
    points1.clear();
    points2.clear();
    numKeypoints = 0;
    if ((keypoints1_SIFT.size() != 0) && (keypoints2_SIFT.size() != 0))
    {
        numKeypoints = Match_find_SIFT(keypoints1_SIFT,
                                       keypoints2_SIFT,
                                       descriptors1_SIFT,
                                       descriptors2_SIFT,
                                       &good_points1,
                                       &good_points2,
                                       &good_matches);
        points1.resize(numKeypoints);
        points2.resize(numKeypoints);
        KeyPoint::convert(good_points1, points1);     // Convert from KeyPoint to Point2f
        KeyPoint::convert(good_points2, points2);
//--- STEP 3 --- Find essential matrix
        if (numKeypoints > 7)
        {
            E = findEssentialMat(points1, points2, K, RANSAC, 0.999, 1.0, Essen_mask);
//--- STEP 4 --- Decompose essential matrix
            //decomposeEssentialMat(E, R1, R2, t);
            recoverPose(E, points1, points2, K, R, t, 600, noArray(), points3D);
            
            for (int i = 0; i < points3D.cols; i++){
                cout << "3Dpoint[ " << i << " ] =";
                for (int j = 0; j < points3D.rows; j++){
                    points3D.at<double>(j, i) /= points3D.at<double>(3, i);
                    cout << " " << points3D.at<double>(j, i) << " ";
                }
                cout << endl;
            }
            
            FileStorage SFM_Result;
            SFM_Result.open("SFM_Result.txt", FileStorage::WRITE);
            SFM_Result << "E" << E;
            SFM_Result << "points1" << points1;
            SFM_Result << "points2" << points2;
            SFM_Result << "K" << K;
            SFM_Result << "R" << R;
            SFM_Result << "t" << t;
            SFM_Result << "points3D" << points3D;
            SFM_Result.release();
            cout << " --- SFM_Result written into file: SFM_Result.txt" << endl;
            
            
            drawMatches(frame1, good_points1, frame2, good_points2, good_matches, frame4);
            imshow("1-2 frame", frame4);
        } 
        else 
        {
            if (!data_frame1->empty()) drawKeyPoints(&frame1, &keypoints1_SIFT);
            if (!data_frame2->empty()) drawKeyPoints(&frame2, &keypoints2_SIFT);
            Rect r1(0, 0, frame1.cols, frame1.rows);
            Rect r2(frame2.cols, 0, frame2.cols, frame2.rows);
            frame1.copyTo(frame4( r1 ));
            frame2.copyTo(frame4( r2 ));
            imshow("1-2 frame", frame4);
            cout << "Not enough keypoints" << endl;
        }
    } 
    else 
    {
        if (!data_frame1->empty()) drawKeyPoints(&frame1, &keypoints1_SIFT);
        if (!data_frame2->empty()) drawKeyPoints(&frame2, &keypoints2_SIFT);
        Rect r1(0, 0, frame1.cols, frame1.rows);
        Rect r2(frame2.cols, 0, frame2.cols, frame2.rows);
        frame1.copyTo(frame4( r1 ));
        frame2.copyTo(frame4( r2 ));
        imshow("1-2 frame", frame4);
        cout <<"No keypoints found in one of the frames" << endl;
    }
}

void SFM_Reconstruction::destroyWinSFM()
{
    namedWindow("1-2 frame", WINDOW_AUTOSIZE);
    destroyWindow("1-2 frame");
}

// SIFT Find matches between points
unsigned long SFM_Reconstruction::Match_find_SIFT( vector< KeyPoint > kpf1,
                                                   vector< KeyPoint > kpf2,
                                                   Mat dpf1,
                                                   Mat dpf2,
                                                   vector< KeyPoint > *gp1,
                                                   vector< KeyPoint > *gp2,
                                                   vector< DMatch > *gm)
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
    //cout << "-- Max dist : " << max_dist << endl;
    //cout << "-- Min dist : " << min_dist << endl;
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
    
    cout << " --- Temp : " << temp << endl;
    return temp;
}

void SFM_Reconstruction::drawKeyPoints( Mat *f, 
                                        vector< KeyPoint > *kp)
{
    vector< Point2f > drawPoint;
    KeyPoint::convert(*kp, drawPoint);
    RNG rng(12345);
    for (size_t i = 0; i < drawPoint.size(); i++) {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        circle(*f, drawPoint[i], 3, color, 1, LINE_8, 0);
        //putText( *f, to_string(i), drawPoint[i], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 2);
    }
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



void SFM_Reconstruction::homo_fundam_Mat(Matx33d K, Matx33d K_1)  //--------------------------------------------- DEBUG -----!!!!!!!!!!!
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
        for (int i = 0; i < points1H.rows; i++)
        {
            
//            cout << "pointBefore[ " << i << " ] = " 
//                 << points1H.at<float>(i, 0) << "; " 
//                 << points1H.at<float>(i, 1) << "; " 
//                 << points1H.at<float>(i, 2) << "; " << endl;
            points1H.at<float>(i, 0) = (static_cast<float>(K_1(0 ,0)) * points1H.at<float>(i, 0)) + static_cast<float>(K_1(0, 2));
            points1H.at<float>(i, 1) = (static_cast<float>(K_1(1 ,1)) * points1H.at<float>(i, 1)) + static_cast<float>(K_1(1, 2));
            points1H.at<float>(i, 2) = static_cast<float>(K_1(2 ,2)) * points1H.at<float>(i, 2);
            points2H.at<float>(i, 0) = (static_cast<float>(K_1(0 ,0)) * points2H.at<float>(i, 0)) + static_cast<float>(K_1(0, 2));
            points2H.at<float>(i, 1) = (static_cast<float>(K_1(1 ,1)) * points2H.at<float>(i, 1)) + static_cast<float>(K_1(1, 2));
            points2H.at<float>(i, 2) = static_cast<float>(K_1(2 ,2)) * points2H.at<float>(i, 2);
            points1[static_cast<size_t>(i)].x = points1H.at<float>(i, 0);
            points1[static_cast<size_t>(i)].y = points1H.at<float>(i, 1);
            points2[static_cast<size_t>(i)].x = points2H.at<float>(i, 0);
            points2[static_cast<size_t>(i)].y = points2H.at<float>(i, 1);
        }
        
        //F = findFundamentalMat(points1, points2, FM_RANSAC, 1.0, 0.99, Fundam_mask);
        E = findEssentialMat(points1, points2, K, RANSAC, 0.999, 1.0, Essen_mask);
        decomposeEssentialMat(E, R1, R2, t);
        cout << "numKeypoints after homo_mask = " << numKeypoints << endl;
        FileStorage fundam;
        fundam.open("Fundamental_matrix.txt", FileStorage::WRITE);
        fundam << "homography_matrix" << retval;
        fundam << "homography_mask" << homo_mask;
        fundam << "fundamental_matrix" << F;
        fundam << "fundamental_mask" << Fundam_mask;
        fundam << "Essential_matrix" << E;
        fundam << "Essential_mask" << Essen_mask;
        fundam << "R1" << R1;
        fundam << "R2" << R2;
        fundam << "t" << t;
        fundam << "K-1" << K_1;
        fundam << "points1H" << points1H;
        fundam << "points1" << points1;
        fundam << "points2H" << points2H;
        fundam << "points2" << points2;
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
        for (size_t k = 0; k < Proj.size(); k++)
        {
            cout << "P" << (k +1) << " [3x4] = " << endl;
            for (int i = 0; i < Proj[k].rows; i++)
            {
                for (int j = 0; j < Proj[k].cols; j++)
                    cout << "\t" << Proj[k].at<double>(i, j) << "\t\t";
                cout << endl;
            }
        }
        
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
        pointsMass[0].at<double>(0, i) = static_cast<double>(points1[static_cast<unsigned long>(i)].x);
        pointsMass[0].at<double>(1, i) = static_cast<double>(points1[static_cast<unsigned long>(i)].y);
        pointsMass[1].at<double>(0, i) = static_cast<double>(points2[static_cast<unsigned long>(i)].x);
        pointsMass[1].at<double>(1, i) = static_cast<double>(points2[static_cast<unsigned long>(i)].y);
        /*cout    << "pointsMass [0](0, " << i << ") = " << pointsMass[0].at<float>(0, i) << endl
                << "pointsMass [0](1, " << i << ") = " << pointsMass[0].at<float>(1, i) << endl
                << "pointsMass [1](0, " << i << ") = " << pointsMass[1].at<float>(0, i) << endl
                << "pointsMass [1](1, " << i << ") = " << pointsMass[1].at<float>(1, i) << endl
                << endl;*/
    }
    
    cv::sfm::triangulatePoints(pointsMass, Proj, points3D);
    
    /*for (int i = 0; i < points3D.cols; i++){
        cout << "3Dpoint[ " << i << " ] =";
        for (int j = 0; j < points3D.rows; j++){
            if ((points3D.at<double>(j, i) > -0.00001) && (points3D.at<double>(j, i) < 0.00001)) points3D.at<double>(j, i) = 0;
            cout << " " << points3D.at<double>(i, j) << " ";
        }
        cout << endl;
    }*/
    cv::FileStorage poins3D;      // Вывод в файл 3д точек
    poins3D.open("poins3D_XYZ.txt", cv::FileStorage::WRITE);
    poins3D << "pointsMass" << pointsMass;
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
