
#include "Header/sfm_train.h"

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

void SFM_Reconstruction::Reconstruction3D(Mat *data_frame1, Mat *data_frame2, Matx33d K)
{
    points3D *= 0;
    points3D_RGB->clear();
    R *= 0;
    t *= 0;
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
            recoverPose(E, points1, points2, K, R, t, 600, noArray(), points3D);
            
            for (int i = 0; i < points3D.cols; i++)
            {
                //cout << "3Dpoint[ " << i << " ] =";
                for (int j = 0; j < points3D.rows; j++)
                {
                    points3D.at<double>(j, i) /= points3D.at<double>(3, i);
                    //cout << " " << points3D.at<double>(j, i) << " ";
                }
                Scalar RGB1 = data_frame1->at< Vec3b >( points1.at( static_cast< size_t >(i) ) );
                Scalar RGB2 = data_frame2->at< Vec3b >( points2.at( static_cast< size_t >(i) ) );
                points3D_RGB[0].push_back ( static_cast< uchar >( (RGB1[2] + RGB2[2]) / 2 ) );
                points3D_RGB[1].push_back ( static_cast< uchar >( (RGB1[1] + RGB2[1]) / 2 ) );
                points3D_RGB[2].push_back ( static_cast< uchar >( (RGB1[0] + RGB2[0]) / 2 ) );
                //cout << endl;
            }
            
            FileStorage SFM_Result;
            SFM_Result.open("SFM_Result.txt", FileStorage::WRITE);
            SFM_Result << "E" << E;
            //SFM_Result << "points1" << points1;
            //SFM_Result << "points2" << points2;
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
            cout << " --- Not enough keypoints" << endl;
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
        cout <<" --- No keypoints found in one of the frames" << endl;
    }
}

void SFM_Reconstruction::Reconstruction3DopticFlow(Mat *data_frame1, Mat *data_frame2, Matx33d K)
{
    points3D *= 0;
    points3D_RGB->clear();
    R *= 0;
    t *= 0;
    frame4 = Mat::zeros(Size(2 * width_frame, height_frame), CV_8UC3);          //  data_frame2->type()
    Mat frame = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);           // data_frame2->type()
    flow = Mat::zeros(Size(width_frame, height_frame), CV_32FC2);
    
    if ((!data_frame1->empty()) && (!data_frame2->empty()))
    {
//--- STEP 1 --- Calculate optical flow -------------------------------------//
        points1.clear();
        points2.clear();
        numKeypoints = 0;
        Mat fg1, fg2;
        cvtColor( *data_frame1, fg1, COLOR_BGR2GRAY );
        cvtColor( *data_frame2, fg2, COLOR_BGR2GRAY );
        //calcOpticalFlowFarneback( frameGREY, frameCacheGREY, flow, 0.9, 1, 12, 2, 8, 1.7, 0 );    // OPTFLOW_FARNEBACK_GAUSSIAN
        optflow::calcOpticalFlowSparseToDense( fg1, fg2, flow, 4, 128, 0.01f, true, 500.0f, 1.5f);
        
        for (int y = 0; y < frame.rows; y += 3) {
            for (int x = 0; x < frame.cols; x += 3) {
                // get the flow from y, x position * 3 for better visibility
                const Point2f flowatxy = flow.at<Point2f>(y, x) * 1;
                // draw line at flow direction
                line(frame, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 200, 0));
                // draw initial point
                circle(frame, Point(x, y), 1, Scalar(0, 0, 0), -1);
                points1.push_back(Point2f(x, y));
                points2.push_back(Point2f((x + flowatxy.x), (y + flowatxy.y)));
//                Scalar RGB1 = data_frame1->at< Vec3b >( y, x );
//                Scalar RGB2 = data_frame2->at< Vec3b >( cvRound(y + flowatxy.y), cvRound(x + flowatxy.x) );
//                points3D_RGB[0].push_back ( static_cast< uchar >( (RGB1[2] + RGB2[2]) / 2 ) );
//                points3D_RGB[1].push_back ( static_cast< uchar >( (RGB1[1] + RGB2[1]) / 2 ) );
//                points3D_RGB[2].push_back ( static_cast< uchar >( (RGB1[0] + RGB2[0]) / 2 ) );
                numKeypoints++;
            }
        }
        //imshow("RGB", points3D_BGR);
        //waitKey(10);
        
//--- STEP 2 --- Find essential matrix --------------------------------------//
        E = findEssentialMat(points1, points2, K, RANSAC, 0.999, 1.0, Essen_mask);
        
//--- STEP 3 --- Calculation of 3d points -----------------------------------//
        recoverPose(E, points1, points2, K, R, t, 600, noArray(), points3D);
        
        for (int i = 0; i < points3D.cols; i++)
        {
            //cout << "3Dpoint[ " << i << " ] =";
            for (int j = 0; j < points3D.rows; j++){
                points3D.at<double>(j, i) /= points3D.at<double>(3, i);
                //cout << " " << points3D.at<double>(j, i) << " ";
            }
            Scalar RGB1 = data_frame1->at< Vec3b >( points1.at( static_cast< size_t >(i) ) );
            Scalar RGB2 = data_frame2->at< Vec3b >( points2.at( static_cast< size_t >(i) ) );
            points3D_RGB[0].push_back ( static_cast< uchar >( (RGB1[2] + RGB2[2]) / 2 ) );
            points3D_RGB[1].push_back ( static_cast< uchar >( (RGB1[1] + RGB2[1]) / 2 ) );
            points3D_RGB[2].push_back ( static_cast< uchar >( (RGB1[0] + RGB2[0]) / 2 ) );
            //cout << endl;
        }
        
        FileStorage SFM_Result;
        SFM_Result.open("SFM_Result_opticflow.txt", FileStorage::WRITE);
        SFM_Result << "E" << E;
        //SFM_Result << "points1" << points1;
        //SFM_Result << "points2" << points2;
        SFM_Result << "K" << K;
        SFM_Result << "R" << R;
        SFM_Result << "t" << t;
        SFM_Result << "points3D" << points3D;
        SFM_Result.release();
        cout << " --- SFM_Result written into file: SFM_Result_opticflow.txt" << endl;
        
        Rect r1(0, 0, frame.cols, frame.rows);
        Rect r2(data_frame2->cols, 0, data_frame2->cols, data_frame2->rows);
        frame.copyTo(frame4( r1 ));
        data_frame2->copyTo(frame4( r2 ));
        imshow("1-2 frame", frame4);
    }
    else
    {   
        Rect r1(0, 0, frame.cols, frame.rows);
        Rect r2(data_frame2->cols, 0, data_frame2->cols, data_frame2->rows);
        frame.copyTo(frame4( r1 ));
        data_frame2->copyTo(frame4( r2 ));
        imshow("1-2 frame", frame4);
        cout << " --- Not enough frames" << endl;
    }
}


void SFM_Reconstruction::opticalFlow(Mat *f, Mat *fc, int win, int vecS)
{
    img2Original = *f;
    flow = Mat(img2Original.cols, img2Original.rows, CV_32FC2);
    cvtColor(*f, frameGREY, COLOR_BGR2GRAY);
    cvtColor(*fc, frameCacheGREY, COLOR_BGR2GRAY);
    //calcOpticalFlowFarneback(frameGREY, frameCacheGREY, flow, 0.9, 1, 12, 2, 8, 1.7, 0);//OPTFLOW_FARNEBACK_GAUSSIAN
    optflow::calcOpticalFlowSparseToDense(frameGREY, frameCacheGREY, flow, 8, 128, 0.05f, true, 500.0f, 1.5f);
    
    
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

void SFM_Reconstruction::destroyWinSFM()
{
    namedWindow("1-2 frame", WINDOW_AUTOSIZE);
    destroyWindow("1-2 frame");
    namedWindow("OpticalFlow", WINDOW_AUTOSIZE);
    destroyWindow("OpticalFlow");
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
    
    cout << " --- Number of similar keypoints found : " << temp << endl;
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
