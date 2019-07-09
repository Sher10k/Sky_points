/*  
 *  Горячии клавиши
 * 
 * "ESC" - Interrupt the cycle
 * "Enter" - Take picture
 * "с" & "C" - Calibrate camera
 * "r" or "R" - Read from file, calibrate mode
 * "0" - use default parameters, calibrate mode
 * "f" & "F" - Output fundamental_matrix into file
 * "m" or "M" - Change mode camera
 * "Space" - Сделать снимок для покадрового режима
 * 
 */
#define CERES_FOUND true
#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <iostream>
#include <string>
//#include <stdio.h>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
//#include <opencv2/core/types_c.h>
//#include <opencv2/core/types.hpp>
//#include <opencv2/core/utility.hpp>
//#include <opencv2/core/ocl.hpp>
//#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui.hpp>
//#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>  // For FundamentalMat

#include <opencv2/sfm.hpp>
#include <opencv2/sfm/simple_pipeline.hpp>
#include <opencv2/sfm/reconstruct.hpp>
//#include <opencv2/sfm/robust.hpp>
#include <opencv2/sfm/triangulation.hpp>

/*#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common_headers.h>

#include <boost/thread/thread.hpp>*/

using namespace std;
using namespace cv;
using namespace cv::sfm;
using namespace cv::xfeatures2d;
//using namespace pcl;

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480

void drawlines(Mat& in_img, std::vector<cv::Point3f>& line, vector<Point2f>& pts) {         // Draw epipolar lines
    
    /*for (int i = 0; i < line.size(); i++) {
        printf("line [ %i ] = X %5.7f, Y %5.7f, Z %5.7f\n", i, line[i].x, line[i].y, line[i].z);
        printf("pts1 [ %i ] = X %5.7f, Y %5.7f\n", i, pts1[i].x, pts1[i].y);
    }*/
    RNG rng(12345);
    for (unsigned int i = 0; i < line.size(); i++) {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        int x0 = 0;
        int y0 = static_cast<int>(-line[i].z / line[i].y);
        int x1 = in_img.cols;
        int y1 = static_cast<int>(-(line[i].z + line[i].x * x1) / line[i].y);
        cv::line(in_img, cv::Point(x0, y0), cv::Point(x1, y1), color, 1, LINE_4, 0);
        cv::circle(in_img, pts[i], 5, color, -1);
    }
}

// Find matches between points 
unsigned long Match_find(vector<KeyPoint> kpf1, 
                         vector<KeyPoint> kpf2, 
                         Mat dpf1, 
                         Mat dpf2, 
                         vector<KeyPoint> *gp1, 
                         vector<KeyPoint> *gp2, 
                         vector<DMatch> *gm, 
                         unsigned long kn, 
                         int  threshold) {
   
    Ptr<BFMatcher> bf = BFMatcher::create(NORM_HAMMING, true);
    BFMatcher matcher(NORM_HAMMING, false);    // NORM_L2, NORM_HAMMING
    vector<DMatch> matches;
    DMatch dist1, dist2;
    
    matcher.match(dpf1, dpf2, matches, noArray());   // Matches key points of the frame with key points of the frame2
    
    // Sort match
    unsigned long temp = 0;
    kn = static_cast<unsigned long>(matches.size());
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
    // Selection of key points on both frames satisfying the threshold
    temp = 0;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance <  threshold) {
            int new_i = static_cast<int>( gp1->size() );
            gp1->push_back( kpf1[ static_cast<unsigned long>( matches[i].queryIdx ) ] );
            gp2->push_back( kpf2[ static_cast<unsigned long>( matches[i].trainIdx ) ] );
            gm->push_back( DMatch(new_i, new_i, 0) );
            temp ++;
        }
    }
    return temp;
}

// SURF Find matches between points
unsigned long Match_find_SURF(vector<KeyPoint> kpf1,
                              vector<KeyPoint> kpf2,
                              Mat dpf1,
                              Mat dpf2,
                              vector<KeyPoint> *gp1,
                              vector<KeyPoint> *gp2,
                              vector<DMatch> *gm,
                              unsigned long kn)
{
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    vector<DMatch> matches;
    DMatch dist1, dist2;

    //matcher.match(dpf1, dpf2, matches, noArray());   // Matches key points of the frame with key points of the frame2
    matcher->match(dpf1, dpf2, matches, noArray());

    // Sort match
    unsigned long temp = 0;
    kn = static_cast<unsigned long>(matches.size());
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
        if (matches[i].distance <  3*min_dist) { // threshold
            int new_i = static_cast<int>( gp1->size() );
            gp1->push_back( kpf1[ static_cast<unsigned long>( matches[i].queryIdx ) ] );
            gp2->push_back( kpf2[ static_cast<unsigned long>( matches[i].trainIdx ) ] );
            gm->push_back( DMatch(new_i, new_i, 0) );
            temp++;
        }
    }
    cout << "-- Temp : " << temp << endl << endl;
    return temp;
}

// SIFT Find matches between points
unsigned long Match_find_SIFT(vector<KeyPoint> kpf1,
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

int main()
{
    Mat frame, frame2, frame3, frameImg, frameImg2, frame_grey, flow, img2Original;
    double frame_pause = 0;
//    double frame_MSEC, frame_MSEC2; 
//    int thresh = 200;
//    int max_thresh = 255;

        //  Initialize VIDEOCAPTURE
    VideoCapture cap;
    int deviceID = 1;           //  camera 1
    int apiID = cv::CAP_ANY;    //  0 = autodetect default API
    cap.open(deviceID + apiID); //  Open camera
    if(!cap.isOpened()) {   // Check if we succeeded
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    else {  //  Info about frame
        cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);     // 320, 640, (640, 1280)
        cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);   // 240, 480, (360, 720)
        //cap.set(CAP_PROP_POS_FRAMES, 0);              // Set zero-frame
        //cap.set(CAP_PROP_FPS, 30);                    // Set FPS
        cap.set(CAP_PROP_AUTOFOCUS, 0);                 // Set autofocus
        cap.read(frame);

        cout    << "Width = " << cap.get(CAP_PROP_FRAME_WIDTH) << endl
                << "Height = " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl
                << "FPS = " << cap.get(CAP_PROP_FPS) << endl
                //<< "nframes = " << cap.get(CAP_PROP_FRAME_COUNT) << endl
                << "Auto focus" << cap.get(CAP_PROP_AUTOFOCUS) << endl
                << "cap : " << cap.get(CAP_PROP_FPS) << endl
                << "----------" <<endl;

            // Calculation FPS
        /*double fps;
        int num_frames = 120;
        frame_MSEC = cap.get(CAP_PROP_POS_MSEC);
        for(int i = 0; i < num_frames; i++) {
            cap.read(frame);
        }
        frame_MSEC2 = cap.get(CAP_PROP_POS_MSEC);
        double seconds = frame_MSEC2 - frame_MSEC;
        cout << "Time taken : " << seconds * 1000 << " seconds" << endl;
        fps  = num_frames / seconds * 1000;
        cout << "Estimated frames per second : " << fps << endl;*/
    }
    
    //-------------------------------VARIABLES------------------------------------------//
    frame_pause = frame_pause / 30 * 1000;  // Convertion from frames per second to msec

        // ORB keypoint
    unsigned long key_num = 10;  // Num keypoint
    Ptr<FeatureDetector> detector = ORB::create(static_cast<int>(key_num), 1.2f, 8, 31, 0, 4, ORB::HARRIS_SCORE, 31);   // HARRIS_SCORE, FAST_SCORE
    //Ptr<SURF> detectorSURF = cv::xfeatures2d::SURF::create(static_cast<double>(key_num), 4, 3, true, false);   // cv::xfeatures2d::
    Ptr<SURF> detectorSURF = cv::xfeatures2d::SURF::create(100);
    Ptr<SIFT> detectorSIFT = cv::xfeatures2d::SIFT::create(0, 4, 0.04, 10, 1.6);
    std::vector<KeyPoint> keypoints_frame, keypoints_frame2, keypoints_frame_SURF, keypoints_frame2_SURF, keypoints_frame_SIFT, keypoints_frame2_SIFT;
    Mat descriptors_frame, descriptors_frame2, descriptors_frame_SURF, descriptors_frame2_SURF, descriptors_frame_SIFT, descriptors_frame2_SIFT;

        // Для калибровки
    Matx33d intrinsic = Matx33d( 10,     0,  FRAME_WIDTH/2,
                                 0,     10,  FRAME_HEIGHT/2,
                                 0,     0,  1);
    Matx<double, 1, 5> distCoeffs = Matx<double, 1, 5>(0.0, 0.0, 0.0, 0.0, 0.0);  // (k1, k2, p1, p2, k3)
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    
        // Fundamental matrix
    Mat fundamental_matrix;

        // Epipolar linu
    std::vector<cv::Point3f> lines[2];
    Mat frame_epipol1, frame_epipol2;
    
        //  Array of array for frames key points  
    Mat Pt1 = cv::Mat::eye(3, 4, CV_64F);   // Projection matrices for each camera
    Mat Pt2 = cv::Mat::eye(3, 4, CV_64F);  
    vector <Mat> Ps(2); // Matx34d          // Vector of projection matrices for each camera
    Ps[0] = cv::Mat(3, 4, CV_64F);
    Ps[1] = cv::Mat(3, 4, CV_64F);
    
        // SFM camera
    cv::sfm::libmv_CameraIntrinsicsOptions camera {SFM_DISTORTION_MODEL_DIVISION, 
                                                    intrinsic(0, 0), 
                                                    intrinsic(1, 1), 
                                                    intrinsic(0, 2), 
                                                    intrinsic(1, 2),
                                                    distCoeffs(0, 0),
                                                    distCoeffs(0, 1),
                                                    distCoeffs(0, 4),
                                                    distCoeffs(0, 2),
                                                    distCoeffs(0, 3)};
    
        // Cloud of points
    /*pcl::PointCloud <pcl::PointXYZ> cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(50.0);*/
    
        // Other variables
    unsigned long t = 0;    // Счетчик найденых точек
    int f = 2;              // Переключение в режим калибровки
    bool mode_cam = false;   // Режим работы камеры
    Mat res;
    Mat frame4 = Mat::zeros(Size(2 * frame.cols, frame.rows), CV_8UC3);
    if (f != 2) namedWindow("frame_epipol_double", WINDOW_AUTOSIZE);                    // Window for output result
    int c;
    
        // START
    while(1) {         // Основной режим работы камеры, потоковый режим, !viewer->wasStopped ()
        if ( (f == 1) && (mode_cam == true) ) {   
            //  Wait for a new frame from camera and store it into 'frameImg'
            if (!cap.read(frameImg)) { // check if we succeeded
                cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            undistort(frameImg, frame, intrinsic, distCoeffs);

            // Delay frame2         ------------------------------------------------------------//
            /*frame_MSEC = cap.get(CAP_PROP_POS_MSEC);
            frame_MSEC2 = cap.get(CAP_PROP_POS_MSEC);
            while((frame_MSEC2 - frame_MSEC) < frame_pause)
            {
                cap.read(frame2);
                frame_MSEC2 = cap.get(CAP_PROP_POS_MSEC);
                //cout  <<    "MSEC = " << cap.get(CAP_PROP_POS_MSEC) << endl;
            }
            //cout << "//---------------------------  = " << frame_pause <<endl;*/
            // End Delay frame2         --------------------------------------------------------//

            // Находим оптимальное преобразование, 
            // согласующееся с большинством пар точек. 
            /*vector<Point2f> pt1, pt2; 
            for( size_t i = 0; i < matches.size(); i++ ) { 
             pt1.push_back(keypoints1[matches[i].queryIdx].pt); 
             pt2.push_back(keypoints2[matches[i].trainIdx].pt); 
            } 
            // H – это матрица оптимального перспективного 
            // преобразования от img1 к img2 
            Mat H = findHomography(pt1, pt2, RANSAC, 10); */
            
            // ORB detector        -------------------------------------------------------------//      step 1
            if (!cap.read(frameImg2)) { // check if we succeeded and store new frame into 'frameImg2'
                cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            undistort(frameImg2, frame2, intrinsic, distCoeffs);
            
            detector->detectAndCompute(frame, noArray(), keypoints_frame, descriptors_frame);       // Detected key points at frame
            detector->detectAndCompute(frame2, noArray(), keypoints_frame2, descriptors_frame2);    // Detected key points at frame2

            if ((keypoints_frame.size() != 0) && (keypoints_frame2.size() != 0)) {          // Проверка на нахождение хотябы одной контрольной точки
                
                vector<DMatch> good_matches;
                vector<KeyPoint> good_points1, good_points2;
                t = Match_find( keypoints_frame,
                                keypoints_frame2,
                                descriptors_frame,
                                descriptors_frame2,
                                &good_points1,
                                &good_points2,
                                &good_matches,
                                key_num,
                                20 );                                                  // Поиск схожих точек удовлетворяющих порогу
                
                // drawing two frames and connecting similar cue points with lines
                drawMatches(frame, good_points1, frame2, good_points2, good_matches, res);
            // END ORB detector        ------------------------------------------------------------//   END step 1
                
                if ((good_points1.size() >= 8) && (good_points2.size() >= 8)) { // Проверка на наличие точек, удовлетворяющих порогу
                    // FindFundamentalMat     -------------------------------------------------------------//      step 2
                    vector<Point2f> points1(t);
                    vector<Point2f> points2(t);
                    vector<Point2f> status(t);
                    
                    for (unsigned long i = 0; i < t; i++) {
                        points1[i].x = good_points1[i].pt.x;
                        points1[i].y = good_points1[i].pt.y;
                        points2[i].x = good_points2[i].pt.x;
                        points2[i].y = good_points2[i].pt.y;
                    }
                    fundamental_matrix = cv::findFundamentalMat(points1, points2, FM_RANSAC, 1.0, 0.99, noArray());
                    /*for(int i = 0; i < fundamental_matrix.rows; i++){     // Отображение элементов 
                        for(int j = 0; j < fundamental_matrix.cols; j++){
                            //printf("fundamental_matrix[%.0f ", ptr[j]);
                            printf("fundamental_matrix [ %i ][ %i ] = %5.7f\n", i, j, fundamental_matrix.at<double>(j,i));
                        }
                        //printf("\n");
                    }
                    //printf("-----\n");*/
                    // END FindFundamentalMat     ---------------------------------------------------------//   END step 2
                    
                    if (!fundamental_matrix.empty()) {  // Draw epilines between two image     -------------//      step 3
                        
                        cv::computeCorrespondEpilines(points1, 1 , fundamental_matrix, lines[0]);
                        cv::computeCorrespondEpilines(points2, 2 , fundamental_matrix, lines[1]);
                                                
                        drawlines(frame, lines[0], points1);
                        drawlines(frame2, lines[1], points2);
                                           
                        Rect r1(0, 0, frame.cols, frame.rows);                // Создаем фрагменты для склеивания зображения
                        Rect r2(frame2.cols, 0, frame2.cols, frame2.rows);
                        frame.copyTo(frame4( r1 ));
                        frame2.copyTo(frame4( r2 ));                    
                        imshow("frame_epipol_double", frame4);
                        
                    }   // END Draw epilines between two image  ---------------------------------------------//   END step 3
                } else {
                    frame4 = res;
                    imshow("frame_epipol_double", frame4);
                }
            } else {                                                                 // if points not found
                Rect r1(0, 0, frame.cols, frame.rows);                
                Rect r2(frame2.cols, 0, frame2.cols, frame2.rows);
                frame.copyTo(frame4( r1 ));
                frame2.copyTo(frame4( r2 )); 
                imshow("frame_epipol_double", frame4);
            }

            // Farneback optical flow -------------------------------------------------------------//      step #
            /*cap.read(frame2);
            frame2.copyTo(img2Original);
            cvtColor(frame, frame, COLOR_BGR2GRAY);
            cvtColor(frame2, frame2, COLOR_BGR2GRAY);
            calcOpticalFlowFarneback(frame, frame2, flow, 0.4, 1, 12, 2, 8, 1.2, 0);
            //calcOpticalFlowSparseToDense();
            for (int y = 0; y < frame2.rows; y += 3) {
                for (int x = 0; x < frame2.cols; x += 3) {
                    // get the flow from y, x position * 3 for better visibility
                    const Point2f flowatxy = flow.at<Point2f>(y, x) * 1;
                    // draw line at flow direction
                    line(img2Original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 0, 0));
                    // draw initial point
                    circle(img2Original, Point(x, y), 1, Scalar(0, 0, 0), -1);
                }
            }
            //imshow("frame", frame);
            //imshow("frame2", frame2);
            imshow("img2Original", img2Original);*/
            // End Farneback optical flow ---------------------------------------------------------//


            //  Harris corner detector      -------------------------------------------------------//
            /*int blockSize = 2;
            int apertureSize = 3;
            double k = 0.04;
            //cvtColor(frame, frame_grey, COLOR_BGR2GRAY);
            Canny(frame, frame_grey, 100, 100, 3, false);
            Mat dst = Mat::zeros( frame.size(), CV_32FC1 );
            cornerHarris( frame_grey, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
            Mat dst_norm, dst_norm_scaled;
            normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
            convertScaleAbs( dst_norm, dst_norm_scaled );
            for( int i = 0; i < dst_norm.rows ; i++ )
            {
                for( int j = 0; j < dst_norm.cols; j++ )
                {
                    if( (int) dst_norm.at<float>(i,j) > thresh )
                    {
                        //circle( dst_norm_scaled, Point(j,i), 1,  Scalar(255), -1, LINE_4, 0 );
                        circle( frame, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
                    }
                }
            }
            imshow("frame", frame);
            imshow("frame_grey", frame_grey);
            //imshow("dst", dst);
            //imshow("dst_norm", dst_norm);
            //imshow("dst_norm_scaled", dst_norm_scaled);*/
            //  End Harris corner detector      ----------------------------------------------------//


            //double K = 0.4;
            //resize(frame, frame, Size(), K, K, CV_INTER_LINEAR);    //  Resize   -----------------//

            //imshow("Frame_live", frame);
            //cout  <<    "MSEC = " << cap.get(CAP_PROP_POS_MSEC) << endl;
        }                                   // Покадровый режим работы камеры
        else if ((f == 1) && (mode_cam == false)) { 
                //  Wait for a new frame from camera and store it into 'frameImg'
            if (!cap.read(frameImg)) { // check if we succeeded
                cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            undistort(frameImg, frame3, intrinsic, distCoeffs);
            imshow("real_time", frame3);
            
                // Вывод предыдущего и текущего кадров вместе
            /*Rect r1(0, 0, frame.cols, frame.rows);
            Rect r2(frame2.cols, 0, frame2.cols, frame2.rows);
            frame.copyTo(frame4( r1 ));
            frame2.copyTo(frame4( r2 ));
            imshow("1-2 frame",frame4);*/
            
            int button_nf = waitKey(1);
            if ( button_nf == 32 )             // If press "space"
            {
                frame3.copyTo( frame );
                //detector->detectAndCompute(frame, noArray(), keypoints_frame, descriptors_frame);             // Detected key points at frame
                detectorSURF->detectAndCompute(frame, noArray(), keypoints_frame_SURF, descriptors_frame_SURF); // SURF detected frame
                detectorSIFT->detectAndCompute(frame, noArray(), keypoints_frame_SIFT, descriptors_frame_SIFT);  // SIFT detected frame
                if ( !frame2.empty() ) 
                {
                    //detector->detectAndCompute(frame2, noArray(), keypoints_frame2, descriptors_frame2);              // Detected key points at frame2
                    detectorSURF->detectAndCompute(frame2, noArray(), keypoints_frame2_SURF, descriptors_frame2_SURF);  // SURF detected frame2
                    detectorSIFT->detectAndCompute(frame2, noArray(), keypoints_frame2_SIFT, descriptors_frame2_SIFT);      // SIFT detected frame2

                    if ((keypoints_frame_SURF.size() != 0) && (keypoints_frame2_SURF.size() != 0)) {  // Matching key points
                        vector <DMatch> good_matches;
                        vector <KeyPoint> good_points1, good_points2;
                        /*t = Match_find_SURF(    keypoints_frame_SURF,
                                                keypoints_frame2_SURF,
                                                descriptors_frame_SURF,
                                                descriptors_frame2_SURF,
                                                &good_points1,
                                                &good_points2,
                                                &good_matches,
                                                key_num);               // t = number of good points */
                        t = Match_find_SIFT(    keypoints_frame_SIFT,
                                                keypoints_frame2_SIFT,
                                                descriptors_frame_SIFT,
                                                descriptors_frame2_SIFT,
                                                &good_points1,
                                                &good_points2,
                                                &good_matches);

                        /*RNG rng(12345);     // Drawing found key points
//                        for (unsigned long i = 0; i < good_points1.size(); i++) {
//                            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//                            circle(frame, good_points1[i].pt, 2, color, 2, LINE_8, 0);
//                            circle(frame2, good_points2[i].pt, 2, color, 2, LINE_8, 0);
//                        }
                        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                        drawKeypoints(frame, good_points1, frame_epipol1, color, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                        drawKeypoints(frame2, good_points2, frame_epipol2, color, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                        Rect r1(0, 0, frame_epipol1.cols, frame_epipol1.rows);
                        Rect r2(frame_epipol2.cols, 0, frame_epipol2.cols, frame_epipol2.rows);
                        frame_epipol1.copyTo(frame4( r1 ));
                        frame_epipol2.copyTo(frame4( r2 ));
                        imshow("1-2 frame", frame4);*/
                        /*drawMatches(frame, good_points1, frame2, good_points2, good_matches, frame4);
                        imshow("1-2 frame", frame4);*/
                        
                        if ((good_points1.size() >= 8) && (good_points2.size() >= 8)) { // Проверка на наличие точек, удовлетворяющих порогу
                            
                            vector <Point2f> points1(t), points2(t), status(t);     // Point2f
                            cv::KeyPoint::convert(good_points1, points1);           // Convert from KeyPoint to Point2f
                            cv::KeyPoint::convert(good_points2, points2);
                            cv::Mat pnts3D(4, static_cast<int>(t), CV_64F);         // 3D points.  static_cast<int>(good_points1.size())
                            
                            vector <cv::Mat> points2frame(2);                       // Vector of key point arrays for each frame
                            points2frame[0] = cv::Mat(2, static_cast<int>(points1.size()), CV_64F);
                            points2frame[1] = cv::Mat(2, static_cast<int>(points2.size()), CV_64F);
                            
                            for (int i = 0; i < static_cast<int>(t); i++) {         // Unioning the key points in new variable
                                points2frame[0].at<float>(0, i) = points1[static_cast<unsigned long>(i)].x;
                                points2frame[0].at<float>(1, i) = points1[static_cast<unsigned long>(i)].y;
                                points2frame[1].at<float>(0, i) = points2[static_cast<unsigned long>(i)].x;
                                points2frame[1].at<float>(1, i) = points2[static_cast<unsigned long>(i)].y;
                                /*cout    << "points2frame [0](0, " << i << ") = " << points2frame[0].at<float>(0, i) << endl
                                        << "points2frame [0](1, " << i << ") = " << points2frame[0].at<float>(1, i) << endl
                                        << "points2frame [1](0, " << i << ") = " << points2frame[1].at<float>(0, i) << endl
                                        << "points2frame [1](1, " << i << ") = " << points2frame[1].at<float>(1, i) << endl
                                        << endl;*/
                            }
                            
                            // FindFundamentalMat     -------------------------------------------------------------//      step 4
                            sfm::reconstruct(points2frame, Ps, pnts3D, intrinsic, true);
                            /*cout    << "pnts3D.cols = " << pnts3D.cols
                                    << endl;
                            for (int i = 0; i < pnts3D.cols; i++){
                                for (int j = 0; j < pnts3D.rows; j++){
                                    cout << " " << pnts3D.at<double>(i, j) << " ";
                                }
                                cout << endl;
                            }
                                // 3D points cloud
                            //pcl::PointCloud <pcl::PointXYZ> cloud;
                            cloud.height = 1;
                            cloud.width = static_cast<unsigned int>( pnts3D.cols );
                            cloud.is_dense = false;
                            cloud.points.resize( cloud.width * cloud.height );
                            
                            for (size_t i = 0; i < cloud.points.size (); ++i)
                            {
                                cloud.points[i].x = pnts3D.at<float>(0, static_cast<int>(i));
                                cloud.points[i].y = pnts3D.at<float>(1, static_cast<int>(i));
                                cloud.points[i].z = pnts3D.at<float>(2, static_cast<int>(i));
                                //cloud.points[i].r = rgb_cenal[2].at(i);
                                //cloud.points[i].g = rgb_cenal[1].at(i);
                                //cloud.points[i].b = rgb_cenal[0].at(i);
                            }
                                // Save 3D points in file
                            pcl::io::savePCDFileASCII ("Reconstruct_cloud.pcd", cloud);
                                // Load 3D points (cloud points)
                            //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
                            pcl::io::loadPCDFile("Reconstruct_cloud.pcd", *cloud2);*/
                                // View cloud points
                            /*pcl::visualization::CloudViewer viewer("Cloud Viewer");
                            viewer.showCloud(cloud2, "cloud");*/
                            
                            
                            
                            // END FindFundamentalMat     ---------------------------------------------------------//    END step 4
                            
                            /*fundamental_matrix = cv::findFundamentalMat(points1, points2, FM_RANSAC, 1.0, 0.99, noArray());
                            if (!fundamental_matrix.empty()) {  // Проверка на пустотности фундаментальной матрицы
                                
//                                projectionsFromFundamental(fundamental_matrix, Pt1, Pt2);
//                                Ps[0] = Pt1;
//                                Ps[1] = Pt2;
                                
                                //cv::sfm::triangulatePoints(points2frame, Ps, pnts3D);                 // Triangulate and find 3D points using inliers, with SFM
                                //cv::triangulatePoints(Pt1, Pt2, point_good1, point_good2, pnts3D);    // Triangulation without SFM
//                                cout    << "pnts3D.cols = " << pnts3D.cols
//                                        << endl;
//                                for (int i = 0; i < pnts3D.cols; i++){
//                                    for (int j = 0; j < pnts3D.rows; j++){
//                                        cout << " " << pnts3D.at<double>(i, j) << " ";
//                                    }
//                                    cout << endl;
//                                }
//                                FileStorage poins3D;      // Вывод в файл 3д точек
//                                poins3D.open("/home/roman/Sky_points/poins3D_XYZ.txt", FileStorage::WRITE);
//                                poins3D << "pnts3D" << pnts3D;
//                                poins3D.release();
                                
                                cv::computeCorrespondEpilines(points1, 1 , fundamental_matrix, lines[0]);   // Расчет эпиполярных линый для 1го кадра
                                cv::computeCorrespondEpilines(points2, 2 , fundamental_matrix, lines[1]);   // Расчет эпиполярных линый для 2го кадра
                                
                                frame.copyTo( frame_epipol1 );
                                frame2.copyTo( frame_epipol2 );
                                //drawlines(frame_epipol1, lines[0], points1);  // Отрисовка эпиполярных линий
                                //drawlines(frame_epipol2, lines[1], points2);
                                
//                                Rect r1(0, 0, frame_epipol1.cols, frame_epipol1.rows);                // Создаем фрагменты для склеивания зображения
//                                Rect r2(frame_epipol2.cols, 0, frame_epipol2.cols, frame_epipol2.rows);
//                                frame_epipol1.copyTo(frame4( r1 ));
//                                frame_epipol2.copyTo(frame4( r2 ));
//                                imshow("1-2 frame", frame4);

                                drawMatches(frame_epipol1, good_points1, frame_epipol2, good_points2, good_matches, frame4);
                                imshow("1-2 frame", frame4);
                            }*/
                            
                            drawMatches(frame, good_points1, frame2, good_points2, good_matches, frame4);
                            imshow("1-2 frame", frame4);
                        }
                    } else {
                        // Вывод первого кадра и пустого кадра
                        frame2 *= 0;
                        Rect r1(0, 0, frame.cols, frame.rows);
                        Rect r2(frame2.cols, 0, frame2.cols, frame2.rows);
                        frame.copyTo(frame4( r1 ));
                        frame2.copyTo(frame4( r2 ));
                        imshow("1-2 frame", frame4);
                    }
                }
                frame.copyTo( frame2 );
                frame *= 0;
                
            } else if ( button_nf == 27 ) {                             // Interrupt the cycle, press "ESC"
                break;
            } else if ( (button_nf == 77) || (button_nf == 109) ) {     // Change mode camera, press "m" or "M"
                mode_cam = !mode_cam;
                frame2 *= 0;                // Frame zeroing
                if (mode_cam == true) {     // Закрываем окна при смене режима
                    destroyWindow("real_time");
                    namedWindow("1-2 frame", WINDOW_AUTOSIZE);
                    destroyWindow("1-2 frame");
                }
            }
            /*if (cloud.size() != 0) {        // View cloud points
                    // Clear the view
                viewer->removeAllShapes();
                viewer->removeAllPointClouds();
                viewer->addPointCloud<pcl::PointXYZ>(cloud2, "sample cloud", 0);
                viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
                viewer->spinOnce (5);
            }*/
        }                                   // Калибровка камеры  press "с" or "C"----------------------//      step 0
        else if ( f == 2 ) {
            int successes = 0;                                      // Счетчик удачных калибровочных кадров
            int num_successes = 10;                                    // Кол-во калибровочных кадров
            Mat calib_frame = Mat::zeros(frame.size(), CV_8UC3);      // Калибровочный кадр
            Mat calib_frame_grey = Mat::zeros(frame.size(), CV_8UC3);

            int numCornersHor = 8;  // 7                                 // Кол-во углов по вертикале и горизонтале,
            int numCornersVer = 6;  // 5                                // на 1 меньше чем кол-во квадратов по вертикале и горизонтале
            int numSquares = numCornersHor * numCornersVer;
            Size board_sz = Size(numCornersHor, numCornersVer);

            vector<vector<Point2f>> image_points;                   // Точки на изображении
            vector<Point2f> calib_frame_corners;                    // Найденые вершины на шахматной доске
            vector<vector<Point3f>> object_points;                  //
            vector<Point3f> obj;
            for(int j = 0; j < numSquares; j++)
                obj.push_back(Point3d(j/numCornersHor, j%numCornersHor, 0.0)); //static_cast<double>(0.0f)

            int batton_calib = 0;
            while ( successes < num_successes ) {                        // Цикл для определенного числа калибровочных кадров
                batton_calib = waitKey(1);
                if( batton_calib == 27 ) {                            // Interrupt the cycle calibration, press "ESC"
                    break;
                } else if ( batton_calib == 13 ) {                    // Take picture Chessboard, press "Enter"
                    if (!cap.read(calib_frame)) {               // check if we succeeded and store frame into calib_frame
                        cerr << "ERROR! blank frame grabbed\n";
                        break;
                    }
                    cvtColor(calib_frame, calib_frame_grey, COLOR_BGR2GRAY);   //CV_BGR2GRAY
                        // Поиск углов на шахматной доске
                    bool found = findChessboardCorners(calib_frame,
                                                       board_sz,
                                                       calib_frame_corners,
                                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE ); //CALIB_CB_NORMALIZE_IMAGE, CV_CALIB_CB_FILTER_QUADS

                    if (found) {    // Проверка удачно найденых углов
                        cornerSubPix(calib_frame_grey,
                                     calib_frame_corners,
                                     Size(11,11),
                                     Size(-1,-1),
                                     TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1) );   // Уточнение углов

                        drawChessboardCorners(calib_frame,
                                              board_sz,
                                              calib_frame_corners,
                                              found );                               //отрисовка углов

                        image_points.push_back(calib_frame_corners);
                        object_points.push_back(obj);
                        cout << "Snap stored!" << endl;

                        successes++;
                    }
                } else if ( (batton_calib == 114) || (batton_calib == 82) || (batton_calib == 48) || (batton_calib == 27) ) {       
                    // Read from file, press "r" or "R", or use default parameters, press "0", or interrupt the calibration cycle, press "ESC" 
                    break;
                }
                cap.read(frame3);
                Rect r1(0, 0, frame3.cols, frame3.rows);                // Создаем фрагменты для склеивания зображения
                Rect r2(frame3.cols, 0, frame3.cols, frame3.rows);
                frame3.copyTo(frame4( r1 ));
                calib_frame.copyTo(frame4( r2 ));
                imshow("calibration", frame4);      // Вывод последнего удачного калибровачного кадра и кадра потока

                batton_calib = 0;
            }

            if ( (batton_calib == 114) || (batton_calib == 82) ) {  // Read from file
                FileStorage fs; 
                fs.open("/home/roman/Sky_points/Calibrate_cam.txt", FileStorage::READ);     // Read from file data calibration
                fs["intrinsic"] >> intrinsic;
                fs["distCoeffs"] >> distCoeffs;
                fs["rvecs"] >> rvecs;
                fs["tvecs"] >> tvecs;
                fs.release();
                
            } else if (batton_calib == 48) {    // use default parameters   
                intrinsic(0, 0) = 600;  // fx
                intrinsic(1, 1) = 600;  // fy
                intrinsic(0, 2) = 320;  // Cx, half of width frame
                intrinsic(1, 2) = 240;  // Cy, half of hight frame
                intrinsic(2, 2) = 1;
                //for (int i = 0; i < 5; i++) distCoeffs(0, i) = 0;
                distCoeffs(0, 0) = 0.0; // k1
                distCoeffs(0, 1) = 0.0; // k2
                distCoeffs(0, 2) = 0.0; // p1
                distCoeffs(0, 3) = 0.0; // p2
                distCoeffs(0, 4) = 0.0; // k3
                
                FileStorage fs;
                fs.open("/home/roman/Sky_points/Calibrate_cam_Zero.txt", FileStorage::WRITE);    // Write in file data calibration
                fs << "intrinsic" << intrinsic;
                fs << "distCoeffs" << distCoeffs;
                fs.release();
                
            } else if (batton_calib == 27) {    // interrupt the calibration cycle
                break;
                
            } else {
                calibrateCamera(object_points,
                                image_points,
                                calib_frame.size(),
                                intrinsic,
                                distCoeffs,
                                rvecs,
                                tvecs,
                                CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3);                                 // Calibrate  | CALIB_FIX_K6
                FileStorage fs;
                fs.open("/home/roman/Sky_points/Calibrate_cam.txt", FileStorage::WRITE);    // Write in file data calibration
                fs << "intrinsic" << intrinsic;
                fs << "distCoeffs" << distCoeffs;
                fs << "rvecs" << rvecs;
                fs << "tvecs" << tvecs;
                fs.release();
            }            
            
            namedWindow("calibration", WINDOW_AUTOSIZE);
            destroyWindow("calibration");
            f = 1;
        }                                   // END Калибровка камеры  ----------------------------------//   END step 0

        // Обработка нажатой кнопки  --------------------------------------------------------//
        c = waitKey(1);
        if( c == 27 ) {                             // Interrupt the cycle, press "ESC"
            break;
        } else if ( c == 13 ) {                     // Take picture, press "Enter"
            //imshow("foto", res);
            imshow("foto", frame4);
        } else if ( (c == 99) || (c == 67) ) {      // Calibrate camera, press "с" & "C"
            f = 2;
            imshow("calibration", frame);
            namedWindow("frame_epipol_double", WINDOW_AUTOSIZE);
            destroyWindow("frame_epipol_double");
        } else if ( (c == 70) || (c == 102) ) {     // Output fundamental_matrix into file, press "f" & "F"
            FileStorage fundam;
            fundam.open("/home/roman/Sky_points/Fundamental_matrix.txt", FileStorage::WRITE);
            fundam << "fundamental_matrix" << fundamental_matrix;
            fundam.release();
        } else if ( (c == 77) || (c == 109) ) {     // Change mode camera, press "m" or "M"
            mode_cam = !mode_cam;
            frame2 *= 0;                // Frame zeroing
            if (mode_cam == true) {     // Закрываем окна при смене режима
                destroyWindow("real_time");
                destroyWindow("1-2 frame");
            } else {
                destroyWindow("frame_epipol_double");
            }
        } 
        // END Обработка нажатой кнопки  ----------------------------------------------------//
    }

    cap.release();
    return 0;
}
