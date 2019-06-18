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
#include <stdio.h>

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
//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>  // For FundamentalMat

#include <opencv2/sfm.hpp>
#include <opencv2/sfm/simple_pipeline.hpp>
#include <opencv2/sfm/reconstruct.hpp>
//#include <opencv2/sfm/robust.hpp>
#include <opencv2/sfm/triangulation.hpp>

using namespace std;
using namespace cv;
using namespace cv::sfm;

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480

Mat drawlines(Mat& img1, std::vector<cv::Point3f>& line, vector<Point2f>& pts) {         // Draw epipolar lines
    
    /*for (int i = 0; i < line.size(); i++) {
        printf("line [ %i ] = X %5.7f, Y %5.7f, Z %5.7f\n", i, line[i].x, line[i].y, line[i].z);
        printf("pts1 [ %i ] = X %5.7f, Y %5.7f\n", i, pts1[i].x, pts1[i].y);
    }*/
    RNG rng(12345);
    for (unsigned int i = 0; i < line.size(); i++) {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        int x0 = 0;
        int y0 = static_cast<int>(-line[i].z / line[i].y);
        int x1 = img1.cols;
        int y1 = static_cast<int>(-(line[i].z + line[i].x * x1) / line[i].y);
        cv::line(img1, cv::Point(x0, y0), cv::Point(x1, y1), color, 1, LINE_4, 0);
        cv::circle(img1, pts[i], 5, color, -1);
    }    
    return img1;
}

// Find matches between points 
unsigned long Match_find(vector<KeyPoint> kpf1, vector<KeyPoint> kpf2, Mat dpf1, Mat dpf2, vector<KeyPoint> *gp1, vector<KeyPoint> *gp2, vector<DMatch> *gm, unsigned long kn, int  threshold) {
   
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

int main()
{
    /*printf("OPENCV_OPENCL_DEVICE='%s'\n", getenv("OPENCV_OPENCL_DEVICE"));
    cv::ocl::setUseOpenCL(true);
    ocl::Context context = ocl::Context::getDefault();*/

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
        cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); //320, 640, (640, 1280)
        cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);   //240, 480, (360, 720)
        cap.set(CAP_PROP_POS_FRAMES, 0);
        cap.read(frame);

        cout  <<    "Width = " << cap.get(CAP_PROP_FRAME_WIDTH) << endl <<
                    "Height = " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl <<
                    "nframes = " << cap.get(CAP_PROP_FRAME_COUNT) << endl <<
                    "----------" <<endl;

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
    // Default cam          ------------------------------------------------------------//
    /*VideoCapture cap2;
    int deviceID2 = 0;           //  camera 1
    int apiID2 = cv::CAP_ANY;    //  0 = autodetect default API
    cap2.open(deviceID2 + apiID2); //  Open camera
    if(!cap2.isOpened()) {   // Check if we succeeded
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    else {  //  Info about frame
        cap2.set(CAP_PROP_FRAME_WIDTH, 640); //320, 640, (640, 1280)
        cap2.set(CAP_PROP_FRAME_HEIGHT, 480);   //240, 480, (360, 720)
        cap2.set(CAP_PROP_POS_FRAMES, 0);
        cap2.read(frame2);

        cout  <<    "Width = " << cap2.get(CAP_PROP_FRAME_WIDTH) << endl <<
                    "Height = " << cap2.get(CAP_PROP_FRAME_HEIGHT) << endl <<
                    "nframes = " << cap2.get(CAP_PROP_FRAME_COUNT) << endl;

            // Calculation FPS
        double fps;
        int num_frames = 120;
        frame_MSEC = cap.get(CAP_PROP_POS_MSEC);
        for(int i = 0; i < num_frames; i++) {
            cap.read(frame);
        }
        frame_MSEC2 = cap.get(CAP_PROP_POS_MSEC);
        double seconds = frame_MSEC2 - frame_MSEC;
        cout << "Time taken : " << seconds * 1000 << " seconds" << endl;
        fps  = num_frames / seconds * 1000;
        cout << "Estimated frames per second : " << fps << endl;
    }*/ //                    ----------------------------------------------------------//

    frame_pause = frame_pause / 30 * 1000;  // Convertion from frames per second to msec

    // ORB keypoint
    unsigned long key_num = 200;  // Num keypoint
    Ptr<FeatureDetector> detector = ORB::create(static_cast<int>(key_num), 1.2f, 8, 31, 0, 4, ORB::HARRIS_SCORE, 31);   // HARRIS_SCORE, FAST_SCORE
    //Ptr<SURF> detector = SURF::create(static_cast<double>(key_num), 4, 3, true, false);   // cv::xfeatures2d::
    std::vector<KeyPoint> keypoints_frame, keypoints_frame2;
    Mat descriptors_frame, descriptors_frame2;

    // Brute-Force Matcher1
    /*-----------------*/

    unsigned long t = 0;              // Счетчик найденых точек
    int f = 2;              // Переключение в режим калибровки
    bool mode_cam = true;   // Режим работы камеры
    Mat res;
    Mat frame4 = Mat::zeros(Size(2 * frame.cols, frame.rows), CV_8UC3);
    namedWindow("frame_epipol_double", WINDOW_AUTOSIZE);                    // Window for output result

    // Для калибровки
    Matx33d intrinsic = Matx33d( 10,     0,  FRAME_WIDTH/2,
                                 0,     10,  FRAME_HEIGHT/2,
                                 0,     0,  1);
    Matx<double, 1, 5> distCoeffs = Matx<double, 1, 5>(0.0, 0.0, 0.0, 0.0, 0.0);  // (k1, k2, p1, p2, k3)
    vector<Mat> rvecs;
    vector<Mat> tvecs;

    /*Mat distCoeffs = Mat(1, 5, CV_32FC1);  // Старая инициализация переменной
    Mat intrinsic = Mat(3, 3, CV_32FC1);
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            intrinsic.ptr<float>(i)[j] = 0;
            //printf("intrinsic(%i, %i) = %7.5f\n", i, j, intrinsic.ptr<double>(i)[j]);
        }
    }
    for (int i = 0; i < 5; i++) {
        distCoeffs.ptr<float>(0)[i] = 0;
        //printf("distCoeffs[ %i ] = %7.5f\n", i, distCoeffs.ptr<double>(0)[i]);
    }*/
    
    Mat fundamental_matrix;

    // Array of array for frames key points
    vector<vector<Point2d>> points2frame(2);
    //vector<vector<KeyPoint>> points2frame(2);
    //vector<vector<Mat>> points2frame(2);
    Matx34d P1, P2;
    //vector<Matx34d> Ps;
    vector<Mat> Ps, points_3d;
    
    
    
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
    
    
    int c;
    while(1) {                      // Основной режим работы камеры, потоковый режим
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
                        
                        std::vector<cv::Point3f> lines[2];
                        cv::computeCorrespondEpilines(points1, 1 , fundamental_matrix, lines[0]);
                        cv::computeCorrespondEpilines(points2, 2 , fundamental_matrix, lines[1]);
                                                
                        Mat frame_epipol1 = drawlines(frame, lines[0], points1);
                        Mat frame_epipol2 = drawlines(frame2, lines[1], points2);
                                           
                        Rect r1(0, 0, frame_epipol1.cols, frame_epipol1.rows);                // Создаем фрагменты для склеивания зображения
                        Rect r2(frame_epipol2.cols, 0, frame_epipol2.cols, frame_epipol2.rows);
                        frame_epipol1.copyTo(frame4( r1 ));
                        frame_epipol2.copyTo(frame4( r2 ));                    
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
        }                           // Покадровый режим работы камеры
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
                detector->detectAndCompute(frame, noArray(), keypoints_frame, descriptors_frame);       // Detected key points at frame
                if ( !frame2.empty() ) 
                {
                    detector->detectAndCompute(frame2, noArray(), keypoints_frame2, descriptors_frame2);    // Detected key points at frame2
                    
                    RNG rng(12345);     // Drawing found key points 
                    for (unsigned long i = 0; i < keypoints_frame.size(); i++) {
                        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                        circle(frame, keypoints_frame[i].pt, 3, color, 2, LINE_8, 0);
                        circle(frame2, keypoints_frame2[i].pt, 3, color, 2, LINE_8, 0);
                    }
                    
                    if ((keypoints_frame.size() != 0) && (keypoints_frame2.size() != 0)) {  // Matching key points
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
                                        20 );
                        drawMatches(frame, good_points1, frame2, good_points2, good_matches, frame4);
                        imshow("1-2 frame", frame4);
                        
                        
                        
                        for (unsigned int i = 0; i < good_points1.size(); i++) { // Unioning the key points in new variable
                            points2frame[0].push_back(good_points1[i].pt);
                            points2frame[1].push_back(good_points2[i].pt);
                        }
                        
                        sfm::reconstruct(points2frame, Ps, points_3d, intrinsic, true);
                        
                        /*if ((good_points1.size() >= 8) && (good_points2.size() >= 8)) { // Проверка на наличие точек, удовлетворяющих порогу
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
                            
                            if (!fundamental_matrix.empty()) {  // Проверка на пустотности фундаментальной матрицы
                                fm::projectionsFromFundamental(fundamental_matrix, P1, P2);    // Расчет матриц проекции
                                Ps.push_back(P1);
                                Ps.push_back(P2);
                                triangulatePoints(points2frame, Ps, points_3d);
                            }
                        }*/
                        
                        //projectionsFromFundamental(F, P, Pp);
                        
                        //reconstruct(points2frame, Ps, points_3d, intrinsic, true);
                        // OpenCV data types
                        /*std::vector<Mat> pts2d;
                        points2frame
                        points2d.getMatVector(pts2d);
                        const int depth = pts2d[0].depth();
                        
                        // Get Projection matrices
                        Matx33d F;
                        Matx34d P, Pp;
                
                        normalizedEightPointSolver(pts2d[0], pts2d[1], F);
                        projectionsFromFundamental(F, P, Pp);
                        Ps.create(2, 1, depth);
                        Mat(P).copyTo(Ps.getMatRef(0));
                        Mat(Pp).copyTo(Ps.getMatRef(1));
                
                        // Triangulate and find 3D points using inliers
                        triangulatePoints(points2d, Ps, points3d);*/
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
        }                           // Калибровка камеры  press "с" or "C"----------------------//      step 0
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
                        printf("Snap stored!\n");

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
        }                           // END Калибровка камеры  ----------------------------------//   END step 0

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
            }
        } 
        // END Обработка нажатой кнопки  ----------------------------------------------------//
    }

    cap.release();
    return 0;
}
