#include <iostream>
#include <string>
#include <stdio.h>

#include <opencv2/core.hpp>
//#include <opencv2/core/types_c.h>
//#include <opencv2/core/utility.hpp>
//#include <opencv2/core/ocl.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui.hpp>
//#include <opencv2/video/tracking.hpp>
//#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>  // For FundamentalMat
//#include <opencv2/sfm.hpp>

using namespace cv;
using namespace std;
//using namespace cv::sfm;

Mat drawlines(Mat& img1, std::vector<cv::Point3f>& line, vector<Point2f>& pts){         // Draw epipolar lines
    
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

int main()
{
    /*printf("OPENCV_OPENCL_DEVICE='%s'\n", getenv("OPENCV_OPENCL_DEVICE"));
    cv::ocl::setUseOpenCL(true);
    ocl::Context context = ocl::Context::getDefault();*/

    Mat frame, frame2, frame3, frameImg, frameImg2, frame_grey, flow, img2Original;
    double frame_MSEC, frame_MSEC2, frame_pause = 0;
    int thresh = 200;
    int max_thresh = 255;

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
        cap.set(CAP_PROP_FRAME_WIDTH, 640); //320, 640, (640, 1280)
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);   //240, 480, (360, 720)
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
    int key_num = 100;  // Num keypoint
    Ptr<FeatureDetector> detector = ORB::create(key_num, 1.2f, 4, 31, 0, 2, ORB::HARRIS_SCORE, 31);
    vector<KeyPoint> keypoints_frame, keypoints_frame2;
    Mat descriptors_frame, descriptors_frame2;

    // Brute-Force Matcher
    Ptr<BFMatcher> bf = BFMatcher::create(NORM_HAMMING, true);
    BFMatcher matcher(NORM_HAMMING);    // NORM_L2
    vector<DMatch> matches;
    DMatch dist1, dist2, temp;

    int t = 0;              // Счетчик найденых точек
    int f = 2;              // Переключение в режим калибровки
    bool mode_cam = true;   // Режим работы камеры
    Mat res;
    Mat frame4 = Mat::zeros(Size(2 * frame.cols, frame.rows), CV_8UC3);
    namedWindow("frame_epipol_double", WINDOW_AUTOSIZE);                    // Window for output result

    // Для калибровки
    Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;

    Mat fundamental_matrix;

    char c;
    while(1) {
        if ( (f == 1) && (mode_cam == true) ) {   // Основной режим работы камеры, потоковый режим
            //  Wait for a new frame from camera and store it into 'frame'
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

            // ORB detector        -------------------------------------------------------------//      step 1
            if (!cap.read(frameImg2)) { // check if we succeeded and store new frame into 'frame2'
                cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            undistort(frameImg2, frame2, intrinsic, distCoeffs);

            detector->detectAndCompute(frame, noArray(), keypoints_frame, descriptors_frame);       // Detected key points at frame
            detector->detectAndCompute(frame2, noArray(), keypoints_frame2, descriptors_frame2);    // Detected key points at frame2

            if ((keypoints_frame.size() != 0) && (keypoints_frame2.size() != 0)) {          // Проверка на нахождение хотябы одной контрольной точки
                
                matcher.match(descriptors_frame, descriptors_frame2, matches, noArray());   // Matches key points of the frame with key points of the frame2
                
                // Sort match
                key_num = static_cast<int>(matches.size());
                for (int i = 0; i < key_num - 1; i++) { //key_num - 1
                    dist1 = matches[i];
                    for (int j = i + 1; j < key_num; j++) {
                        dist2 = matches[j];
                        if (dist2.distance < dist1.distance) {
                            dist1 = matches[j];
                            t = j;
                        }
                        if (j == key_num - 1) {
                            matches[t] = matches[i];
                            matches[i] = dist1;
                            break;
                        }
                    }
                }

                // Selection of key points on both frames satisfying the threshold
                t = 0;
                vector<DMatch> good_matches;
                vector<KeyPoint> inliers1, inliers2;
                for (size_t i = 0; i < matches.size(); i++) {
                    if (matches[i].distance < 10) {
                        int new_i = static_cast<int>(inliers1.size());
                        inliers1.push_back(keypoints_frame[matches[i].queryIdx]);
                        inliers2.push_back(keypoints_frame2[matches[i].trainIdx]);
                        good_matches.push_back(DMatch(new_i, new_i, 0));
                        t ++;
                    }
                }
                // drawing two frames and connecting similar cue points with lines
                drawMatches(frame, inliers1, frame2, inliers2, good_matches, res);
                
                // END ORB detector        ------------------------------------------------------------//   END step 1
                vector<Point2f> points1(t);
                vector<Point2f> points2(t);
                vector<Point2f> status(t);
                
                if ((inliers1.size() >= 8) && (inliers2.size() >= 8)) { // Проверка на наличие точек, удовлетворяющих порогу
                    // FindFundamentalMat     -------------------------------------------------------------//      step 2
                    for (int i = 0; i < t; i++) {
                        points1[i].x = inliers1[i].pt.x;
                        points1[i].y = inliers1[i].pt.y;
                        points2[i].x = inliers2[i].pt.x;
                        points2[i].y = inliers2[i].pt.y;
                    }
                    fundamental_matrix = cv::findFundamentalMat(points1, points2, FM_RANSAC, 1.0, 0.99, noArray());
//                    for(int i = 0; i < fundamental_matrix.rows; i++){     // Отображение элементов 
//                        for(int j = 0; j < fundamental_matrix.cols; j++){
//                            //printf("fundamental_matrix[%.0f ", ptr[j]);
//                            printf("fundamental_matrix [ %i ][ %i ] = %5.7f\n", i, j, fundamental_matrix.at<double>(j,i));
//                        }
//                        //printf("\n");
//                    }
//                    //printf("-----\n");
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
                } else imshow("frame_epipol_double", res);
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
        }
        else if ((f == 1) && (mode_cam == false)) { // Покадровый режим работы камеры
            
        }         // Калибровка камеры  press "с" or "C"--------------------------------------------//      step 0
        else if ( f == 2 ) {
            int successes = 0;                                      // Счетчик удачных калибровочных кадров
            int num_successes = 3;                                    // Кол-во калибровочных кадров
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

            int batton = 0;

            while ( successes < num_successes ) {                        // Цикл для определенного числа калибровочных кадров
                batton = static_cast<char>(waitKey(1));
                if( batton == 27 ) {                            // Interrupt the cycle calibration, press "ESC"
                    break;
                } else if ( batton == 13 ) {                    // Take picture Chessboard, press "Enter"
                    if (!cap.read(calib_frame)) {               // check if we succeeded and store frame into calib_frame
                        cerr << "ERROR! blank frame grabbed\n";
                        break;
                    }
                    cvtColor(calib_frame, calib_frame_grey, CV_BGR2GRAY);
                        // Поиск углов на шахматной доске
                    bool found = findChessboardCorners(calib_frame,
                                                       board_sz,
                                                       calib_frame_corners,
                                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE); //CALIB_CB_NORMALIZE_IMAGE, CV_CALIB_CB_FILTER_QUADS

                    if (found) {    // Проверка удачно найденых углов
                        cornerSubPix(calib_frame_grey,
                                     calib_frame_corners,
                                     Size(11,11),
                                     Size(-1,-1),
                                     TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));   // Уточнение углов

                        drawChessboardCorners(calib_frame,
                                              board_sz,
                                              calib_frame_corners,
                                              found);                               //отрисовка углов

                        image_points.push_back(calib_frame_corners);
                        object_points.push_back(obj);
                        printf("Snap stored!\n");

                        successes++;
                    }
                } else if ( (batton == 114) || (batton == 82) ) {       // Read from file, press "p" or "P"
                    break;
                }
                cap.read(frame3);
                Rect r1(0, 0, frame3.cols, frame3.rows);                // Создаем фрагменты для склеивания зображения
                Rect r2(frame3.cols, 0, frame3.cols, frame3.rows);
                frame3.copyTo(frame4( r1 ));
                calib_frame.copyTo(frame4( r2 ));
                imshow("calibration", frame4);      // Вывод последнего удачного калибровачного кадра и кадра потока

                batton = 0;
            }
            //if( batton == 27 ) break;

            if (( (batton == 114) || (batton == 82) )) {
                FileStorage fs; 
                fs.open("/home/roman/Sky_points/Calibrate_cam.txt", FileStorage::READ);     // Read from file data calibration
                fs["intrinsic"] >> intrinsic;
                fs["distCoeffs"] >> distCoeffs;
                fs["rvecs"] >> rvecs;
                fs["tvecs"] >> tvecs;
                fs.release();                
            } else {
                calibrateCamera(object_points,
                                image_points,
                                calib_frame.size(),
                                intrinsic,
                                distCoeffs,
                                rvecs,
                                tvecs,
                                CALIB_FIX_K4|CALIB_FIX_K5);                                 // Calibrate
    
                FileStorage fs;
                fs.open("/home/roman/Sky_points/Calibrate_cam.txt", FileStorage::WRITE);    // Write in file data calibration
                fs << "intrinsic" << intrinsic;
                fs << "distCoeffs" << distCoeffs;
                fs << "rvecs" << rvecs;
                fs << "tvecs" << tvecs;
                fs.release();
            }            

            destroyWindow("calibration");
            f = 1;
        }         // END Калибровка камеры  --------------------------------------------------------//   END step 0

        // Обработка нажатой кнопки  --------------------------------------------------------//
        c = static_cast<char>(waitKey(1));
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
        } else if ( (c == 80) || (c == 112) ) {     // Change mode camera, press "p" or "P"
            mode_cam = !mode_cam;
        } 
        // END Обработка нажатой кнопки  ----------------------------------------------------//
    }

    cap.release();
    return 0;
}
