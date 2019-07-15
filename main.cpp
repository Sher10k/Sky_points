/*  
 *  Горячии клавиши
 * 
 * "ESC" - Interrupt the cycle
 * "Enter" - Take picture
 * "с" & "C" - Calibrate camera
 * "r" or "R" - Read from file, calibrate mode
 * "0" - use default parameters, calibrate mode
 * "l" & "L" -  Output flow into file 
 * "f" & "F" - Output fundamental_matrix into file
 * "Space" - Сделать снимок для покадрового режима
 * 
 */

#define CERES_FOUND true
#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <iostream>
//#include <string>
//#include <stdio.h>
//#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
//#include <opencv2/core/types_c.h>
//#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
//#include <opencv2/core/ocl.hpp>
//#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui.hpp>
//#include <opencv2/video.hpp>
//#include <opencv2/video/tracking.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include <opencv2/calib3d.hpp>  // For FundamentalMat

#include <opencv2/sfm.hpp>
//#include <opencv2/sfm/simple_pipeline.hpp>
#include <opencv2/sfm/reconstruct.hpp>
//#include <opencv2/sfm/robust.hpp>
#include <opencv2/sfm/triangulation.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/console/parse.h>
//#include <pcl/features/normal_3d.h>
//#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/cloud_viewer.h>

//#include <boost/thread/thread.hpp>

#include "Header/camcalibration.h"
#include "Header/sfm_train.h"

using namespace std;
using namespace cv;
using namespace cv::sfm;
using namespace cv::xfeatures2d;
using namespace pcl;

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


int main()
{
    //-------------------------------------- VARIABLES ------------------------------------------//
    Mat frameRAW, frame, frameCache; 
//    double frame_MSEC, frame_MSEC2; 
//    int thresh = 200;
//    int max_thresh = 255;
    double frame_pause = 0;
    frame_pause = frame_pause / 30 * 1000;  // Convertion from frames per second to msec

        // Fundamental matrix
    Mat fundamental_matrix;

        // Epipolar linu
    std::vector<cv::Point3f> lines[2];
    Mat frame_epipol1, frame_epipol2;
    
        //  Array of array for frames key points  
    Mat Pt1 = cv::Mat::eye(3, 4, CV_64F);   // Projection matrices for each camera
    Mat Pt2 = cv::Mat::eye(3, 4, CV_64F);  
    vector<Mat> Ps(2); // Matx34d          // Vector of projection matrices for each camera
    Ps[0] = cv::Mat(3, 4, CV_64F);
    Ps[1] = cv::Mat(3, 4, CV_64F);
    
        // SFM camera
    /*cv::sfm::libmv_CameraIntrinsicsOptions camera {SFM_DISTORTION_MODEL_DIVISION, 
                                                    intrinsic(0, 0), 
                                                    intrinsic(1, 1), 
                                                    intrinsic(0, 2), 
                                                    intrinsic(1, 2),
                                                    distCoeffs(0, 0),
                                                    distCoeffs(0, 1),
                                                    distCoeffs(0, 4),
                                                    distCoeffs(0, 2),
                                                    distCoeffs(0, 3)};*/
    
        // Cloud of points
    //std::vector<pcl::visualization::Camera> camera; 
    pcl::PointCloud <pcl::PointXYZ> cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    // Evil functions
    viewer->initCameraParameters();
    viewer->setCameraPosition(5, 5, 5,    0, 0, 0,   0, 0, 1);
    viewer->setCameraFieldOfView(0.523599);
    viewer->setCameraClipDistances(0.00522511, 50);
    
        // Other variables
    int f = 2;              // Переключение в режим калибровки
    Mat frame4 = Mat::zeros(Size(2 * frame.cols, frame.rows), CV_8UC3);
    char nF = 1;
    int win = 3, vecS = 1;
    int click;
    Matx33d K_1;

    
    //-------------------------------------- Initialize VIDEOCAPTURE ----------------------------//
    VideoCapture cap;
    int deviceID = 0;                   //  camera 1
    int apiID = cv::CAP_ANY;            //  0 = autodetect default API
    cap.open(deviceID + apiID);         //  Open camera
    if(!cap.isOpened()) {               // Check if we succeeded
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    else {  //  Info about frame
        cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);     // 320, 640, (640, 1280)
        cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);   // 240, 480, (360, 720)
        //cap.set(CAP_PROP_POS_FRAMES, 0);              // Set zero-frame
        //cap.set(CAP_PROP_AUTOFOCUS, 0);               // Set autofocus
        //cap.set(CAP_PROP_FPS, 30);                    // Set FPS
        cap.read(frameRAW);

        cout << "Width = " << cap.get(CAP_PROP_FRAME_WIDTH) << endl
             << "Height = " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl
             << "FPS = " << cap.get(CAP_PROP_FPS) << endl
             //<< "nframes = " << cap.get(CAP_PROP_FRAME_COUNT) << endl
             //<< "Auto focus" << cap.get(CAP_PROP_AUTOFOCUS) << endl
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
    
    //-------------------------------------- Initialize calibration -----------------------------//
    CalibrationCamera Calib(&cap);
    Calib.printParam();
    
    //-------------------------------------- Initialize SFM -------------------------------------//
    SFM_Reconstruction MySFM(&cap);
    
    
    //------------------------------------------ START ------------------------------------------//
    while(1) {         
        if (f == 1) {     // Покадровый режим работы камеры
                //  Wait for a new frame from camera and store it into 'frameRAW'
            if (!cap.read(frameRAW)) { // check if we succeeded
                cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            undistort(frameRAW, frame, Calib.cameraMatrix, Calib.distCoeffs);
            imshow("real_time", frame);
            
                // Вывод предыдущего и текущего кадров вместе
            /*Rect r1(0, 0, frame.cols, frame.rows);
            Rect r2(frame2.cols, 0, frame2.cols, frame2.rows);
            frame.copyTo(frame4( r1 ));
            frame2.copyTo(frame4( r2 ));
            imshow("1-2 frame",frame4);*/
            
            int button_nf = waitKey(1);
            if ( button_nf == 32 )             // If press "space"
            {
                if (nF == 1)
                {
                    MySFM.detectKeypoints( &frame );
                    nF = 2;
                } 
                else 
                {
                    MySFM.detectKeypoints( &frame );
                    MySFM.matchKeypoints();
                    if (MySFM.numKeypoints > 7 )
                    {
                        MySFM.homo_fundam_Mat(K_1);
                        
                        MySFM.projectionsMat();
                        MySFM.triangulationPoints();
                        
                        drawMatches(MySFM.frame1, MySFM.good_points1, MySFM.frame2, MySFM.good_points2, MySFM.good_matches, MySFM.frame4);
                        imshow("1-2 frame", MySFM.frame4);
                        
                        
                            // 3D points cloud
                        //pcl::PointCloud <pcl::PointXYZ> cloud;
                        cloud.height = 1;
                        cloud.width = static_cast<unsigned int>( MySFM.points3D.cols );
                        cloud.is_dense = false;
                        cloud.points.resize( cloud.width * cloud.height );
                        
                        for (size_t i = 0; i < cloud.points.size (); ++i)
                        {
                            cloud.points[i].x = MySFM.points3D.at<float>(0, static_cast<int>(i));
                            cloud.points[i].y = MySFM.points3D.at<float>(1, static_cast<int>(i));
                            cloud.points[i].z = MySFM.points3D.at<float>(2, static_cast<int>(i));
                            //cloud.points[i].r = rgb_cenal[2].at(i);
                            //cloud.points[i].g = rgb_cenal[1].at(i);
                            //cloud.points[i].b = rgb_cenal[0].at(i);
                        }
                            // Save 3D points in file
                        pcl::io::savePCDFileASCII ("Reconstruct_cloud.pcd", cloud);
                            // Load 3D points (cloud points)
                        //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
                        pcl::io::loadPCDFile("Reconstruct_cloud.pcd", *cloud2);
                        
                        /*viewer->getCameras(camera);
                        //Print recorded points on the screen: 
                        cout << "camera: " << endl 
                             << " - pos: ("	<< camera[0].pos[0] << ", "	<< camera[0].pos[1] << ", "	<< camera[0].pos[2] << ")"	<< endl 
                             << " - view: ("	<< camera[0].view[0] << ", "	<< camera[0].view[1] << ", "	<< camera[0].view[2] << ")"	<< endl 
                             << " - focal: ("	<< camera[0].focal[0] << ", "	<< camera[0].focal[1] << ", "	<< camera[0].focal[2] << ")"	<< endl;	
                        */
                            // View cloud points
                        /*pcl::visualization::CloudViewer viewer("Cloud Viewer");
                        viewer.showCloud(cloud2, "cloud");*/
                        
                    } else {
                        // Вывод первого кадра и пустого кадра
                        MySFM.frame2 *= 0;
                        Rect r1(0, 0, MySFM.frame1.cols, MySFM.frame1.rows);
                        Rect r2(MySFM.frame2.cols, 0, MySFM.frame2.cols, MySFM.frame2.rows);
                        MySFM.frame1.copyTo(MySFM.frame4( r1 ));
                        MySFM.frame2.copyTo(MySFM.frame4( r2 ));
                        imshow("1-2 frame", MySFM.frame4);
                    }                    
                }
                MySFM.f1Tof2();
                MySFM.frame1 *= 0;
                
            } else if ( button_nf == 27 ) {                             // Interrupt the cycle, press "ESC"
                nF = 1;
                break;
            } else if ( (button_nf == 99) || (button_nf == 67) ) {      // Calibrate camera, press "с" & "C"
                nF = 1;
                f = 2;
                namedWindow("real_time", WINDOW_AUTOSIZE);
                destroyWindow("real_time");
                namedWindow("1-2 frame", WINDOW_AUTOSIZE);
                destroyWindow("1-2 frame");
            }
            
            if (cloud.size() != 0) {        // View cloud points
                    // Clear the view
                viewer->removeAllShapes();
                viewer->removeAllPointClouds();
                viewer->addPointCloud<pcl::PointXYZ>(cloud2, "sample cloud", 0);
                viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
                viewer->spinOnce (5);
            }
            
            
            /*if ( !frameCache.empty() )
            {
                if ( (button_nf == 43) || (button_nf == 61) )  
                {
                    win += 2;
                    if (win > frame.rows / 2) win = frame.rows / 2;
                    vecS ++;
                    if (vecS > frame.rows / 10) vecS = frame.rows / 10;
                    cout << "win = " << win << endl;
                    cout << "vecS = " << vecS << endl;
                }
                if ( (button_nf == 45) || (button_nf == 95) )
                {
                    win -= 2;
                    if (win <= 0) win = 1;
                    vecS --;
                    if (vecS <= 0) vecS = 1;
                    cout << "win = " << win << endl;
                    cout << "vecS = " << vecS << endl;
                }
                MySFM.opticalFlow(&frame, &frameCache, win, 1);
                imshow("OpticalFlow", MySFM.img2Original);
            }*/
            frame.copyTo(frameCache);
        }                                   
        else if ( f == 2 ) {                            // Калибровка камеры  press "с" or "C"----------------------//      step 0
            Calib.calibratCamera(8, 6, 10);
            Calib.printParam();
            invert(Calib.cameraMatrix, K_1, DECOMP_LU);
            f = 1;
        }                                               // END Калибровка камеры  ----------------------------------//   END step 0

        //-------------------------------------- MENU -------------------------------------------//
        click = waitKey(1);
        if( click == 27 ) {                                 // Interrupt the cycle, press "ESC"
            break;
        } else if ( click == 13 ) {                         // Take picture, press "Enter"
            imshow("foto", frame);
        } else if ( (click == 99) || (click == 67) ) {          // Calibrate camera, press "с" & "C"
            f = 2;
            namedWindow("real_time", WINDOW_AUTOSIZE);
            destroyWindow("real_time");
            namedWindow("1-2 frame", WINDOW_AUTOSIZE);
            destroyWindow("1-2 frame");
            namedWindow("OpticalFlow", WINDOW_AUTOSIZE);
            destroyWindow("OpticalFlow");
        } else if ( (click == 70) || (click == 102) ) {         // Output fundamental_matrix into file, press "f" & "F"
            FileStorage fundam;
            fundam.open("Fundamental_matrix.txt", FileStorage::WRITE);
            fundam << "homography_matrix" << MySFM.retval;
            fundam << "homography_mask" << MySFM.homo_mask;
            fundam << "fundamental_matrix" << MySFM.F;
            fundam << "fundamental_mask" << MySFM.Fundam_mask;
            fundam.release();
            cout << " --- Fundamental matrix written into file: Fundamental_matrix.txt" << endl << endl;
        } else if ( (click == 76) || (click == 108) ) {         // Output flow into file, press "l" & "L"
            FileStorage FLOW;
            FLOW.open("FLOW_frame.txt", FileStorage::WRITE);
            FLOW << "flow" << MySFM.flow;
            FLOW.release();
        }
        //------------------------------------ END MENU -----------------------------------------//
    }
    //------------------------------------------- END -------------------------------------------//
    
    cap.release();
    return 0;
}
