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


int main()
{
    //-------------------------------------- VARIABLES ------------------------------------------//
    Mat frameRAW, frame, frameCache;
    
        // Cloud of points
    //std::vector<pcl::visualization::Camera> camera; 
    pcl::PointCloud <pcl::PointXYZ> cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0, "global");
    viewer->setSize(30, 30);
    // Evil functions
    viewer->initCameraParameters();
    viewer->setCameraPosition(5, 5, 5,    0, 0, 0,   0, 0, 1);
    viewer->setCameraFieldOfView(0.523599);
    viewer->setCameraClipDistances(0.00522511, 1);
    char cloud_flag = 0;
    
        // Other variables
    int f = 2;              // Переключение в режим калибровки
    Mat frame4 = Mat::zeros(Size(2 * frame.cols, frame.rows), CV_8UC3);
    //int win = 3, vecS = 1;
    int click;
    Matx33d K_1;

    
    //-------------------------------------- Initialize VIDEOCAPTURE ----------------------------//
    VideoCapture cap;
    int deviceID = 1;                   //  camera 1
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
        double frame_MSEC, frame_MSEC2; 
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
    frame.copyTo(frameCache);
    frameCache *= 0;
    
    //-------------------------------------- Initialize calibration -----------------------------//
    CalibrationCamera Calib(&cap);
    Calib.printParam();
    
    //-------------------------------------- Initialize SFM -------------------------------------//
    SFM_Reconstruction MySFM(&cap);
    
    
    //------------------------------------------ START ------------------------------------------//
    while(1) {         
        if (f == 1) {                                   // Main loop -----------------------------------------------//
                //  Wait for a new frame from camera and store it into 'frameRAW'
            if (!cap.read(frameRAW)) { // check if we succeeded
                cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            undistort(frameRAW, frame, Calib.cameraMatrix, Calib.distCoeffs);
            imshow("Real time", frame);
            
                // Вывод предыдущего и текущего кадров вместе
            /*Rect r1(0, 0, frame.cols, frame.rows);
            Rect r2(frame2.cols, 0, frame2.cols, frame2.rows);
            frame.copyTo(frame4( r1 ));
            frame2.copyTo(frame4( r2 ));
            imshow("1-2 frame",frame4);*/
            
            int button_nf = waitKey(1);
            if ( button_nf == 32 )             // If press "space"
            {
                MySFM.Reconstruction3D(&frameCache, &frame, Calib.cameraMatrix);    // Put old frame then new frame
                
                if (!MySFM.points3D.empty())
                {
                        // 3D points cloud
                    cloud.height = 1;
                    cloud.width = static_cast<unsigned int>( MySFM.points3D.cols );
                    cloud.is_dense = false;
                    cloud.points.resize( cloud.width * cloud.height );
                    
//                    double X = 0, Y = 0, Z = 0, W = 0;
//                    float X1 = 5, Y1 = 7, Z1 = 13;
                    for (size_t i = 0; i < cloud.points.size (); ++i)
                    {
                        cloud.points[i].x = (float)MySFM.points3D.at<double>(0, static_cast<int>(i));
                        cloud.points[i].y = (float)MySFM.points3D.at<double>(1, static_cast<int>(i));
                        cloud.points[i].z = (float)MySFM.points3D.at<double>(2, static_cast<int>(i));
                    }
                        // Save 3D points in file
                    pcl::io::savePCDFileASCII ("Reconstruct_cloud.pcd", cloud);
                        // Load 3D points (cloud points)
                    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::io::loadPCDFile("Reconstruct_cloud.pcd", *cloud2);  // test_pcd.pcd
                    
                    string str = "sample cloud";
                    str += cloud_flag;
                    
                    viewer->addPointCloud<pcl::PointXYZ>(cloud2, str, 0);
                    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, str);                        
                    cloud_flag++;
                    
                    /*viewer->updatePointCloud<pcl::PointXYZ>(cloud2, "sample cloud");
                    pcl::io::loadPCDFile("test_pcd.pcd", *cloud2);  // test_pcd.pcd
                    viewer->addPointCloud<pcl::PointXYZ>(cloud2, "sample cloud", 0);*/
                }
                
                frame.copyTo(frameCache);
                
            } else if ( button_nf == 27 ) {                             // Interrupt the cycle, press "ESC"
                break;
            } else if ( (button_nf == 99) || (button_nf == 67) ) {      // Calibrate camera, press "с" & "C"
                f = 2;
                namedWindow("Real time", WINDOW_AUTOSIZE);
                destroyWindow("Real time");
                MySFM.destroyWinSFM();
            }
            
            if (cloud.size() != 0) {        // View cloud points
                    // Clear the view
//                viewer->removeAllShapes();
//                viewer->removeAllPointClouds();
//                viewer->addPointCloud<pcl::PointXYZ>(cloud2, "sample cloud", 0);
//                viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
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
        }                                               // END Main loop -------------------------------------------//
        else if ( f == 2 ) {                            // Калибровка камеры  press "с" or "C"----------------------//      step 0
            //Calib.calibrCameraChess(10, 7, 10);    // 8, 6
            Calib.calibrCameraChArUco(11, 8, 10, 7, 10, 10);
            Calib.printParam();
            invert(Calib.cameraMatrix, K_1, DECOMP_LU);
            f = 1;
        }                                               // END Калибровка камеры  ----------------------------------//   END step 0

        //-------------------------------------- MENU -------------------------------------------//
        click = waitKey(1);
        if( click == 27 ) {                                     // Interrupt the cycle, press "ESC"
            break;
        } else if ( click == 13 ) {                             // Take picture, press "Enter"
            imshow("foto", frame);
        } else if ( (click == 99) || (click == 67) ) {          // Calibrate camera, press "с" & "C"
            f = 2;
            namedWindow("Real time", WINDOW_AUTOSIZE);
            destroyWindow("Real time");
            MySFM.destroyWinSFM();
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
