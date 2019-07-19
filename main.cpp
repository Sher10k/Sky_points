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

void RtxRt(Mat *dR1, Mat *dt1, Mat *dR2, Mat *dt2)
{
// --- Temporary matrices ---------------------------------------------------//
    Mat tempRt1 = Mat::zeros(4, 4, CV_64F);
    tempRt1.at< double >(3, 3) = 1;
    Mat tempRt2 = Mat::zeros(4, 4, CV_64F);
    tempRt2.at< double >(3, 3) = 1;
    Mat ResultRt = Mat::zeros(4, 4, CV_64F);
    for (int i = 0; i < 3; i++) 
    {
        for (int j = 0; j < 3; j++)
        {
            tempRt1.at< double >(i, j) = dR1->at<double>(i, j);
            tempRt2.at< double >(i, j) = dR2->at< double >(i, j);
        }
    }
    for (int i = 0; i < 3; i++) 
    {
        tempRt1.at< double >(i, 3) = dt1->at< double >(i, 0);
        tempRt2.at< double >(i, 3) = dt2->at< double >(i, 0);
    }
    cout << "TempRt1 = " << endl;
    for (int i = 0; i < tempRt1.rows; i++)
    {
        for (int j = 0; j < tempRt1.cols; j++)
            cout << " " << tempRt1.at< double >(i, j) << "\t\t";
        cout << endl;
    }
    cout << "TempRt2 = " << endl;
    for (int i = 0; i < tempRt2.rows; i++)
    {
        for (int j = 0; j < tempRt2.cols; j++)
            cout << " " << tempRt2.at< double >(i, j) << "\t\t";
        cout << endl;
    }
    
// --- Multiplication matrix ------------------------------------------------//
    for (int i = 0; i < 4; i++) 
    {
        for (int j = 0; j < 4; j++)
        {
            ResultRt.at< double >(i, j) = (tempRt1.at< double >(i, 0)) * (tempRt2.at< double >(0, j)) + 
                                          (tempRt1.at< double >(i, 1)) * (tempRt2.at< double >(1, j)) + 
                                          (tempRt1.at< double >(i, 2)) * (tempRt2.at< double >(2, j)) +
                                          (tempRt1.at< double >(i, 3)) * (tempRt2.at< double >(3, j));
        }
    }
    for (int i = 0; i < 3; i++) 
    {
        for (int j = 0; j < 3; j++)
        {
            dR1[0].at< double >(i, j) = ResultRt.at< double >(i, j);
        }
    }
    for (int i = 0; i < 3; i++) 
    {
        dt1[0].at< double >(i, 0) = ResultRt.at< double >(i, 3);
    }
// --- Print matrix ---------------------------------------------------------//
    cout << "ResultRt = " << endl;
    for (int i = 0; i < ResultRt.rows; i++)
    {
        for (int j = 0; j < ResultRt.cols; j++)
            cout << " " << ResultRt.at< double >(i, j) << "\t\t";
        cout << endl;
    }
//    cout << "R1 = " << endl;
//    for (int i = 0; i < dR1->rows; i++)
//    {
//        for (int j = 0; j < dR1->cols; j++)
//            cout << " " << dR1->at< double >(i, j) << "\t\t";
//        cout << endl;
//    }
//    cout << "t1 = " << endl;
//    for (int i = 0; i < dt1->rows; i++)
//    {
//        for (int j = 0; j < dt1->cols; j++)
//            cout << " " << dt1->at< double >(i, j) << "\t\t";
//        cout << endl;
//    }
}

void RtxXYZ(Mat *dR, Mat *dt, Mat *p3d)
{
    Mat tempRt = Mat::zeros(4, 4, CV_64F);
    tempRt.at< double >(3, 3) = 1;
    for (int i = 0; i < 3; i++) 
    {
        for (int j = 0; j < 3; j++)
        {
            tempRt.at< double >(i, j) = dR->at<double>(i, j);
        }
    }
    for (int i = 0; i < 3; i++) 
    {
        tempRt.at< double >(i, 3) = dt->at< double >(i, 0);
    }
    for (int n = 0; n < p3d->cols; n++) 
    {
        for (int i = 0; i < 4; i++) 
        {
            p3d->at< double >(i, n) = (tempRt.at< double >(i, 0)) * (p3d->at< double >(0, n)) + 
                                      (tempRt.at< double >(i, 1)) * (p3d->at< double >(1, n)) + 
                                      (tempRt.at< double >(i, 2)) * (p3d->at< double >(2, n)) +
                                      (tempRt.at< double >(i, 3)) * (p3d->at< double >(3, n));
        }
    }
//    for (int i = 0; i < p3d->cols; i++)
//    {
//        cout << "3Dpoint[ " << i << " ] =";
//        for (int j = 0; j < p3d->rows; j++){
//            cout << " " << p3d->at<double>(j, i) << " ";
//        }
//        cout << endl;
//    }
}

int main()
{
    //-------------------------------------- VARIABLES ------------------------------------------//
    Mat frameRAW, frame, frameCache;
    
        // Camera position matrix
    double dataR[9] = { 1, 0, 0, 
                        0, 1, 0, 
                        0, 0, 1 };
    vector< Mat > Rotation(2);
    Rotation[0] = Mat(3, 3, CV_64F, dataR);
    Rotation[1] = Mat(3, 3, CV_64F, dataR);
    double datat[3] = { 0, 0, 0 };
    vector< Mat > translation(2);
    translation[0] = Mat(3, 1, CV_64F, datat);
    translation[1] = Mat(3, 1, CV_64F, datat);
    
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
    viewer->setCameraFieldOfView(0.523599); // 0.523599
    viewer->setCameraClipDistances(0, 100);
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
            
            int button_nf = waitKey(1);
            if ( button_nf == 32 )             // If press "space"
            {
                MySFM.Reconstruction3D(&frameCache, &frame, Calib.cameraMatrix);    // Put old frame then new frame
                if (!MySFM.R.empty() && !MySFM.t.empty())
                {
                    RtxRt(&Rotation[0], &translation[0], &MySFM.R, &MySFM.t);
                }
                
                if (!MySFM.points3D.empty())
                {
                    RtxXYZ(&Rotation[1], &translation[1], &MySFM.points3D);
                    for (int i = 0; i < 3; i++) 
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            Rotation[1].at< double >(i, j) = Rotation[0].at<double>(i, j);
                        }
                    }
                    for (int i = 0; i < 3; i++) 
                    {
                        translation[1].at< double >(i, 3) = translation[0].at< double >(i, 0);
                    }
                    for (int i = 0; i < MySFM.points3D.cols; i++)
                    {
                        //cout << "3Dpoint[ " << i << " ] =";
                        for (int j = 0; j < MySFM.points3D.rows; j++){
                            MySFM.points3D.at<double>(j, i) /= MySFM.points3D.at<double>(3, i);
                            //cout << " " << points3D.at<double>(j, i) << " ";
                        }
                        //cout << endl;
                    }
                    
                        // 3D points cloud
                    cloud.height = 1;
                    cloud.width = static_cast<unsigned int>( MySFM.points3D.cols );
                    cloud.is_dense = false;
                    cloud.points.resize( cloud.width * cloud.height );
                    
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
        }
        //------------------------------------ END MENU -----------------------------------------//
    }
    //------------------------------------------- END -------------------------------------------//
    
    cap.release();
    return 0;
}


