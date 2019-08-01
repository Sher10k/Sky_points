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
//#define EIGEN_RUNTIME_NO_MALLOC

#include <iostream>
//#include <string>
//#include <stdio.h>
//#include <thread>

#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>
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
//#include <pcl/impl/point_types.hpp>
//#include <pcl/console/parse.h>
//#include <pcl/features/normal_3d.h>
#include <pcl/visualization/common/common.h>
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
using namespace Eigen;

#define FRAME_WIDTH 640     // 320, 640, (640, 1280)
#define FRAME_HEIGHT 480    // 240, 480, (360, 720)
#define VIEWER_WIN_WIDTH 640
#define VIEWER_WIN_HEIGHT 480

void drawCamera( boost::shared_ptr < visualization::PCLVisualizer > &view, 
                 vector < visualization::Camera > *camera,
                 Scalar color,
                 vector < Matrix4d > *Rt, 
                 size_t cf )
{
    string str = "cam_";
    Vector4d CameraRt;
    CameraRt << camera->at(cf).pos[0], camera->at(cf).pos[1], camera->at(cf).pos[2], 1;
    CameraRt = Rt->at(cf) * CameraRt;
    camera->at(cf).pos[0] = CameraRt(0);
    camera->at(cf).pos[1] = CameraRt(1);
    camera->at(cf).pos[2] = CameraRt(2);
    CameraRt<< camera->at(cf).view[0], camera->at(cf).view[1], camera->at(cf).view[2], 1;
    CameraRt = Rt->at(cf) * CameraRt;
    camera->at(cf).view[0] = CameraRt(0);
    camera->at(cf).view[1] = CameraRt(1);
    camera->at(cf).view[2] = CameraRt(2);
    double f_length = sqrt( (CameraRt(0) * CameraRt(0)) + (CameraRt(1) * CameraRt(1)) + (CameraRt(2) * CameraRt(2)) );
    
    view->addSphere( PointXYZ( static_cast<float>(camera->at(cf).pos[0]), 
                               static_cast<float>(camera->at(cf).pos[1]), 
                               static_cast<float>(camera->at(cf).pos[2]) ), 
                       0.3, 
                       color[2] / 255, 
                       color[1] / 255, 
                       color[0] / 255, 
                       str + "sphere" + to_string(cf), 0 );
    ModelCoefficients cone_coeff;
    cone_coeff.values.resize (7);
    cone_coeff.values[0] = static_cast<float>( camera->at(cf).pos[0] );
    cone_coeff.values[1] = static_cast<float>( camera->at(cf).pos[1] );
    cone_coeff.values[2] = static_cast<float>( camera->at(cf).pos[2] );
    cone_coeff.values[3] = static_cast<float>( -camera->at(cf).view[0] / f_length  / 2 );
    cone_coeff.values[4] = static_cast<float>( -camera->at(cf).view[1] / f_length  / 2 );
    cone_coeff.values[5] = static_cast<float>( camera->at(cf).view[2] / f_length  / 2 );
    cone_coeff.values[6] = static_cast<float>( 15 );
    view->addCone( cone_coeff, str + "cone" + to_string(cf), 0 );
    
    cout << "Cam[ " << cf << " ]: " << endl 
         << " - pos: (" << camera->at(cf).pos[0] << ", " << camera->at(cf).pos[1] << ", " << camera->at(cf).pos[2] << ")" << endl 
         << " - clip: (" << camera->at(cf).clip[0] << ", " << camera->at(cf).clip[1] << ")" << endl
         << " - fovy: (" << camera->at(cf).fovy << ")" << endl
         << " - view: (" << camera->at(cf).view[0] << ", " << camera->at(cf).view[1] << ", " << camera->at(cf).view[2] << ")" << endl 
         << " - focal: (" << camera->at(cf).focal[0] << ", " << camera->at(cf).focal[1] << ", " << camera->at(cf).focal[2] << ")" << endl
         << " - window_pos: (" << camera->at(cf).window_pos[0] << ", " << camera->at(cf).window_pos[1] << ")" << endl
         << " - window_size: (" << camera->at(cf).window_size[0] << ", " << camera->at(cf).window_size[1] << ")" << endl
         << endl;
}

int main(int argc, char *argv[])  //int argc, char *argv[]
{
    //-------------------------------------- VARIABLES ------------------------------------------//
    Mat frameRAW, frame, frame2, frameCache;
    frameCache *= 0;
    frame2 = Mat::zeros(Size(2 * VIEWER_WIN_WIDTH, VIEWER_WIN_HEIGHT), CV_8UC3);
    
        // Cloud of points
    //std::vector<pcl::visualization::Camera> camera; 
    PointCloud < PointXYZRGB > cloud;
    PointCloud < PointXYZRGB > ::Ptr cloud2 ( new PointCloud < PointXYZRGB > );
    boost::shared_ptr < visualization::PCLVisualizer > viewer ( new visualization::PCLVisualizer ("3D Viewer") );
    vector < visualization::Camera > cam;
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0, "global");
    size_t cloud_flag = 0;
        // Camera position matrix
    vector < Matrix4d > Rt;
    Matrix4d one;
    one << 1,0,0,0,
           0,1,0,0,
           0,0,1,0,
           0,0,0,1;
    Rt.push_back(one);
    cout << "Rt[ " << cloud_flag << " ]= " << endl 
         << Rt[ cloud_flag ] << endl;
        // Init of camera default parameters
    viewer->getCameras(cam);
    cam[cloud_flag].pos[0] = 0;
    cam[cloud_flag].pos[1] = 0;
    cam[cloud_flag].pos[2] = 0;
    cam[cloud_flag].view[0] = 0;
    cam[cloud_flag].view[1] = 0;
    cam[cloud_flag].view[2] = 1;
    drawCamera( viewer, & cam, Scalar(0,0,255), & Rt ,cloud_flag );
    
        // Other variables
    int f = 2;              // Переключение в режим калибровки
    //int win = 3, vecS = 1;
    int click;
    Matx33d K_1;
    RNG rng(12345);
    
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

        cout << " --- VideoCapture" <<endl
             << "CAP = " << cap.getBackendName() << endl
             << "Width = " << cap.get(CAP_PROP_FRAME_WIDTH) << endl
             << "Height = " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl
             << "FPS = " << cap.get(CAP_PROP_FPS) << endl
             //<< "nframes = " << cap.get(CAP_PROP_FRAME_COUNT) << endl
             //<< "Auto focus" << cap.get(CAP_PROP_AUTOFOCUS) << endl
             << " --- " <<endl;

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
    
    //-------------------------------------- Initialize calibration -----------------------------//
    CalibrationCamera Calib(&cap);
    Calib.Read_from_file(0);
    Calib.printParam();
    f = 1;
    Matrix3d K;
    cv2eigen(Calib.cameraMatrix, K);
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
            
            Mat fL, fR;
            if (!frameCache.empty()) resize( frameCache, fL, Size(VIEWER_WIN_WIDTH, VIEWER_WIN_HEIGHT), 0, 0, INTER_LINEAR );
            else fL = Mat::zeros( Size(VIEWER_WIN_WIDTH, VIEWER_WIN_HEIGHT), CV_8UC3 );
            resize( frame, fR, Size(VIEWER_WIN_WIDTH, VIEWER_WIN_HEIGHT), 0, 0, INTER_LINEAR );
            Rect r1( 0, 0, fL.cols, fL.rows );
            Rect r2( fL.cols, 0, fR.cols, fR.rows );
            fL.copyTo(frame2( r1 ));
            fR.copyTo(frame2( r2 ));
            imshow( "Real time", frame2 );
            
            int button_nf = waitKey(1);
            if ( button_nf == 32 )             // If press "space"
            {
                    // SFM reconstruction
                //MySFM.Reconstruction3D( & frameCache, & frame, Calib.cameraMatrix );    // Put old frame then new frame
                MySFM.Reconstruction3DopticFlow( & frameCache, & frame, Calib.cameraMatrix );
                
                if (!MySFM.points3D.empty())
                {
                    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                    cloud_flag++;
                    Rt.push_back( Rt[ cloud_flag - 1 ] );
                    Matrix4d tempRt;
                    tempRt << MySFM.R.at<double>(0, 0), MySFM.R.at<double>(0, 1), -MySFM.R.at<double>(0, 2), -MySFM.t.at<double>(0, 0),
                              MySFM.R.at<double>(1, 0), MySFM.R.at<double>(1, 1), -MySFM.R.at<double>(1, 2), -MySFM.t.at<double>(0, 1),
                              -MySFM.R.at<double>(2, 0), -MySFM.R.at<double>(2, 1), MySFM.R.at<double>(2, 2), MySFM.t.at<double>(0, 2),
                              0,                        0,                        0,                        1;
                    cout << "tempRt[ " << cloud_flag << " ]= " << endl 
                         << tempRt << endl;
                    Rt[ cloud_flag ] = Rt[ cloud_flag ] * tempRt;
                    //Rt[ cloud_flag ] = tempRt * Rt[ cloud_flag ] ;
                    cout << "Rt[ " << cloud_flag << " ]= " << endl 
                         << Rt[ cloud_flag ] << endl;
                    
                        // 3D points cloud
                    cloud.height = 1;
                    cloud.width = static_cast<unsigned int>( MySFM.points3D.cols );
                    cloud.is_dense = false;
                    cloud.points.resize( cloud.width * cloud.height );
                    
                    for (size_t i = 0; i < cloud.points.size (); ++i)
                    {
                        Vector4d temp3Dpoint;
                        temp3Dpoint << MySFM.points3D.at< double >(0, static_cast<int>(i)), 
                                       MySFM.points3D.at< double >(1, static_cast<int>(i)),
                                       MySFM.points3D.at< double >(2, static_cast<int>(i)),
                                       MySFM.points3D.at< double >(3, static_cast<int>(i));
                        temp3Dpoint = Rt[ cloud_flag - 1 ] * temp3Dpoint;
                        cloud.points[i].x = static_cast<float>(temp3Dpoint(0));  // (float)
                        cloud.points[i].y = static_cast<float>(temp3Dpoint(1));
                        cloud.points[i].z = static_cast<float>(temp3Dpoint(2));
                        cloud.points[i].r = static_cast< uint8_t >( MySFM.points3D_BGR.at(i)[2] );
                        cloud.points[i].g = static_cast< uint8_t >( MySFM.points3D_BGR.at(i)[1] );
                        cloud.points[i].b = static_cast< uint8_t >( MySFM.points3D_BGR.at(i)[0] );
//                        cloud.points[i].r = static_cast< uint8_t >( color[2] );
//                        cloud.points[i].g = static_cast< uint8_t >( color[1] );
//                        cloud.points[i].b = static_cast< uint8_t >( color[0] );
                    }
                        // Save 3D points in file
                    pcl::io::savePCDFileASCII ("Reconstruct_cloud.pcd", cloud);
                        // Load 3D points (cloud points)
                    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::io::loadPCDFile("Reconstruct_cloud.pcd", *cloud2);  // test_pcd.pcd
                    
                    string str = "sample cloud";
                    str += to_string(cloud_flag);
                    
                    viewer->addPointCloud<pcl::PointXYZRGB>(cloud2, str, 0);
                    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, str);
                    viewer->initCameraParameters();
                    viewer->setCameraFieldOfView(0.5); // 0.523599 vertical field of view in radians
                    viewer->setCameraClipDistances(-1000, 2000);
                    viewer->setCameraPosition(-5, -5, -10,    0, 0, 10,   0, -1, 0);
                    viewer->getCameraParameters( argc, argv );
                    viewer->setPosition(0, 0);
                    
                        // Draw camera
                    cam.push_back(cam[ cloud_flag - 1 ]);
                    drawCamera( viewer, & cam, color, & Rt ,cloud_flag );
                    
//                    viewer->updatePointCloud<pcl::PointXYZ>(cloud2, "sample cloud");
//                    pcl::io::loadPCDFile("test_pcd.pcd", *cloud2);  // test_pcd.pcd
//                    viewer->addPointCloud<pcl::PointXYZ>(cloud2, "sample cloud", 0);
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
            }
            frame.copyTo(frameCache);*/
        }                                               // END Main loop -------------------------------------------//
        else if ( f == 2 ) {                            // Калибровка камеры  press "с" or "C"----------------------//      step 0
            //Calib.calibrCameraChess(10, 7, 10);    // 8, 6
            Calib.calibrCameraChArUco(11, 8, 10, 7, 10, 10);
            Calib.printParam();
            cv2eigen(Calib.cameraMatrix, K);
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
    
    viewer->close();
    cap.release();
    return 0;
}


