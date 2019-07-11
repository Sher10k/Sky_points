
#include "Header/camcalibration.h"

using namespace std;
using namespace cv;

CalibrationCamera::CalibrationCamera(VideoCapture *data_CAP)
{
    setParam(data_CAP);
}

void CalibrationCamera::setParam(VideoCapture *data_CAP)
{
    CAP = *data_CAP;
    Mat frame;
    CAP.read(frame);
    width_frame = frame.cols;
    height_frame = frame.rows;
}

void CalibrationCamera::printParam()
{
    cout    << "CAP = " << CAP.getBackendName() << endl;
//    cout    << "Width = " << CAP.get(CAP_PROP_FRAME_WIDTH) << endl
//            << "Height = " << CAP.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "cameraMatrix = " << endl;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            cout << " " << cameraMatrix(i, j) << "\t\t";
        cout << endl;
    }
    
    cout << "distCoeffs = " << endl;
    for (int j = 0; j < 5; j++)
        cout << " " << distCoeffs(0, j) << "\t";
    cout << endl;
    
    cout << "rvecs = " << endl;
    for (unsigned long i = 0; i < rvecs.size(); i++)
    {
        cout << i << " = [  ";
        for (int j = 0; j < rvecs[i].rows; j++)
            cout << rvecs[i].at<double>(j) << "   ";
        cout << " ]" << endl;
    }
    
    cout << "tvecs = " << endl;
    for (unsigned long i = 0; i < tvecs.size(); i++)
    {
        cout << i << " = [  ";
        for (int j = 0; j < tvecs[i].rows; j++)
            cout << tvecs[i].at<double>(j) << "   ";
        cout << " ]" << endl;
    }
}

void CalibrationCamera::calibratCamera(int numCornersHor,           // Кол-во углов по вертикале и горизонтале,
                                       int numCornersVer,           // на 1 меньше чем кол-во квадратов по вертикале и горизонтале
                                       int num_successes            // Кол-во калибровочных кадров
                                       )
{
    Mat frame, frame2, frame3;
    Mat frame4 = Mat::zeros(Size(2 * width_frame, height_frame), CV_8UC3);
    
    int successes = 0;                                                              // Счетчик удачных калибровочных кадров
    Mat calib_frame = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);         // Калибровочный кадр,  frame.size()
    Mat calib_frame_grey = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);
    
        // Chess bord variables
    int numSquares = numCornersHor * numCornersVer;                     // Board square
    Size board_sz = Size(numCornersHor, numCornersVer);                 // Size board
    vector<vector<Point2f>> image_points;                               // Точки на изображении
    vector<Point2f> calib_frame_corners;                                // Найденые вершины на шахматной доске
    vector<vector<Point3f>> object_points;                              //
    vector<Point3f> obj;
    
    for(int j = 0; j < numSquares; j++)
        obj.push_back(Point3d(j/numCornersHor, j%numCornersHor, 0.0));  //static_cast<double>(0.0f)
    
    int batton_calib = 0;
    while ( successes < num_successes ) {                               // Цикл для определенного числа калибровочных кадров
        batton_calib = waitKey(1);
        if( batton_calib == 27 ) {                                      // Interrupt the cycle calibration, press "ESC"
            break;
        } else if ( batton_calib == 13 ) {                              // Take picture Chessboard, press "Enter"
            if (!CAP.read(calib_frame)) {                               // check if we succeeded and store frame into calib_frame
                cerr << "ERROR! blank frame grabbed\n";
                break;
            }
                // Поиск углов на шахматной доске
            bool found = findChessboardCorners(calib_frame,
                                               board_sz,
                                               calib_frame_corners,
                                               CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE ); //CALIB_CB_NORMALIZE_IMAGE, CV_CALIB_CB_FILTER_QUADS

            if (found) {    // Проверка удачно найденых углов
                cvtColor(calib_frame, calib_frame_grey, COLOR_BGR2GRAY);                            //CV_BGR2GRAY
                cornerSubPix(calib_frame_grey,
                             calib_frame_corners,
                             Size(11,11),
                             Size(-1,-1),
                             TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1) );   // Уточнение углов
                drawChessboardCorners(calib_frame,
                                      board_sz,
                                      calib_frame_corners,
                                      found );                                                      //отрисовка углов
                image_points.push_back(calib_frame_corners);
                object_points.push_back(obj);
                cout << "Snap stored!" << endl;

                successes++;
            }
        } else if ( (batton_calib == 114) || (batton_calib == 82) || (batton_calib == 48) || (batton_calib == 27) ) {       
            // Read from file, press "r" or "R", or use default parameters, press "0", or interrupt the calibration cycle, press "ESC" 
            break;
        }
        CAP.read(frame3);
        Rect r1(0, 0, frame3.cols, frame3.rows);                // Создаем фрагменты для склеивания зображения
        Rect r2(frame3.cols, 0, frame3.cols, frame3.rows);
        frame3.copyTo(frame4( r1 ));
        calib_frame.copyTo(frame4( r2 ));
        imshow("calibration", frame4);      // Вывод последнего удачного калибровачного кадра и кадра потока

        batton_calib = 0;
    }
    namedWindow("calibration", WINDOW_AUTOSIZE);
    destroyWindow("calibration");
    //cout << "batton_calib = " << batton_calib << endl;
    if ( (batton_calib == 114) || (batton_calib == 82) ) {  // Read from file
        FileStorage fs; 
        fs.open("Calibrate_cam.txt", FileStorage::READ);     // Read from file data calibration
        fs["intrinsic"] >> cameraMatrix;
        fs["distCoeffs"] >> distCoeffs;
        fs["rvecs"] >> rvecs;
        fs["tvecs"] >> tvecs;
        fs.release();
        
    } else if (batton_calib == 48) {    // use default parameters   
        cameraMatrix(0, 0) = 600;  // fx
        cameraMatrix(1, 1) = 600;  // fy
        cameraMatrix(0, 2) = 320;  // Cx, half of width frame
        cameraMatrix(1, 2) = 240;  // Cy, half of hight frame
        cameraMatrix(2, 2) = 1;
        for (int i = 0; i < 5; i++) distCoeffs(0, i) = 0.0;
        /*distCoeffs(0, 0) = 0.0; // k1
        distCoeffs(0, 1) = 0.0; // k2
        distCoeffs(0, 2) = 0.0; // p1
        distCoeffs(0, 3) = 0.0; // p2
        distCoeffs(0, 4) = 0.0; // k3*/
        
        FileStorage fs;
        fs.open("Calibrate_cam_Zero.txt", FileStorage::WRITE);    // Write in file data calibration
        fs << "intrinsic" << cameraMatrix;
        fs << "distCoeffs" << distCoeffs;
        fs.release();
        
    } else if (batton_calib != 27) {    // interrupt the calibration cycle
        calibrateCamera(object_points,
                        image_points,
                        calib_frame.size(),
                        cameraMatrix,
                        distCoeffs,
                        rvecs,
                        tvecs,
                        CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3);                                 // Calibrate  | CALIB_FIX_K6
        FileStorage fs;
        fs.open("Calibrate_cam.txt", FileStorage::WRITE);    // Write in file data calibration
        fs << "intrinsic" << cameraMatrix;
        fs << "distCoeffs" << distCoeffs;
        fs << "rvecs" << rvecs;
        fs << "tvecs" << tvecs;
        fs.release();
    }
}




