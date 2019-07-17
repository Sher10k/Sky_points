
#include "Header/camcalibration.h"

CalibrationCamera::CalibrationCamera(cv::VideoCapture *data_CAP)
{
    setParam(data_CAP);
}

void CalibrationCamera::setParam(cv::VideoCapture *data_CAP)
{
    CAP = *data_CAP;
    cv::Mat frame;
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
    cout << endl;
}

void CalibrationCamera::calibrCameraChess(int _numCornersHor, 
                                       int _numCornersVer, 
                                       unsigned int _nFrames)
{
    Mat frame, frame2, frame3;
    Mat frame4 = Mat::zeros(Size(2 * width_frame, height_frame), CV_8UC3);
    Mat calib_frame = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);         // Калибровочный кадр,  frame.size()
    Mat calib_frame_grey = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);
    calibrationFlags = cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3;      // Calibration flags    | CALIB_FIX_K6
    
        // Chess bord variables
    int numSquares = _numCornersHor * _numCornersVer;                               // Board square
    Size board_sz = Size(_numCornersHor, _numCornersVer);                           // Size board
    vector<vector<Point2f>> image_points;                                           // Точки на изображении
    vector<Point2f> calib_frame_corners;                                            // Найденые вершины на шахматной доске
    vector<vector<Point3f>> object_points;                                          //
    vector<Point3f> obj;
    for(int j = 0; j < numSquares; j++)
        obj.push_back(Point3d(j/_numCornersHor, j%_numCornersHor, 0.0));            //static_cast<double>(0.0f)
    
    unsigned int successes = 0;                                         // Счетчик удачных калибровочных кадров
    int batton_calib = 0;
    
    while ( successes < _nFrames )                                // Цикл для определенного числа калибровочных кадров
    {
        batton_calib = cv::waitKey(1);
        if( batton_calib == 27 )                                        // Interrupt the cycle calibration, press "ESC"
        {
            break;
        } 
        else if ( batton_calib == 13 )                                  // Take picture Chessboard, press "Enter"
        {
            if (!CAP.read(calib_frame))                                 // check if we succeeded and store frame into calib_frame
            {
                std::cerr << "ERROR! blank frame grabbed\n";
                break;
            }
                // Find cernels on chessboard
            bool found = findChessboardCorners(calib_frame,
                                               board_sz,
                                               calib_frame_corners,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE ); //CALIB_CB_NORMALIZE_IMAGE, CV_CALIB_CB_FILTER_QUADS

            if (found)    // Проверка удачно найденых углов
            {
                cvtColor(calib_frame, calib_frame_grey, cv::COLOR_BGR2GRAY);                            //CV_BGR2GRAY
                cornerSubPix(calib_frame_grey,
                             calib_frame_corners,
                             cv::Size(11,11),
                             cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1) );   // Уточнение углов
                drawChessboardCorners(calib_frame,
                                      board_sz,
                                      calib_frame_corners,
                                      found );                                                      //отрисовка углов
                image_points.push_back(calib_frame_corners);
                object_points.push_back(obj);
                cout << "Snap stored!" << endl;

                successes++;
             }
        } 
        else if ( (batton_calib == 114) || (batton_calib == 82) || (batton_calib == 48) || (batton_calib == 27) )
        {       
            // Read from file, press "r" or "R", or use default parameters, press "0", or interrupt the calibration cycle, press "ESC" 
            break;
        }
        CAP.read(frame3);
            putText( frame3, 
                     "Press 'c' to add current frame. 'ESC' to finish and calibrate Chess_board",
                     Point(5, 20), 
                     FONT_HERSHEY_SIMPLEX, 
                     0.5, 
                     Scalar(255, 0, 0), 
                     2);
        cv::Rect r1(0, 0, frame3.cols, frame3.rows);                // Создаем фрагменты для склеивания зображения
        cv::Rect r2(frame3.cols, 0, frame3.cols, frame3.rows);
        frame3.copyTo(frame4( r1 ));
        calib_frame.copyTo(frame4( r2 ));
        imshow("calibration", frame4);      // Вывод последнего удачного калибровачного кадра и кадра потока

        batton_calib = 0;
    }
    namedWindow("calibration", cv::WINDOW_AUTOSIZE);
    cv::destroyWindow("calibration");
    
    if ( (batton_calib == 114) || (batton_calib == 82) )  // Read from file
    {
        cv::FileStorage fs; 
        fs.open("Calibrate_cam_Chess.txt", cv::FileStorage::READ);     // Read from file data calibration
        fs["intrinsic"] >> cameraMatrix;
        fs["distCoeffs"] >> distCoeffs;
        fs["rvecs"] >> rvecs;
        fs["tvecs"] >> tvecs;
        fs.release();
        cout << " --- Calibration data read frome file: Calibrate_cam_Chess.txt" << endl << endl;
    }
    else if ((batton_calib == 48) || (batton_calib == 27))    // use default parameters  
    { 
        cameraMatrix(0, 0) = 600;  // fx
        cameraMatrix(1, 1) = 600;  // fy
        cameraMatrix(0, 2) = 320;  // Cx, half of width frame
        cameraMatrix(1, 2) = 240;  // Cy, half of hight frame
        cameraMatrix(2, 2) = 1;
        for (int i = 0; i < 5; i++) distCoeffs(0, i) = 0.0; // k1 k2 p1 p2 k3
        rvecs.clear();
        tvecs.clear();
        
        cv::FileStorage fs;
        fs.open("Calibrate_cam_Zero.txt", cv::FileStorage::WRITE);    // Write in file data calibration
        fs << "intrinsic" << cameraMatrix;
        fs << "distCoeffs" << distCoeffs;
        fs << "rvecs" << rvecs;
        fs << "tvecs" << tvecs;
        fs.release();
        cout << " --- Default calibration data written into file: Calibrate_cam_Zero.txt" << endl << endl;    
    }
    else if (batton_calib != 27)     // interrupt the calibration cycle
    {
        calibrateCamera( object_points,
                         image_points,
                         calib_frame.size(),
                         cameraMatrix,
                         distCoeffs,
                         rvecs,
                         tvecs,
                         calibrationFlags);
        
        cv::FileStorage fs;
        fs.open("Calibrate_cam_Chess.txt", cv::FileStorage::WRITE);    // Write in file data calibration
        fs << "intrinsic" << cameraMatrix;
        fs << "distCoeffs" << distCoeffs;
        fs << "rvecs" << rvecs;
        fs << "tvecs" << tvecs;
        fs.release();
        cout << " --- Calibration data written into file: Calibrate_cam_Chess.txt" << endl << endl;
    }
}


void CalibrationCamera::calibrCameraChArUco(int _numrCellX,             // 11, default 5;
                                            int _numCellY,              // 8,  default 7;
                                            float _squareLength,        // 10, default 0.04f;
                                            float _markerLength,        // 7,  default 0.02f;
                                            int _dictionaryId,
                                            unsigned int _nFrames)
{
    Mat frame, frame2, frame3;
    Mat frame4 = Mat::zeros(Size(2 * width_frame, height_frame), CV_8UC3);
    Mat calib_frame = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);         // Калибровочный кадр,  frame.size()
    Mat calib_frame_grey = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);
    calibrationFlags = cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3;      // Calibration flags    | CALIB_FIX_K6
    
        // ChArUco board variables
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(_dictionaryId));  // DICT_6X6_250
        // create charuco board object
    cv::Ptr<aruco::CharucoBoard> charucoboard = aruco::CharucoBoard::create( _numrCellX, 
                                                                             _numCellY, 
                                                                             _squareLength, 
                                                                             _markerLength, 
                                                                             dictionary);
    vector< Mat > allCharucoCorners;
    vector< Mat > allCharucoIds;
    vector< Mat > filteredImages;
    allCharucoCorners.reserve(static_cast<unsigned int>(_nFrames));
    allCharucoIds.reserve(static_cast<unsigned int>(_nFrames));
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
        // collect data from each frame
    vector< vector< vector< Point2f > > > allCorners;
    vector< vector< int > > allIds;
    vector< Mat > allImgs;
    
    unsigned int successes = 0;                                         // Счетчик удачных калибровочных кадров
    int batton_calib = 0;
    
    while ( successes < _nFrames )                                // Цикл для определенного числа калибровочных кадров
    {
        batton_calib = cv::waitKey(1);
        if( batton_calib == 27 )                                        // Interrupt the cycle calibration, press "ESC"
        {
            break;
        } 
        else if ( batton_calib == 13 )                                  // Take picture Chessboard, press "Enter"
        {
            if (!CAP.read(calib_frame))                                 // check if we succeeded and store frame into calib_frame
            {
                std::cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            vector< int > ids;
            vector< vector< Point2f > > corners;    
            
                // detect markers
            aruco::detectMarkers( calib_frame, 
                                  dictionary, 
                                  corners, 
                                  ids, 
                                  detectorParams);
            
            if (ids.size() > 0)    // Проверка удачно найденых углов
            {
                    // interpolate charuco corners
                Mat currentCharucoCorners, currentCharucoIds;
                aruco::interpolateCornersCharuco( corners, 
                                                  ids, 
                                                  calib_frame, 
                                                  charucoboard, 
                                                  currentCharucoCorners,
                                                  currentCharucoIds);
                    // draw results
                aruco::drawDetectedMarkers(calib_frame, corners);
                if(currentCharucoCorners.total() > 0) aruco::drawDetectedCornersCharuco(calib_frame, currentCharucoCorners, currentCharucoIds);
                
                cout << "Frame captured" << endl;
                allCorners.push_back(corners);
                allIds.push_back(ids);
                allImgs.push_back(calib_frame);
                
                successes++;
            }
        } 
        else if ( (batton_calib == 114) || (batton_calib == 82) || (batton_calib == 48) || (batton_calib == 27) )
        {       
            // Read from file, press "r" or "R", or use default parameters, press "0", or interrupt the calibration cycle, press "ESC" 
            break;
        }
        CAP.read(frame3);
        putText( frame3, 
                 "Press 'c' to add current frame. 'ESC' to finish and calibrate ChArUco_board",
                 Point(5, 20), 
                 FONT_HERSHEY_SIMPLEX, 
                 0.5, 
                 Scalar(255, 0, 0), 
                 2);
        cv::Rect r1(0, 0, frame3.cols, frame3.rows);                // Создаем фрагменты для склеивания зображения
        cv::Rect r2(frame3.cols, 0, frame3.cols, frame3.rows);
        frame3.copyTo(frame4( r1 ));
        calib_frame.copyTo(frame4( r2 ));
        imshow("calibration", frame4);      // Вывод последнего удачного калибровачного кадра и кадра потока

        batton_calib = 0;
    }
    namedWindow("calibration", cv::WINDOW_AUTOSIZE);
    cv::destroyWindow("calibration");
    
    if ( (batton_calib == 114) || (batton_calib == 82) )  // Read from file
    {
        cv::FileStorage fs; 
        fs.open("Calibrate_cam_ChArUco.txt", cv::FileStorage::READ);     // Read from file data calibration
        fs["intrinsic"] >> cameraMatrix;
        fs["distCoeffs"] >> distCoeffs;
        fs["rvecs"] >> rvecs;
        fs["tvecs"] >> tvecs;
        fs.release();
        cout << " --- Calibration data read frome file: Calibrate_cam_ChArUco.txt" << endl << endl;
    }
    else if ((batton_calib == 48) || (batton_calib == 27))    // use default parameters  
    { 
        cameraMatrix(0, 0) = 600;  // fx
        cameraMatrix(1, 1) = 600;  // fy
        cameraMatrix(0, 2) = 320;  // Cx, half of width frame
        cameraMatrix(1, 2) = 240;  // Cy, half of hight frame
        cameraMatrix(2, 2) = 1;
        for (int i = 0; i < 5; i++) distCoeffs(0, i) = 0.0; // k1 k2 p1 p2 k3
        rvecs.clear();
        tvecs.clear();
        
        cv::FileStorage fs;
        fs.open("Calibrate_cam_Zero.txt", cv::FileStorage::WRITE);    // Write in file data calibration
        fs << "intrinsic" << cameraMatrix;
        fs << "distCoeffs" << distCoeffs;
        fs << "rvecs" << rvecs;
        fs << "tvecs" << tvecs;
        fs.release();
        cout << " --- Default calibration data written into file: Calibrate_cam_Zero.txt" << endl << endl;    
    }
    else if (batton_calib != 27)     // interrupt the calibration cycle
    {
        for(unsigned int i = 0; i < _nFrames; i++) 
        {
                // interpolate using camera parameters
            Mat currentCharucoCorners, currentCharucoIds;
            aruco::interpolateCornersCharuco( allCorners[i], 
                                              allIds[i], 
                                              allImgs[i], 
                                              charucoboard,
                                              currentCharucoCorners, 
                                              currentCharucoIds);
    
            allCharucoCorners.push_back(currentCharucoCorners);
            allCharucoIds.push_back(currentCharucoIds);
        }
        calibrateCameraCharuco( allCharucoCorners,
                                allCharucoIds, 
                                charucoboard, 
                                calib_frame.size(),
                                cameraMatrix,
                                distCoeffs,
                                rvecs,
                                tvecs,
                                calibrationFlags);
        
        cv::FileStorage fs;
        fs.open("Calibrate_cam_ChArUco.txt", cv::FileStorage::WRITE);    // Write in file data calibration
        fs << "intrinsic" << cameraMatrix;
        fs << "distCoeffs" << distCoeffs;
        fs << "rvecs" << rvecs;
        fs << "tvecs" << tvecs;
        fs.release();
        cout << " --- Calibration data written into file: Calibrate_cam_ChArUco.txt" << endl << endl;
    }
}

