QT -= gui

CONFIG += c++14 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

#VLIBS_DIR = $$PWD/vlibs
#include( $$VLIBS_DIR/vopencv/vopencv.pri)
#DISTFILES +=

CONFIG      *= link_pkgconfig
PKGCONFIG   *= opencv4
PKGCONFIG   *= metslib

PKGCONFIG   += pcl_2d-1.9
PKGCONFIG   += pcl_common-1.9
PKGCONFIG   += pcl_features-1.9
PKGCONFIG   += pcl_filters-1.9
PKGCONFIG   += pcl_geometry-1.9
PKGCONFIG   += pcl_io-1.9
PKGCONFIG   += pcl_kdtree-1.9
PKGCONFIG   += pcl_keypoints-1.9
PKGCONFIG   += pcl_ml-1.9
PKGCONFIG   += pcl_octree-1.9
PKGCONFIG   += pcl_outofcore-1.9
PKGCONFIG   += pcl_people-1.9
PKGCONFIG   += pcl_recognition-1.9
PKGCONFIG   += pcl_registration-1.9
PKGCONFIG   += pcl_sample_consensus-1.9
PKGCONFIG   += pcl_search-1.9
PKGCONFIG   += pcl_segmentation-1.9
PKGCONFIG   += pcl_stereo-1.9
PKGCONFIG   += pcl_surface-1.9
PKGCONFIG   += pcl_tracking-1.9
PKGCONFIG   += pcl_visualization-1.9

#PKGCONFIG   += eigen3
#LIBS        += -L/usr/local/lib
#INCLUDEPATH += /usr/local/include/pcl-1.9/pcl
INCLUDEPATH += /usr/local/include/vtk-8.0

#LIBS += -L/usr/local/lib -Wl,-rpath=/usr/local/lib
LIBS += -lboost_system \
        #-lvtksys-8.0 \
        -lvtkCommonCore-8.0


#INCLUDEPATH += /usr/local/include/opencv4
#LIBS += -L/usr/local/lib

#LIBS += -lopencv_core \
#        -lopencv_imgproc \
#        -lopencv_imgcodecs \
#        -lopencv_highgui \
#        -lopencv_objdetect \
#        -lopencv_features2d \
#        -lopencv_xfeatures2d \
#        -lopencv_videoio \
#        -lopencv_tracking \
#        -lopencv_calib3d \
#        -lopencv_sfm

HEADERS +=
