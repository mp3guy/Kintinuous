/*
 * This file is part of Kintinuous.
 *
 * Copyright (C) 2015 The National University of Ireland Maynooth and 
 * Massachusetts Institute of Technology
 *
 * The use of the code within this file and all code within files that 
 * make up the software that is Kintinuous is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.cs.nuim.ie/research/vision/data/kintinuous/code.php> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email commercialisation@nuim.ie.
 */

#ifndef MAINCONTROLLER_H_
#define MAINCONTROLLER_H_

#include <iostream>
#include <fstream>

#include <GL/glew.h>
#include <zlib.h>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>

#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "utils/Stopwatch.h"
#include "utils/RawLogReader.h"
#include "utils/LiveLogReader.h"
#include "utils/ThreadDataPack.h"
#include "frontend/cuda/internal.h"
#include "backend/MeshGenerator.h"
#include "backend/PlaceRecognition.h"
#include "backend/CloudSliceProcessor.h"
#include "backend/Deformation.h"
#include "backend/TrackerInterface.h"
#include "utils/ConfigArgs.h"
#include "frontend/Resolution.h"
#include "PangoVis.h"

class MainController
{
    public:
        MainController(int argc, char * argv[]);
        virtual ~MainController();

        int start();

        static MainController * controller;

        //Proxy functions for the GUI
        void complete();

        void save();

        void reset();

        void setPark(const bool park);

        void shutdown();

        uint64_t getMaxLag();

    private:
        bool setup();
        int mainLoop();

        void loadCalibration();

        cv::Mat * depthIntrinsics;
        PangoVis * pangoVis;
        TrackerInterface * trackerInterface;
        MeshGenerator * meshGenerator;
        PlaceRecognition * placeRecognition;
        CloudSliceProcessor * cloudSliceProcessor;
        Deformation * deformation;

        RawLogReader * rawRead;
        LiveLogReader * liveRead;
        LogReader * logRead;

        boost::thread_group threads;
        std::vector<ThreadObject *> systemComponents;
};

#endif /* MAINCONTROLLER_H_ */
