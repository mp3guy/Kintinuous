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

#ifndef PLACERECOGNITION_H_
#define PLACERECOGNITION_H_

#include "../frontend/Volume.h"
#include "../utils/ThreadObject.h"
#include "Surf3DTools.h"
#include "DBowInterfaceSurf.h"
#include "PNPSolver.h"

#include <sstream>
#include <string>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/filters/voxel_grid.h>
#include "DepthCamera.h"

class PlaceRecognition : public ThreadObject
{
    public:
        PlaceRecognition(cv::Mat * depthIntrinsics);

        virtual ~PlaceRecognition();

        void reset();

    private:
        bool inline process();

        void processLoopClosureDetection(int matchId);

        Eigen::Matrix4f icpDepthFrames(Eigen::Matrix4f & bootstrap, unsigned short * frame1, unsigned short * frame2, float & score);

        int latestProcessedFrame;
        DBowInterfaceSurf * dbowInterface;
        cv::Mat * oldImage;
        cv::Mat * newImage;
        cv::Mat * imageGray;
        cv::Mat * depthMapNew;
        cv::Mat * depthMapOld;
        DepthCamera * depthCamera;
        int latestFrameIdCopy;
};

#endif /* PLACERECOGNITION_H_ */
