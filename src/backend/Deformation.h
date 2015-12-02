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


#ifndef DEFORMATION_H_
#define DEFORMATION_H_

#include "../frontend/Volume.h"
#include "../utils/ThreadObject.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <vector>
#include <sstream>
#include <boost/thread.hpp>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <boost/thread/condition_variable.hpp>
#include <pcl/filters/voxel_grid.h>

#include "DeformationGraph.h"
#include "iSAMInterface.h"

class Deformation : public ThreadObject
{
    public:
        Deformation();
        virtual ~Deformation();

        void reset();

        ThreadMutexObject<bool> hasLooped;

        void saveCloud();

        void saveMesh();

    private:
        bool inline process();

        void addCameraCamera();
        void addCameraLoop();
        void addVertices();

        iSAMInterface * iSAM;

        int latestDensePoseIdCopy;
        int latestPoseIdCopy;
        int latestLoopIdCopy;
        int latestProcessedDensePose;
        int latestProcessedPose;
        int latestProcessedLoop;

        DeformationGraph * deformationGraph;

        std::vector<uint64_t> vertexTimes;

        std::vector<std::pair<uint64_t, PointXYZRGBNormal> > poseGraphPoints;
        std::vector<uint64_t> graphPoseTimes;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphPosePoints;

        bool shouldTime;
};

#endif /* DEFORMATION_H_ */
