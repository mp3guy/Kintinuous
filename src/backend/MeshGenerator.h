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

#ifndef MESHGENERATOR_H_
#define MESHGENERATOR_H_

#include "../utils/ThreadObject.h"
#include "../frontend/Volume.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/voxel_grid.h>

class MeshGenerator : public ThreadObject
{
    public:
        MeshGenerator();

        virtual ~MeshGenerator();

        void reset();

        void save();

    private:
        bool inline process();
        void calculateMesh(pcl::PointCloud<pcl::PointXYZRGBNormal> * cloud_with_normals, pcl::PolygonMesh *& target);

        int latestPoseIdCopy;
        ThreadMutexObject<bool> calcFinished;
};

#endif /* MESHGENERATOR_H_ */
