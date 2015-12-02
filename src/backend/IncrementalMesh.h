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

#ifndef INCREMENTALMESH_H_
#define INCREMENTALMESH_H_

#include "../frontend/Volume.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/voxel_grid.h>

#include "UGP3.h"

class IncrementalMesh
{
    public:
        IncrementalMesh();
        virtual ~IncrementalMesh();

        void saveMesh(std::string file);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr computeIncrementalMesh(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals,
                                                                            std::vector<uint64_t> & pointTimes);

        pcl::PolygonMesh::Ptr mesh;

        void updateInternalState(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src);

    private:
        UpdatableGPT * updatableGPT;
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr treeGPT;
};

#endif /* INCREMENTALMESH_H_ */
