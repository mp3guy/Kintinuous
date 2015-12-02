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

#include "IncrementalMesh.h"

IncrementalMesh::IncrementalMesh()
 : mesh(pcl::PolygonMesh::Ptr(new pcl::PolygonMesh)),
   updatableGPT(new UpdatableGPT),
   treeGPT(pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr(new pcl::search::KdTree<pcl::PointXYZRGBNormal>))
{
    const float3 & voxelSizeMeters = Volume::get().getVoxelSizeMeters();

    float leafSize = std::max(voxelSizeMeters.x,
                     std::max(voxelSizeMeters.y,
                              voxelSizeMeters.z));

    // Set the maximum distance between connected points (maximum edge length)
    updatableGPT->setSearchRadius(leafSize * 4.0);

    // Set typical values for the parameters
    updatableGPT->setMu(2);
    updatableGPT->setMaximumNearestNeighbors(100);
    updatableGPT->setMaximumSurfaceAngle(M_PI / 2); // 45 degrees
    updatableGPT->setMinimumAngle(M_PI / 36); // 10 degrees
    updatableGPT->setMaximumAngle(2.5 * M_PI / 3); // 120 degrees
    updatableGPT->setNormalConsistency(false);
}

IncrementalMesh::~IncrementalMesh()
{
    delete updatableGPT;
}

void IncrementalMesh::saveMesh(std::string file)
{
    std::string filePLY = file;
    filePLY.append(".ply");
    pcl::io::savePLYFile(filePLY, *mesh, 5);
}

void IncrementalMesh::updateInternalState(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src)
{
    updatableGPT->updatePoints(src, *mesh);
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr IncrementalMesh::computeIncrementalMesh(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals,
                                                                                     std::vector<uint64_t> & pointTimes)
{
    if(mesh->polygons.size() == 0)
    {
        //First run, so initialise
        treeGPT->setInputCloud(cloud_with_normals->makeShared());
        updatableGPT->setSearchMethod(treeGPT);
        updatableGPT->setInputCloud(cloud_with_normals->makeShared());
        updatableGPT->reconstruct(*mesh);

        return cloud_with_normals;
    }
    else
    {
        Eigen::AlignedBox3f bv;

        pcl::PointXYZRGBNormal min_pt, max_pt;
        pcl::getMinMax3D(*cloud_with_normals, min_pt, max_pt);

        bv.extend(min_pt.getVector3fMap());
        bv.extend(max_pt.getVector3fMap());

        //Remove potential collision points
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr boundary = updatableGPT->getAndSetBoundaryFringe(bv);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cleanCloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        if(boundary->size() == 0)
        {
            cleanCloud = cloud_with_normals;
        }
        else
        {
            pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr boundaryTree (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);

            boundaryTree->setInputCloud(boundary);

            std::vector<int> pointIdxRadiusSearch(1);
            std::vector<float> pointRadiusSquaredDistance(1);

            std::vector<uint64_t> newTimes;

            for(unsigned int i = 0; i < cloud_with_normals->size(); i++)
            {
                if(boundaryTree->radiusSearch(cloud_with_normals->at(i), updatableGPT->getSearchRadius() / 4.0, pointIdxRadiusSearch, pointRadiusSquaredDistance) == 0)
                {
                    cleanCloud->push_back(cloud_with_normals->at(i));
                    newTimes.push_back(pointTimes.at(i));
                }
            }

            pointTimes = newTimes;
        }

        //Increment
        updatableGPT->updateMesh(cleanCloud, *mesh);

        return cleanCloud;
    }
}
