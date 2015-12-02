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

#include "MeshGenerator.h"

MeshGenerator::MeshGenerator()
 : ThreadObject("MeshGeneratorThread")
{
    reset();
}

MeshGenerator::~MeshGenerator()
{

}

void MeshGenerator::reset()
{
	calcFinished.assignAndNotifyAll(false);
}

void MeshGenerator::save()
{
    assert(threadPack.finalised.getValue() && threadPack.cloudSlices.size() > 1);

    pcl::PointCloud<pcl::PointXYZRGBNormal> * fullCloud = new pcl::PointCloud<pcl::PointXYZRGBNormal>;
    pcl::PointCloud<pcl::PointXYZRGBNormal> * fullCloudF = new pcl::PointCloud<pcl::PointXYZRGBNormal>;

    int latestPoseIdCopy = threadPack.latestPoseId.getValue();

    fullCloud->height = threadPack.cloudSlices.at(1)->processedCloud->height;
    fullCloud->width= threadPack.cloudSlices.at(1)->processedCloud->width;

    for(int i = 1; i < latestPoseIdCopy; i++)
    {
    	fullCloud->insert(fullCloud->end(), threadPack.cloudSlices.at(i)->processedCloud->begin(), threadPack.cloudSlices.at(i)->processedCloud->end());
    }

    if(ConfigArgs::get().extractOverlap && !ConfigArgs::get().saveOverlap)
    {
        pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor;
        sor.setInputCloud (fullCloud->makeShared());

        const float3 & voxelSizeMeters = Volume::get().getVoxelSizeMeters();

        float leafSize = std::max(voxelSizeMeters.x,
                         std::max(voxelSizeMeters.y,
                                  voxelSizeMeters.z));

        sor.setLeafSize(leafSize, leafSize, leafSize);

        sor.filter(*fullCloudF);

        delete fullCloud;

        fullCloud = fullCloudF;
    }
    else
    {
        delete fullCloudF;
    }
    
    std::cout << "Calculating mesh... ";
    std::cout.flush();

    pcl::PolygonMesh * bigMesh = 0;

    if(ConfigArgs::get().extractOverlap && ConfigArgs::get().saveOverlap)
    {
        std::cout << "(merging for export)... ";
        std::cout.flush();

    	assert(threadPack.triangles.size() > 0);

    	int totalTriangles = 0;

    	for(unsigned int i = 0; i < threadPack.triangles.size(); i++)
    	{
    		totalTriangles += threadPack.triangles.at(i)->polygons.size();
    	}

    	bigMesh = new pcl::PolygonMesh();
    	*bigMesh = *threadPack.triangles.at(0);

    	pcl::PointCloud<pcl::PointXYZRGBNormal> destPoints;
        pcl::PointCloud<pcl::PointXYZRGBNormal> srcPoints;

        pcl::fromPCLPointCloud2(bigMesh->cloud, destPoints);

        int offset = destPoints.size();

    	for(unsigned int i = 1; i < threadPack.triangles.size(); i++)
    	{
    		pcl::fromPCLPointCloud2(threadPack.triangles.at(i)->cloud, srcPoints);

    		destPoints.insert(destPoints.end(), srcPoints.begin(), srcPoints.end());

    		int nextIndex = bigMesh->polygons.size();

    		bigMesh->polygons.insert(bigMesh->polygons.end(),
    								 threadPack.triangles.at(i)->polygons.begin(),
    								 threadPack.triangles.at(i)->polygons.end());

    		for(unsigned int j = nextIndex; j < bigMesh->polygons.size(); j++)
    		{
    			for(unsigned int k = 0; k < bigMesh->polygons.at(j).vertices.size(); k++)
    			{
    				bigMesh->polygons.at(j).vertices.at(k) += offset;
    			}
    		}

    		offset = destPoints.size();

            std::cout << "\rCalculating mesh... (merging for export)... " << (int)(((float)bigMesh->polygons.size() / (float)totalTriangles) * 100.0f) << "%";
            std::cout.flush();
    	}

    	pcl::toPCLPointCloud2(destPoints, bigMesh->cloud);
    }
    else
    {
        std::cout << "(from scratch)... ";
        std::cout.flush();

        bigMesh = new pcl::PolygonMesh();

        boost::thread * calcThread = new boost::thread(boost::bind(&MeshGenerator::calculateMesh, this, fullCloud, bigMesh));

        int ticker = 1;

        while(!calcFinished.getValue())
        {
        	if(ticker % 2)
        	{
        		std::cout << "\rCalculating mesh... (from scratch)... .  ";
        		std::cout.flush();
        	}
			else if(ticker % 3)
		    {
				std::cout << "\rCalculating mesh... (from scratch)... .. ";
				std::cout.flush();
		    }
			else if(ticker % 4)
			{
				std::cout << "\rCalculating mesh... (from scratch)... ...";
				std::cout.flush();
			}
			else
			{
				std::cout << "\rCalculating mesh... (from scratch)...    ";
				std::cout.flush();
			}
        	ticker++;
        	usleep(250000);
        }

        std::cout << "\r                                         ";
        std::cout << "\rCalculating mesh... (from scratch)...";

        calcThread->join();

        delete calcThread;
    }

    std::cout << " calculated\nSaving " << bigMesh->polygons.size() << " triangles... ";
    std::cout.flush();

    std::string filePLY = ConfigArgs::get().saveFile;
    filePLY.append(".ply");
    pcl::io::savePLYFile(filePLY, *bigMesh, 5);

    delete bigMesh;
    delete fullCloud;

    std::cout << "PLY saved" << std::endl;
}

void MeshGenerator::calculateMesh(pcl::PointCloud<pcl::PointXYZRGBNormal> * cloud_with_normals, pcl::PolygonMesh *& target)
{
    calcFinished.assignAndNotifyAll(false);
    // Create search tree*
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    tree2->setInputCloud (cloud_with_normals->makeShared());

    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;

    const float3 & voxelSizeMeters = Volume::get().getVoxelSizeMeters();

    float leafSize = std::max(voxelSizeMeters.x,
                     std::max(voxelSizeMeters.y,
                              voxelSizeMeters.z));

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius(leafSize * 2.0);

    // Set typical values for the parameters
    gp3.setMu(2);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 2); // 45 degrees
    gp3.setMinimumAngle(M_PI / 36); // 10 degrees
    gp3.setMaximumAngle(2.5 * M_PI / 3); // 120 degrees
    gp3.setNormalConsistency(false);

    // Get result
    gp3.setInputCloud (cloud_with_normals->makeShared());
    gp3.setSearchMethod (tree2);

    gp3.reconstruct (*target);

    calcFinished.assignAndNotifyAll(true);
}

bool inline MeshGenerator::process()
{
    latestPoseIdCopy = threadPack.latestPoseId.getValueWait();

    int latestMeshIdCopy = threadPack.latestMeshId.getValue();

    while(latestMeshIdCopy < latestPoseIdCopy)
    {
        threadPack.cloudSlices.at(latestMeshIdCopy)->processedCloud->width = threadPack.cloudSlices.at(latestMeshIdCopy)->processedCloud->size();
        threadPack.cloudSlices.at(latestMeshIdCopy)->processedCloud->height = 1;

        pcl::PolygonMesh * newMesh = new pcl::PolygonMesh();

        if(threadPack.cloudSlices.at(latestMeshIdCopy)->processedCloud->size() > 0)
        {
            TICK(threadIdentifier);
            calculateMesh(threadPack.cloudSlices.at(latestMeshIdCopy)->processedCloud, newMesh);
        }

        if(threadPack.cloudSlices.at(latestMeshIdCopy)->processedCloud->size() > 0)
        {
            TOCK(threadIdentifier);
        }

        threadPack.triangles.push_back(newMesh);

        threadPack.latestMeshId++;
        latestMeshIdCopy++;
    }
    
    if(latestMeshIdCopy)
    {
        lagTime.assignValue(Stopwatch::getCurrentSystemTime() - threadPack.cloudSlices.at(latestMeshIdCopy - 1)->lagTime);
    }

    if(threadPack.cloudSliceProcessorFinished.getValue())
    {
        latestPoseIdCopy = threadPack.latestPoseId.getValue();
        int latestMeshIdCopy = threadPack.latestMeshId.getValue();

        if(latestMeshIdCopy < latestPoseIdCopy)
        {
            return true;
        }

        threadPack.meshGeneratorFinished.assignAndNotifyAll(true);
        lagTime.assignValue(0);
        return false;
    }

    return true;
}
