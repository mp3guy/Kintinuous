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

#include "CloudSliceProcessor.h"

CloudSliceProcessor::CloudSliceProcessor()
 : ThreadObject("CloudSliceProcessorThread")
{
    reset();
}

CloudSliceProcessor::~CloudSliceProcessor()
{

}

void CloudSliceProcessor::reset()
{
    latestPushedCloud = 0;
	cycledMutex = false;
}

bool inline CloudSliceProcessor::process()
{
    boost::mutex::scoped_lock lock(threadPack.tracker->cloudMutex);

	threadPack.tracker->cloudSignal.wait(threadPack.tracker->cloudMutex);

    std::vector<CloudSlice *> * trackerSlices = &threadPack.tracker->getCloudSlices();

	numClouds = trackerSlices->size();

	cycledMutex = threadPack.tracker->cycledMutex;

	if(cycledMutex)
	{
	    threadPack.tracker->cycledMutex = false;
	}

	if(threadPack.cloudSlices.size() == 0)
	{
	    uint64_t initTime = threadPack.tracker->init_utime.getValue();

	    if(initTime == std::numeric_limits<unsigned long long>::max())
	    {
	        return true;
	    }

		Eigen::Matrix<float, 3, 3, Eigen::RowMajor> lastRotation = threadPack.tracker->getLastRotation();
		Eigen::Vector3f lastTranslation = threadPack.tracker->getLastTranslation();

		threadPack.cloudSlices.push_back(new CloudSlice(new pcl::PointCloud<pcl::PointXYZRGB>(),
                                                        CloudSlice::FIRST,
                                                        CloudSlice::FAIL,
                                                        lastTranslation,
                                                        lastRotation,
                                                        initTime,
                                                        Stopwatch::getCurrentSystemTime(),
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        &threadPack.tracker->placeRecognitionBuffer[0]));

		threadPack.cloudSlices.back()->processedCloud = new pcl::PointCloud<pcl::PointXYZRGBNormal>();

		threadPack.latestPoseId.assignAndNotifyAll(threadPack.cloudSlices.size());
	}

	lock.unlock();

	if(cycledMutex)
    {
        while(latestPushedCloud < numClouds)
        {
            trackerSlices->at(latestPushedCloud)->processedCloud = new pcl::PointCloud<pcl::PointXYZRGBNormal>();

            if(trackerSlices->at(latestPushedCloud)->cloud->size() > 0)
            {
                TICK(threadIdentifier);

                if(ConfigArgs::get().weightCull > 0)
                {
                    pcl::PointCloud<pcl::PointXYZRGB> * tempCloud = new pcl::PointCloud<pcl::PointXYZRGB>();

                    for(unsigned int i = 0; i < trackerSlices->at(latestPushedCloud)->cloud->size(); i++)
                    {
                        if(trackerSlices->at(latestPushedCloud)->cloud->at(i).a >= ConfigArgs::get().weightCull)
                        {
                            tempCloud->push_back(trackerSlices->at(latestPushedCloud)->cloud->at(i));
                        }
                    }

                    trackerSlices->at(latestPushedCloud)->cloud->clear();

                    trackerSlices->at(latestPushedCloud)->cloud->insert(trackerSlices->at(latestPushedCloud)->cloud->end(), tempCloud->begin(), tempCloud->end());

                    delete tempCloud;
                }

                //This isn't a double check, after culling weights the cloud might be empty :(
                if(trackerSlices->at(latestPushedCloud)->cloud->size() > 0)
                {
                    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
                    sor.setInputCloud(trackerSlices->at(latestPushedCloud)->cloud->makeShared());

                    const float3 & voxelSizeMeters = Volume::get().getVoxelSizeMeters();

                    float leafSize = std::max(voxelSizeMeters.x,
                                     std::max(voxelSizeMeters.y,
                                              voxelSizeMeters.z));

                    sor.setLeafSize(leafSize, leafSize, leafSize);

                    pcl::PointCloud<pcl::PointXYZRGB> * tempCloud = new pcl::PointCloud<pcl::PointXYZRGB>();

                    sor.filter(*tempCloud);

                    trackerSlices->at(latestPushedCloud)->cloud->clear();

                    trackerSlices->at(latestPushedCloud)->cloud->insert(trackerSlices->at(latestPushedCloud)->cloud->end(), tempCloud->begin(), tempCloud->end());

                    delete tempCloud;

                    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
                    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
                    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);

                    tree->setInputCloud(trackerSlices->at(latestPushedCloud)->cloud->makeShared());

                    n.setInputCloud(trackerSlices->at(latestPushedCloud)->cloud->makeShared());
                    n.setSearchMethod(tree);
                    n.setKSearch(20);
                    n.compute(*normals);

                    pcl::concatenateFields(*trackerSlices->at(latestPushedCloud)->cloud, *normals, *trackerSlices->at(latestPushedCloud)->processedCloud);

                    TOCK(threadIdentifier);
                }
            }

            threadPack.cloudSlices.push_back(trackerSlices->at(latestPushedCloud));

            threadPack.latestPoseId.assignAndNotifyAll(threadPack.cloudSlices.size());

            latestPushedCloud++;
        }
    }

	if(latestPushedCloud)
	{
        lagTime.assignValue(Stopwatch::getCurrentSystemTime() - trackerSlices->at(latestPushedCloud - 1)->lagTime);
	}

    if(threadPack.cloudSlices.size() && threadPack.cloudSlices.back()->dimension == CloudSlice::FINAL)
    {
        threadPack.cloudSliceProcessorFinished.assignAndNotifyAll(true);
        lagTime.assignValue(0);
        return false;
    }

    return true;
}

void CloudSliceProcessor::save()
{
    assert(threadPack.finalised.getValue() && threadPack.cloudSlices.size() > 1);

    pcl::PointCloud<pcl::PointXYZRGBNormal> * fullCloud = new pcl::PointCloud<pcl::PointXYZRGBNormal>;

    int latestPoseIdCopy = threadPack.latestPoseId.getValue();

    fullCloud->height = threadPack.cloudSlices.at(1)->processedCloud->height;
    fullCloud->width = threadPack.cloudSlices.at(1)->processedCloud->width;

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

        pcl::PointCloud<pcl::PointXYZRGBNormal> * tempCloud = new pcl::PointCloud<pcl::PointXYZRGBNormal>();

        sor.filter(*tempCloud);

        fullCloud->clear();

        fullCloud->insert(fullCloud->end(), tempCloud->begin(), tempCloud->end());

        delete tempCloud;
    }

    std::cout << "Saving " << fullCloud->size() << " points... ";
    std::cout.flush();

    std::string filePCD = ConfigArgs::get().saveFile;
    filePCD.append(".pcd");
    pcl::io::savePCDFile(filePCD, *fullCloud, true);

    std::cout << "PCD saved" << std::endl;

    delete fullCloud;
    return;
}
