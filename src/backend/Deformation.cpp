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

#include "Deformation.h"

Deformation::Deformation()
 : ThreadObject("DeformationThread"),
   hasLooped(false),
   iSAM(0),
   deformationGraph(0),
   graphPosePoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>)
{
    reset();
}

Deformation::~Deformation()
{
    std::string filename = ConfigArgs::get().saveFile;
    filename.append(".poses");

    std::ofstream file;
    file.open(filename.c_str(), std::fstream::out);

    std::vector<std::pair<uint64_t, Eigen::Matrix4f> > newPoseGraph;
    iSAM->getCameraPoses(newPoseGraph);

    for(unsigned int i = 0; i < newPoseGraph.size(); i++)
    {
        file << std::setprecision(6) << std::fixed << (double)newPoseGraph.at(i).first / 1000000.0 << " ";

        Eigen::Vector3f trans = newPoseGraph.at(i).second.topRightCorner(3, 1);
        Eigen::Matrix3f rot = newPoseGraph.at(i).second.topLeftCorner(3, 3);

        file << trans(0) << " " << trans(1) << " " << trans(2) << " ";

        Eigen::Quaternionf currentCameraRotation(rot);

        file << currentCameraRotation.x() << " " << currentCameraRotation.y() << " " << currentCameraRotation.z() << " " << currentCameraRotation.w() << "\n";
    }

    file.close();

    delete iSAM;

    if(deformationGraph)
    {
        delete deformationGraph;
    }
    deformationGraph = 0;
}

void Deformation::saveCloud()
{
    std::cout << "Saving " << threadPack.pointPool->size() << " points... ";
    std::cout.flush();

    pcl::PCDWriter writer;
    std::stringstream strs;
    strs << ConfigArgs::get().saveFile << "_opt.pcd";

    if(threadPack.pointPool->size() > 0)
    {
        writer.write(strs.str(), *threadPack.pointPool, true);
    }

    std::cout << "PCD saved" << std::endl;

}

void Deformation::saveMesh()
{
    if(threadPack.incrementalMesh)
    {
        std::cout << "Saving " << threadPack.incrementalMesh->mesh->polygons.size() << " triangles... ";
        std::cout.flush();

        assert(threadPack.incrementalMesh->mesh->cloud.width == threadPack.pointPool->size());

        std::stringstream strs;
        strs << ConfigArgs::get().saveFile << "_opt";

        threadPack.incrementalMesh->saveMesh(strs.str());

        std::cout << "PLY saved" << std::endl;
    }
}

void Deformation::reset()
{
    if(iSAM)
    {
        delete iSAM;
    }
    iSAM = new iSAMInterface;

    if(deformationGraph)
    {
        delete deformationGraph;
    }
    deformationGraph = 0;

    latestDensePoseIdCopy = 0;
    latestPoseIdCopy = 0;
    latestLoopIdCopy = 0;
    latestProcessedLoop = 0;
    latestProcessedDensePose = 0;
    latestProcessedPose = 0;

    vertexTimes.clear();

    graphPoseTimes.clear();
    graphPosePoints->clear();
}

void Deformation::addCameraCamera()
{
    static KintinuousTracker::DensePose lastPose;
    //Add camera poses into iSAM
    for(int i = latestProcessedDensePose; i < latestDensePoseIdCopy; i++)
    {
        if(i == 0)
        {
            lastPose = threadPack.tracker->densePoseGraph.at(i);
        }
        else
        {
            if(!ConfigArgs::get().fastLoops ||
               (threadPack.tracker->densePoseGraph.at(i).pose.topRightCorner(3, 1) - lastPose.pose.topRightCorner(3, 1)).norm() > ConfigArgs::get().denseSamplingRate ||
                threadPack.tracker->densePoseGraph.at(i).isLoopPose)
            {
                iSAM->addCameraCameraConstraint(lastPose.timestamp,
                                                threadPack.tracker->densePoseGraph.at(i).timestamp,
                                                lastPose.pose.topLeftCorner(3, 3),
                                                lastPose.pose.topRightCorner(3, 1),
                                                threadPack.tracker->densePoseGraph.at(i).pose.topLeftCorner(3, 3),
                                                threadPack.tracker->densePoseGraph.at(i).pose.topRightCorner(3, 1));

                lastPose = threadPack.tracker->densePoseGraph.at(i);
            }
        }
    }

    if(threadPack.finalised.getValue() &&
       threadPack.cloudSlices.back()->dimension == CloudSlice::FINAL &&
       threadPack.cloudSlices.back()->cloud->size() > 0)
    {
        iSAM->addCameraCameraConstraint(lastPose.timestamp,
                                        threadPack.tracker->densePoseGraph.back().timestamp,
                                        lastPose.pose.topLeftCorner(3, 3),
                                        lastPose.pose.topRightCorner(3, 1),
                                        threadPack.tracker->densePoseGraph.back().pose.topLeftCorner(3, 3),
                                        threadPack.tracker->densePoseGraph.back().pose.topRightCorner(3, 1));

        lastPose = threadPack.tracker->densePoseGraph.back();
    }
}

void Deformation::addCameraLoop()
{
    bool loopAdded = false;

    for(int i = latestProcessedLoop; i < latestLoopIdCopy; i++)
    {
        std::vector<std::pair<uint64_t, Eigen::Vector3d> > prevPoseGraph;

        iSAM->getCameraPositions(prevPoseGraph);

        pcl::PointXYZRGBNormal newPoint;
        int originalPointPool = threadPack.pointPool->size();

        Eigen::Vector4d point(0, 0, 0, 1);
        Eigen::Vector4d transPoint(0, 0, 0, 1);
        Eigen::Vector3d transPoint3d(0, 0, 0);

        boost::mutex::scoped_lock lock(threadPack.poolMutex);

        for(int i = 0; i < latestLoopIdCopy; i++)
        {
            Eigen::Matrix4d camPose = iSAM->getCameraPose(threadPack.loopClosureConstraints.at(i)->time1).cast<double>();

            for(unsigned int j = 0; j < threadPack.loopClosureConstraints.at(i)->inliers1Proj.size(); j++)
            {
                point(0) = threadPack.loopClosureConstraints.at(i)->inliers1Proj.at(j)(0);
                point(1) = threadPack.loopClosureConstraints.at(i)->inliers1Proj.at(j)(1);
                point(2) = threadPack.loopClosureConstraints.at(i)->inliers1Proj.at(j)(2);

                transPoint = camPose * point;

                newPoint.x = transPoint(0);
                newPoint.y = transPoint(1);
                newPoint.z = transPoint(2);

                threadPack.pointPool->push_back(newPoint);

                vertexTimes.push_back(threadPack.loopClosureConstraints.at(i)->time1);
            }

            camPose = iSAM->getCameraPose(threadPack.loopClosureConstraints.at(i)->time2).cast<double>();

            for(unsigned int j = 0; j < threadPack.loopClosureConstraints.at(i)->inliers2Proj.size(); j++)
            {
                point(0) = threadPack.loopClosureConstraints.at(i)->inliers2Proj.at(j)(0);
                point(1) = threadPack.loopClosureConstraints.at(i)->inliers2Proj.at(j)(1);
                point(2) = threadPack.loopClosureConstraints.at(i)->inliers2Proj.at(j)(2);

                transPoint = camPose * point;

                newPoint.x = transPoint(0);
                newPoint.y = transPoint(1);
                newPoint.z = transPoint(2);

                threadPack.pointPool->push_back(newPoint);

                vertexTimes.push_back(threadPack.loopClosureConstraints.at(i)->time2);
            }
        }

        Eigen::Matrix4f prevPose = iSAM->getCameraPose(prevPoseGraph.back().first);

        std::map<uint64_t, int> cameraPoseMap;

        for(unsigned int i = 0; i < prevPoseGraph.size(); i++)
        {
            newPoint.x = prevPoseGraph.at(i).second(0);
            newPoint.y = prevPoseGraph.at(i).second(1);
            newPoint.z = prevPoseGraph.at(i).second(2);

            threadPack.pointPool->push_back(newPoint);

            vertexTimes.push_back(prevPoseGraph.at(i).first);

            cameraPoseMap[prevPoseGraph.at(i).first] = threadPack.pointPool->size() - 1;
        }

        isam::Pose3d_Pose3d_Factor * factor = iSAM->addLoopConstraint(threadPack.loopClosureConstraints.at(i)->time1,
                                                                      threadPack.loopClosureConstraints.at(i)->time2,
                                                                      threadPack.loopClosureConstraints.at(i)->constraint);

        double residual = iSAM->optimise();

        loopAdded = residual < ConfigArgs::get().isamThresh;

        if(loopAdded)
        {
            threadPack.lastLoopTime.assignValue(Stopwatch::getCurrentSystemTime());

            shouldTime = true;

            assert(deformationGraph);

            deformationGraph->clearConstraints();

            deformationGraph->appendVertices(&vertexTimes, originalPointPool);

            std::vector<std::pair<uint64_t, Eigen::Vector3d> > newPoseGraph;

            iSAM->getCameraPositions(newPoseGraph);

            for(unsigned int i = 0; i < newPoseGraph.size(); i++)
            {
                deformationGraph->addConstraint(cameraPoseMap[newPoseGraph.at(i).first], newPoseGraph.at(i).second);
            }

            Eigen::Matrix4f nextPose = iSAM->getCameraPose(prevPoseGraph.back().first);

            threadPack.loopOffset.assignValue(threadPack.loopOffset.getValue() * (nextPose * prevPose.inverse()));

            int count = 0;
            for(int i = 0; i < latestLoopIdCopy; i++)
            {
                Eigen::Matrix4d camPose = iSAM->getCameraPose(threadPack.loopClosureConstraints.at(i)->time1).cast<double>();

                for(unsigned int j = 0; j < threadPack.loopClosureConstraints.at(i)->inliers1Proj.size(); j++)
                {
                    point(0) = threadPack.loopClosureConstraints.at(i)->inliers1Proj.at(j)(0);
                    point(1) = threadPack.loopClosureConstraints.at(i)->inliers1Proj.at(j)(1);
                    point(2) = threadPack.loopClosureConstraints.at(i)->inliers1Proj.at(j)(2);

                    transPoint = camPose * point;

                    transPoint3d = transPoint.head(3);

                    deformationGraph->addConstraint(originalPointPool + count++, transPoint3d);
                }

                camPose = iSAM->getCameraPose(threadPack.loopClosureConstraints.at(i)->time2).cast<double>();

                for(unsigned int j = 0; j < threadPack.loopClosureConstraints.at(i)->inliers2Proj.size(); j++)
                {
                    point(0) = threadPack.loopClosureConstraints.at(i)->inliers2Proj.at(j)(0);
                    point(1) = threadPack.loopClosureConstraints.at(i)->inliers2Proj.at(j)(1);
                    point(2) = threadPack.loopClosureConstraints.at(i)->inliers2Proj.at(j)(2);

                    transPoint = camPose * point;

                    transPoint3d = transPoint.head(3);

                    deformationGraph->addConstraint(originalPointPool + count++, transPoint3d);
                }
            }

            deformationGraph->optimiseGraphSparse();
            deformationGraph->applyGraphToVertices(8);

            vertexTimes.resize(originalPointPool);
            threadPack.pointPool->resize(originalPointPool);

            if(threadPack.incrementalMesh)
            {
                boost::mutex::scoped_lock lockMesh(threadPack.incMeshMutex);

                threadPack.incrementalMesh->updateInternalState(threadPack.pointPool);

                threadPack.incMeshLooped.assignValue(true);

                lockMesh.unlock();
            }

            threadPack.poolLooped.assignValue(true);
        }
        else
        {
            std::cout << "Rejecting high residual from iSAM" << std::endl;
            iSAM->removeFactor(factor);
            iSAM->optimise();
            vertexTimes.resize(originalPointPool);
            threadPack.pointPool->resize(originalPointPool);
        }
    }

}

void Deformation::addVertices()
{
    unsigned int startingPointCount = threadPack.pointPool->size();

    if(threadPack.incrementalMesh)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr aggCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr aggCloudFiltered(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        std::vector<uint64_t> pointTimes;

        for(int i = latestProcessedPose; i < latestPoseIdCopy; i++)
        {
            if(threadPack.cloudSlices.at(i)->processedCloud->size() > 0)
            {
                shouldTime = true;

                pcl::PointCloud<pcl::PointXYZRGBNormal> tempCloud;

                tempCloud.insert(tempCloud.end(),
                                 threadPack.cloudSlices.at(i)->processedCloud->begin(),
                                 threadPack.cloudSlices.at(i)->processedCloud->end());

                Eigen::Matrix4f rawTransform = Eigen::Matrix4f::Identity();

                rawTransform.topLeftCorner(3, 3) = threadPack.cloudSlices.at(i)->cameraRotation;
                rawTransform.topRightCorner(3, 1) = threadPack.cloudSlices.at(i)->cameraTranslation;

                Eigen::Matrix4f relativeTransform = iSAM->getCameraPose(threadPack.cloudSlices.at(i)->utime) * rawTransform.inverse();

                pcl::transformPointCloud(tempCloud, tempCloud, relativeTransform);

                aggCloud->insert(aggCloud->end(), tempCloud.begin(), tempCloud.end());

                for(size_t j = 0; j < aggCloud->size(); j++)
                {
                    pointTimes.push_back(threadPack.cloudSlices.at(i)->utime);
                }
            }
        }

        pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor;
        sor.setInputCloud(aggCloud);

        const float3 & voxelSizeMeters = Volume::get().getVoxelSizeMeters();

        float leafSize = std::max(voxelSizeMeters.x,
                         std::max(voxelSizeMeters.y,
                                  voxelSizeMeters.z));

        sor.setLeafSize(leafSize, leafSize, leafSize);

        sor.filter(*aggCloudFiltered);

        if(aggCloudFiltered->size())
        {
            boost::mutex::scoped_lock lockMesh(threadPack.incMeshMutex);

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sourceCloud = threadPack.incrementalMesh->computeIncrementalMesh(aggCloudFiltered, pointTimes);

            lockMesh.unlock();

            vertexTimes.insert(vertexTimes.end(), pointTimes.begin(), pointTimes.end());

            boost::mutex::scoped_lock lock(threadPack.poolMutex);

            threadPack.pointPool->insert(threadPack.pointPool->end(),
                                         sourceCloud->begin(),
                                         sourceCloud->end());
        }
    }
    else
    {
        for(int i = latestProcessedPose; i < latestPoseIdCopy; i++)
        {
            if(threadPack.cloudSlices.at(i)->processedCloud->size() > 0)
            {
                shouldTime = true;

                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sourceCloud = threadPack.cloudSlices.at(i)->processedCloud->makeShared();

                pcl::PointCloud<pcl::PointXYZRGBNormal> tempCloud;

                tempCloud.insert(tempCloud.end(),
                                 sourceCloud->begin(),
                                 sourceCloud->end());

                Eigen::Matrix4f rawTransform = Eigen::Matrix4f::Identity();

                rawTransform.topLeftCorner(3, 3) = threadPack.cloudSlices.at(i)->cameraRotation;
                rawTransform.topRightCorner(3, 1) = threadPack.cloudSlices.at(i)->cameraTranslation;

                Eigen::Matrix4f relativeTransform = iSAM->getCameraPose(threadPack.cloudSlices.at(i)->utime) * rawTransform.inverse();

                pcl::transformPointCloud(tempCloud, tempCloud, relativeTransform);

                boost::mutex::scoped_lock lock(threadPack.poolMutex);

                int lastEnd = threadPack.pointPool->size();

                threadPack.pointPool->insert(threadPack.pointPool->end(),
                                             tempCloud.begin(),
                                             tempCloud.end());

                for(unsigned int j = lastEnd; j < threadPack.pointPool->size(); j++)
                {
                    vertexTimes.push_back(threadPack.cloudSlices.at(i)->utime);
                }
            }
        }
    }

    if(startingPointCount == threadPack.pointPool->size())
    {
        //No need to resample if no points were added
        return;
    }

    std::vector<std::pair<uint64_t, Eigen::Vector3d> > prevPoseGraph;
    iSAM->getCameraPositions(prevPoseGraph);

    const unsigned int k = 4;
    unsigned int poseCount = 0;
    unsigned int lastPose = 0;

    for(unsigned int i = 1; i < prevPoseGraph.size() && poseCount < k; i++)
    {
        if((prevPoseGraph.at(lastPose).second - prevPoseGraph.at(i).second).norm() > ConfigArgs::get().denseSamplingRate)
        {
            poseCount++;
            lastPose = i;
        }
    }

    if(poseCount >= k)
    {
        pcl::PointXYZRGBNormal newPoint;
        int originalPointPool = threadPack.pointPool->size();

        std::vector<std::pair<uint64_t, Eigen::Vector3d> > prevPoseGraph;
        iSAM->getCameraPositions(prevPoseGraph);

        for(unsigned int i = 0; i < prevPoseGraph.size(); i++)
        {
            newPoint.x = prevPoseGraph.at(i).second(0);
            newPoint.y = prevPoseGraph.at(i).second(1);
            newPoint.z = prevPoseGraph.at(i).second(2);

            graphPosePoints->push_back(newPoint);
            graphPoseTimes.push_back(prevPoseGraph.at(i).first);
        }

        if(!deformationGraph)
        {
            deformationGraph = new DeformationGraph(k);
            deformationGraph->initialiseGraphPoses(threadPack.pointPool.get(),
                                                   ConfigArgs::get().denseSamplingRate,
                                                   graphPosePoints.get(),
                                                   &graphPoseTimes,
                                                   &vertexTimes,
                                                   originalPointPool);
        }
        else
        {
            deformationGraph->appendGraphPoses(ConfigArgs::get().denseSamplingRate,
                                               graphPosePoints.get(),
                                               &graphPoseTimes,
                                               &vertexTimes,
                                               originalPointPool);
        }

        graphPoseTimes.clear();
        graphPosePoints->clear();
    }
}

bool inline Deformation::process()
{
    latestPoseIdCopy = threadPack.latestPoseId.getValue();

    if(latestPoseIdCopy == 0)
    {
        return true;
    }

    shouldTime = false;

    //We have to wait for the dense poses to catch up with the slices (they may appear slightly before)
    do
    {
        latestDensePoseIdCopy = threadPack.tracker->latestDensePoseId.getValueWait(50000);
    } while(latestDensePoseIdCopy == 0 || (threadPack.tracker->densePoseGraph.at(latestDensePoseIdCopy - 1).timestamp < threadPack.cloudSlices.at(latestPoseIdCopy - 1)->utime));

    TICK(threadIdentifier);

    latestLoopIdCopy = threadPack.latestLoopId.getValue();

    addCameraCamera();
    latestProcessedDensePose = latestDensePoseIdCopy;

    addVertices();
    latestProcessedPose = latestPoseIdCopy;

    threadPack.readyForLoop.assignValue(false);
    addCameraLoop();
    latestProcessedLoop = latestLoopIdCopy;
    threadPack.readyForLoop.assignValue(true);

    if(latestProcessedPose > 1)
    {
        for(int i = 0; i < latestProcessedPose; i++)
        {
            Eigen::Matrix4f prevPose = Eigen::Matrix4f::Identity();
            prevPose.topLeftCorner(3, 3) = threadPack.cloudSlices.at(i)->cameraRotation;
            prevPose.topRightCorner(3, 1) = threadPack.cloudSlices.at(i)->cameraTranslation;

            Eigen::Matrix4f newPose = iSAM->getCameraPose(threadPack.cloudSlices.at(i)->utime);
            threadPack.cloudSlices.at(i)->cameraRotation = newPose.topLeftCorner(3, 3);
            threadPack.cloudSlices.at(i)->cameraTranslation = newPose.topRightCorner(3, 1);

            if(ConfigArgs::get().dynamicCube)
            {
                static bool once = false;
                if(!once && (newPose.topRightCorner(3, 1) - prevPose.topRightCorner(3, 1)).norm() > 0.001)
                {
                    threadPack.isamOffset.assignValue((newPose * prevPose.inverse()));
                    once = true;
                }

                threadPack.cloudSlices.at(i)->poseIsam.assignValue(true);
            }
        }
    }

    if(shouldTime)
    {
        TOCK(threadIdentifier);
    }

    if(latestProcessedPose || latestProcessedLoop)
    {
        lagTime.assignValue(Stopwatch::getCurrentSystemTime() - std::min(latestProcessedPose ? threadPack.cloudSlices.at(latestProcessedPose - 1)->lagTime : std::numeric_limits<uint64_t>::max(),
                                                                         latestProcessedLoop ? threadPack.loopClosureConstraints.at(latestProcessedLoop - 1)->lagTime : std::numeric_limits<uint64_t>::max()));
    }

    if(threadPack.placeRecognitionFinished.getValue())
    {
        latestDensePoseIdCopy = threadPack.tracker->latestDensePoseId.getValue();
        latestPoseIdCopy = threadPack.latestPoseId.getValue();
        latestLoopIdCopy = threadPack.latestLoopId.getValue();

        if(latestProcessedLoop < latestLoopIdCopy ||
           latestProcessedDensePose < latestDensePoseIdCopy ||
           latestProcessedPose < latestPoseIdCopy ||
           threadPack.cloudSlices.at(latestPoseIdCopy - 1)->dimension != CloudSlice::FINAL)
        {
            return true;
        }

        threadPack.deformationFinished.assignAndNotifyAll(true);
        lagTime.assignValue(0);
        return false;
    }

    return true;
}
