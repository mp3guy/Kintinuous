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

#include "PlaceRecognition.h"

PlaceRecognition::PlaceRecognition(cv::Mat * depthIntrinsics)
 : ThreadObject("PlaceRecognitionThread"),
   depthCamera(new DepthCamera(depthIntrinsics))
{
    dbowInterface = new DBowInterfaceSurf(Resolution::get().width(), Resolution::get().height(), DBowInterfaceSurf::LOOP_DETECTION, ConfigArgs::get().vocabFile);
    oldImage = new cv::Mat(Resolution::get().rows(), Resolution::get().cols(), CV_8UC3);
    newImage = new cv::Mat(Resolution::get().rows(), Resolution::get().cols(), CV_8UC3);
    imageGray = new cv::Mat(Resolution::get().rows(), Resolution::get().cols(), CV_8U);
    depthMapNew = new cv::Mat(Resolution::get().rows(), Resolution::get().cols(), CV_16U);
    depthMapOld = new cv::Mat(Resolution::get().rows(), Resolution::get().cols(), CV_16U);
    reset();
}

PlaceRecognition::~PlaceRecognition()
{
    delete dbowInterface;
    delete oldImage;
    delete newImage;
    delete imageGray;
    delete depthMapNew;
    delete depthMapOld;
    delete depthCamera;
}

void PlaceRecognition::reset()
{
    latestProcessedFrame = 0;
    dbowInterface->reset();
}

bool inline PlaceRecognition::process()
{
    latestFrameIdCopy = threadPack.tracker->placeRecognitionId.getValueWait();

    while(latestProcessedFrame < latestFrameIdCopy)
    {
        TICK(threadIdentifier);

        assert(threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].rgbImage);

        if(!threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].isCompressed)
        {
            memcpy(newImage->data, threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].rgbImage, Resolution::get().numPixels() * 3);
            threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].compress();
        }
        else
        {
            threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].decompressImgTo(newImage->data);
        }

        cv::cvtColor(*newImage, *imageGray, CV_RGB2GRAY);

        DLoopDetector::DetectionResult result;

        dbowInterface->detectSURF(*imageGray,
                                  threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].descriptor,
                                  threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].keyPoints);

        result = dbowInterface->detectLoop();

        if(result.detection())
        {
            processLoopClosureDetection(result.match);
        }
        latestProcessedFrame++;

        TOCK(threadIdentifier);
    }

    if(latestProcessedFrame)
    {
        lagTime.assignValue(Stopwatch::getCurrentSystemTime() - threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame - 1].lagTime);
    }

    if(threadPack.cloudSliceProcessorFinished.getValue())
    {
        latestFrameIdCopy = threadPack.tracker->placeRecognitionId.getValue();

        if(latestProcessedFrame < latestFrameIdCopy)
        {
            return true;
        }

        threadPack.placeRecognitionFinished.assignAndNotifyAll(true);

        lagTime.assignValue(0);

        return false;
    }

    return true;
}

void PlaceRecognition::processLoopClosureDetection(int matchId)
{
    while(!threadPack.readyForLoop.getValueWait(1000));

    uint64_t lastLoopTime = threadPack.lastLoopTime.getValue();

    if(lastLoopTime > 0 && Stopwatch::getCurrentSystemTime() - lastLoopTime <= uint64_t(ConfigArgs::get().loopThrottle) * 1000000ll)
    {
        return;
    }

    vector<vector<float> > matches1, matches2;
    std::vector<InterestPoint *> iP1, iP2;

    assert(threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].isCompressed &&
           threadPack.tracker->placeRecognitionBuffer[matchId].isCompressed);

    //For new image
    threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].decompressDepthTo(depthMapNew->data);

    Surf3DTools::Surf3DImage *  image3DSurfOne = Surf3DTools::calculate3dPointsSURF(depthCamera,
                                                                                    depthMapNew,
                                                                                    threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].descriptor,
                                                                                    threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].keyPoints);

    threadPack.tracker->placeRecognitionBuffer[matchId].decompressImgTo(oldImage->data);
    cv::cvtColor(*oldImage, *imageGray, CV_RGB2GRAY);
    threadPack.tracker->placeRecognitionBuffer[matchId].decompressDepthTo(depthMapOld->data);

    Surf3DTools::Surf3DImage *  image3DSurfTwo = Surf3DTools::calculate3dPointsSURF(depthCamera,
                                                                                    depthMapOld,
                                                                                    threadPack.tracker->placeRecognitionBuffer[matchId].descriptor,
                                                                                    threadPack.tracker->placeRecognitionBuffer[matchId].keyPoints);

    Surf3DTools::surfMatch3D(image3DSurfOne, image3DSurfTwo, matches1, matches2);

    assert(matches1.size() == matches2.size());

    if(matches1.size() < 40)
    {
        delete image3DSurfOne;
        delete image3DSurfTwo;
        return;
    }

    for (std::vector<std::vector<float> >::const_iterator m = matches1.begin(); m != matches1.end(); m++)
    {
        iP1.push_back(new InterestPoint((*m)[0],(*m)[1],(*m)[2],(*m)[3],(*m)[4]));
    }

    for (std::vector<std::vector<float> >::const_iterator m = matches2.begin(); m != matches2.end(); m++)
    {
        iP2.push_back(new InterestPoint((*m)[0],(*m)[1],(*m)[2],(*m)[3],(*m)[4]));
    }

    std::vector<std::pair<int2, int2> > inliers;
    isam::Pose3d pose;

    PNPSolver * pnpSolver = new PNPSolver(depthCamera);

    pnpSolver->getRelativePose(pose, inliers, iP1, iP2);

    std::cout << "Loop found with image " << matchId << ", matches: " << matches1.size() << ", inliers: " << round(float(inliers.size() * 100) / matches1.size()) << "%... ";
    std::cout.flush();

    if(float(inliers.size()) / matches1.size() > ConfigArgs::get().inlierRatio)
    {
        float score = 0;

        Eigen::Matrix4d T;
        T.topLeftCorner(3,3) = pose.rot().wRo();
        T.col(3).head(3) << pose.x(), pose.y(), pose.z();
        T.row(3) << 0., 0., 0., 1.;

        Eigen::Matrix4f bootstrap = T.cast<float>().inverse();

        TICK("LoopConstraint");
        Eigen::Matrix4f icpTrans = icpDepthFrames(bootstrap,
                                                  (unsigned short *)depthMapOld->data,
                                                  (unsigned short *)depthMapNew->data,
                                                  score);
        TOCK("LoopConstraint");
        if(score < 0.01)
        {
            uint64_t time1 = threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].utime;
            uint64_t time2 = threadPack.tracker->placeRecognitionBuffer[matchId].utime;

            std::vector<Eigen::Vector3d> inliers1Proj, inliers2Proj;

            depthCamera->projectInlierMatches(inliers,
                                              inliers1Proj,
                                              inliers2Proj,
                                              (unsigned short *)depthMapNew->data,
                                              (unsigned short *)depthMapOld->data);

            threadPack.loopClosureConstraints.push_back(new LoopClosureConstraint(time1, time2, icpTrans, inliers1Proj, inliers2Proj, threadPack.tracker->placeRecognitionBuffer[latestProcessedFrame].lagTime));

            threadPack.latestLoopId.assignValue(threadPack.loopClosureConstraints.size());

            std::cout << "accepted! - " << latestProcessedFrame << " images" << std::endl;

            lastLoopTime = 0;
        }
        else
        {
            std::cout << "rejected on ICP score" << std::endl;
        }
    }
    else
    {
        std::cout << "rejected on inlier percentage" << std::endl;
    }

    for(unsigned int i = 0; i < matches1.size(); i++)
    {
        delete iP1.at(i);
        delete iP2.at(i);
    }

    delete image3DSurfOne;
    delete image3DSurfTwo;
    delete pnpSolver;
}

Eigen::Matrix4f PlaceRecognition::icpDepthFrames(Eigen::Matrix4f & bootstrap, unsigned short * frame1, unsigned short * frame2, float & score)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOne = depthCamera->convertToXYZPointCloud(frame1);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTwo = depthCamera->convertToXYZPointCloud(frame2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOneF(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTwoF(new pcl::PointCloud<pcl::PointXYZ>);

    const float3 & voxelSizeMeters = Volume::get().getVoxelSizeMeters();

    pcl::VoxelGrid<pcl::PointXYZ> sor;

    sor.setLeafSize(voxelSizeMeters.x * 2.5f, voxelSizeMeters.y * 2.5f, voxelSizeMeters.z * 2.5f);

    sor.setInputCloud(cloudOne);
    sor.filter(*cloudOneF);

    sor.setInputCloud(cloudTwo);
    sor.filter(*cloudTwoF);

    pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned (new pcl::PointCloud <pcl::PointXYZ>);

    pcl::transformPointCloud(*cloudOneF, *cloudOneF, bootstrap);

    icp.setInputSource(cloudOneF);
    icp.setInputTarget(cloudTwoF);
    icp.align(*aligned);

    std::cout << "score: " << icp.getFitnessScore() << ", ";
    std::cout.flush();

    Eigen::Matrix4f d = icp.getFinalTransformation() * bootstrap;

    score = icp.getFitnessScore();

    return d;
}
