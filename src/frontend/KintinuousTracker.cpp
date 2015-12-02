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
 *
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <fstream>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "../utils/ConfigArgs.h"
#include "KintinuousTracker.h"
#include "Volume.h"

KintinuousTracker::KintinuousTracker(cv::Mat * depthIntrinsics)
 : tsdfAvailable(false),
   imageAvailable(false),
   cycledMutex(false),
   global_time_(0),
   place_recognition_movement(0.15),
   cloud_device_ (Resolution::get().numPixels() * 3),
   liveTsdf(0),
   liveImage(0),
   cycled(false),
   overlap(0),
   parked(false),
   icp(0),
   rgbd(0),
   groundTruth(0)
{
    intr.fx = depthIntrinsics->at<double>(0, 0);
    intr.fy = depthIntrinsics->at<double>(1, 1);
    intr.cx = depthIntrinsics->at<double>(0, 2);
    intr.cy = depthIntrinsics->at<double>(1, 2);

    const Eigen::Vector3f volume_size = Eigen::Vector3f::Constant(Volume::get().getVolumeSize());
    const Eigen::Vector3i volume_resolution(VOLUME_X, VOLUME_Y, VOLUME_Z);

    tsdf_volume_ = new TsdfVolume(volume_resolution);

    tsdf_volume_->setSize(volume_size);

    color_volume_ = new ColorVolume(*tsdf_volume_);

    initialRotation = Eigen::Matrix3f::Identity();

    if(ConfigArgs::get().staticMode || ConfigArgs::get().dynamicCube)
    {
        volumeBasis = volume_size * 0.5f - Eigen::Vector3f(0, 0, (volume_size(2) * 0.5) + (ConfigArgs::get().staticMode ? 0.45 : 0));
    }
    else
    {
        volumeBasis = volume_size * 0.5f;
    }

    const float default_tranc_dist = std::max(0.01f, volume_size(0) / 100.0f);
    tsdf_volume_->setTsdfTruncDist (default_tranc_dist);

    allocateBuffers();

    rmats_.reserve(30000);
    tvecs_.reserve(30000);

    voxelWrap.x = 0;
    voxelWrap.y = 0;
    voxelWrap.z = 0;

    init_utime.assignValue(std::numeric_limits<unsigned long long>::max());
    firstDepthData.assignValue(0);
    firstRgbImage.assignValue(0);

    if(ConfigArgs::get().trajectoryFile.size())
    {
        groundTruth = new GroundTruthOdometry(tvecs_,
                                              rmats_,
                                              camera_trajectory,
                                              current_utime);

        odometryProvider = dynamic_cast<OdometryProvider *>(groundTruth);

        lastOdometry = CloudSlice::GROUNDTRUTH;

        std::cout << "Using ground truth odometry" << std::endl;
    }
    else if(ConfigArgs::get().useRGBD || ConfigArgs::get().useRGBDICP)
    {
        rgbd = new RGBDOdometry(tvecs_,
                                rmats_,
                                vmaps_g_prev_,
                                nmaps_g_prev_,
                                vmaps_curr_,
                                nmaps_curr_,
                                intr);

        odometryProvider = dynamic_cast<OdometryProvider *>(rgbd);

        lastOdometry = CloudSlice::RGBD;

        if(ConfigArgs::get().useRGBDICP)
        {
            std::cout << "Using ICP+RGB-D odometry" << std::endl;
        }
        else
        {
            std::cout << "Using RGB-D odometry" << std::endl;
        }
    }
    else
    {
        icp = new ICPOdometry(tvecs_,
                              rmats_,
                              vmaps_g_prev_,
                              nmaps_g_prev_,
                              vmaps_curr_,
                              nmaps_curr_,
                              intr);

        odometryProvider = dynamic_cast<OdometryProvider *>(icp);

        lastOdometry = CloudSlice::ICP;

        std::cout << "Using ICP odometry" << std::endl;
    }

    reset();
}

KintinuousTracker::~KintinuousTracker()
{
    if(icp)
        delete icp;

    if(rgbd)
        delete rgbd;

    if(groundTruth)
        delete groundTruth;

    delete color_volume_;
    delete tsdf_volume_;
}

void KintinuousTracker::outputPose(uint64_t & timestamp, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & Rcurr)
{
    std::string filename = ConfigArgs::get().saveFile;
    filename.append(".poses");

    std::ofstream file;
    file.open(filename.c_str(), std::fstream::app);

    std::stringstream strs;

    strs << std::setprecision(6) << std::fixed << (double)timestamp / 1000000.0 << " ";

    file << strs.str() << currentGlobalCamera(0) << " " << currentGlobalCamera(1) << " " << currentGlobalCamera(2) << " ";

    Eigen::Quaternionf currentCameraRotation(Rcurr);

    file << currentCameraRotation.x() << " " << currentCameraRotation.y() << " " << currentCameraRotation.z() << " " << currentCameraRotation.w() << "\n";

    file.close();
}

void KintinuousTracker::loadTrajectory(const std::string & filename)
{
    std::ifstream file;
    std::string line;
    file.open(filename.c_str());

    double trajSum = 0.0;
    bool first = true;
    Eigen::Vector3f lastT;

    while (!file.eof())
    {
        long long unsigned int utime;
        float x, y, z, qx, qy, qz, qw;
        std::getline(file, line);
        int n = sscanf(line.c_str(), "%llu,%f,%f,%f,%f,%f,%f,%f", &utime, &x, &y, &z, &qx, &qy, &qz, &qw);

        if(file.eof())
            break;

        assert(n == 8);

        Eigen::Quaternionf q(qw, qx, qy, qz);
        Eigen::Vector3f t(x, y, z);

        if(!first)
        {
            trajSum += (t - lastT).norm();
        }

        first = false;
        lastT = t;

        Eigen::Isometry3f T;
        T.setIdentity();
        T.pretranslate(t).rotate(q);
        camera_trajectory[utime] = T;
    }
    std::cout << "Done loading ground truth, length: " << trajSum << std::endl;
    current_utime = 0;
}

void KintinuousTracker::reset()
{
    modelColor.create(Resolution::get().rows(), Resolution::get().cols());
    modelSurface.create(Resolution::get().rows(), Resolution::get().cols());
    modelDepth.create(Resolution::get().rows(), Resolution::get().cols());
    global_time_ = 0;
    rmats_.clear ();
    tvecs_.clear ();

    cycled = false;

    rmats_.push_back(initialRotation);
    tvecs_.push_back(volumeBasis);

    Eigen::Vector3f initialTrans = volumeBasis;
    initialTrans(0) -= Volume::get().getVolumeSize() * 0.5;
    initialTrans(1) -= Volume::get().getVolumeSize() * 0.5;
    initialTrans(2) -= Volume::get().getVolumeSize() * 0.5;

    const float3 & voxelSizeMeters = Volume::get().getVoxelSizeMeters();

    Eigen::Vector3f currentCameraTranslation = initialTrans;
    currentCameraTranslation(0) += voxelWrap.x * voxelSizeMeters.x;
    currentCameraTranslation(1) += voxelWrap.y * voxelSizeMeters.y;
    currentCameraTranslation(2) += voxelWrap.z * voxelSizeMeters.z;

    currentGlobalCamera = currentCameraTranslation;

    lastPlaceRecognitionTrans = currentGlobalCamera;
    lastPlaceRecognitionRot = initialRotation;

    for(int i = 0; i < PR_BUFFER_SIZE; i++)
    {
        placeRecognitionBuffer[i].dump();
    }

    placeRecognitionId.assignValue(0);

    densePoseGraph.clear();
    latestDensePoseId.assignValue(0);

    voxelWrap.x = 0;
    voxelWrap.y = 0;
    voxelWrap.z = 0;

    odometryProvider->reset();

    init_utime.assignValue(std::numeric_limits<unsigned long long>::max());

    unsigned short * firstDepth = firstDepthData.getValue();

    if(firstDepth)
    {
        delete [] firstDepth;
    }

    unsigned char * firstImg = firstRgbImage.getValue();

    if(firstImg)
    {
        delete [] firstImg;
    }

    firstDepthData.assignValue(0);
    firstRgbImage.assignValue(0);

    if(ConfigArgs::get().staticMode)
    {
        parked = true;
    }
    else
    {
        parked = false;
    }

    for(unsigned int i = 0; i < sharedCloudSlices.size(); i++)
    {
        delete sharedCloudSlices.at(i);
    }
    sharedCloudSlices.clear();

    tsdf_volume_->reset();
    color_volume_->reset();

    if(!ConfigArgs::get().onlineDeformation)
    {
        std::string filename = ConfigArgs::get().saveFile;
        filename.append(".poses");
        std::ofstream file;
        file.open(filename.c_str());
        file.close();
    }
}

void KintinuousTracker::allocateBuffers()
{    
    depths_curr_.resize (ICPOdometry::LEVELS);

    vmaps_g_prev_.resize (ICPOdometry::LEVELS);
    nmaps_g_prev_.resize (ICPOdometry::LEVELS);

    vmaps_curr_.resize (ICPOdometry::LEVELS);
    nmaps_curr_.resize (ICPOdometry::LEVELS);

    for (int i = 0; i < ICPOdometry::LEVELS; ++i)
    {
        int pyr_rows = Resolution::get().rows() >> i;
        int pyr_cols = Resolution::get().cols() >> i;

        depths_curr_[i].create (pyr_rows, pyr_cols);

        vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
        nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

        vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
        nmaps_curr_[i].create (pyr_rows*3, pyr_cols);
    }

    vmap_curr_color.create(Resolution::get().rows(), Resolution::get().cols());
    depthRawScaled_.create(Resolution::get().rows(), Resolution::get().cols());
}

void KintinuousTracker::repositionCube(Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & currentRotation)
{
    Eigen::Vector3f rotations = rodrigues2(currentRotation);

    float yRot = rotations(1);

    const float PI = 3.14159265359f;

    float radius = Volume::get().getVolumeSize() * 0.5;

    Eigen::Vector3f newPosition = volumeBasis;

    newPosition(0) = radius * (cos(yRot + (PI / 2)) + 1.0f);
    newPosition(2) = radius * (sin(yRot - (PI / 2)) + 1.0f);

    Eigen::Vector3f currentTranslation = (tvecs_.at(tvecs_.size() - 1) - newPosition);

    Eigen::Vector3f voxelSize = tsdf_volume_->getVoxelSize();

    const int thresh = parked ? ConfigArgs::get().staticMode ? VOLUME_X * 3 : VOLUME_X : ConfigArgs::get().voxelShift;

    int xVoxelTrans = 0;
    int yVoxelTrans = 0;
    int zVoxelTrans = 0;

    if((int)std::floor(currentTranslation(0) / voxelSize(0)) < 0)
    {
        xVoxelTrans = std::max(-thresh, (int)std::floor(currentTranslation(0) / voxelSize(0)));
    }
    else
    {
        xVoxelTrans = std::min(thresh, (int)std::floor(currentTranslation(0) / voxelSize(0)));
    }

    if((int)std::floor(currentTranslation(1) / voxelSize(1)) < 0)
    {
        yVoxelTrans = std::max(-thresh, (int)std::floor(currentTranslation(1) / voxelSize(1)));
    }
    else
    {
        yVoxelTrans = std::min(thresh, (int)std::floor(currentTranslation(1) / voxelSize(1)));
    }

    if((int)std::floor(currentTranslation(2) / voxelSize(2)) < 0)
    {
        zVoxelTrans = std::max(-thresh, (int)std::floor(currentTranslation(2) / voxelSize(2)));
    }
    else
    {
        zVoxelTrans = std::min(thresh, (int)std::floor(currentTranslation(2) / voxelSize(2)));
    }

    if(xVoxelTrans >= thresh || xVoxelTrans <= -thresh ||
       yVoxelTrans >= thresh || yVoxelTrans <= -thresh ||
       zVoxelTrans >= thresh || zVoxelTrans <= -thresh)
    {
        volumeBasis = newPosition;
    }
}

void KintinuousTracker::processFrame(const DeviceArray2D<unsigned short>& depth_raw,
                                     const DeviceArray2D<PixelRGB> & colors,
                                     unsigned char * rgbImage,
                                     unsigned short * depthData,
                                     uint64_t timestamp,
                                     bool compression,
                                     uint8_t * lastCompressedDepth,
                                     int depthSize,
                                     uint8_t * lastCompressedImage,
                                     int imageSize)
{  
    lagTime = Stopwatch::getCurrentSystemTime();

    lastDepthData = depthData;
    lastRgbImage = rgbImage;

    if(groundTruth &&!groundTruth->preRun(rgbImage, depthData, timestamp))
    {
        return;
    }

    if(icp || ConfigArgs::get().useRGBDICP || !ConfigArgs::get().disableColorAngleWeight)
    {
        bilateralFilter(depth_raw, depths_curr_[0]);

        for (int i = 1; i < ICPOdometry::LEVELS; ++i)
        {
            pyrDown(depths_curr_[i-1], depths_curr_[i]);
        }

        for (int i = 0; i < ICPOdometry::LEVELS; ++i)
        {
            createVMap(intr(i), depths_curr_[i], vmaps_curr_[i]);
            createNMap(vmaps_curr_[i], nmaps_curr_[i]);
        }
    }

    if(global_time_ == 0)
    {
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> init_Rcam = rmats_[rmats_.size() - 1];
        Eigen::Vector3f   init_tcam = tvecs_[tvecs_.size() - 1];

        Mat33&  device_Rcam = device_cast<Mat33> (init_Rcam);
        float3& device_tcam = device_cast<float3>(init_tcam);

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> init_Rcam_inv = init_Rcam.inverse ();
        Mat33&   device_Rcam_inv = device_cast<Mat33> (init_Rcam_inv);
        float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

        int3 emptyVoxel;

        emptyVoxel.x = 0;
        emptyVoxel.y = 0;
        emptyVoxel.z = 0;

        if(rgbd)
        {
            rgbd->firstRun(depth_raw, colors);
        }

        integrateTsdfVolume(depth_raw,
                            intr,
                            device_volume_size,
                            device_Rcam_inv,
                            device_tcam,
                            tsdf_volume_->getTsdfTruncDist(),
                            tsdf_volume_->data(),
                            depthRawScaled_,
                            emptyVoxel,
                            color_volume_->data(),
                            colors,
                            nmaps_curr_[0],
                            !ConfigArgs::get().disableColorAngleWeight);

        for (int i = 0; i < ICPOdometry::LEVELS; ++i)
        {
            tranformMaps(vmaps_curr_[i], nmaps_curr_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
        }

        ++global_time_;

        init_utime.assignValue(timestamp);

        last_utime = timestamp;
        current_utime = timestamp;

        Eigen::Matrix4f curr = Eigen::Matrix4f::Identity();
        curr.topLeftCorner(3,3) = init_Rcam;
        curr.topRightCorner(3,1) = currentGlobalCamera;

        densePoseGraph.push_back(DensePose(current_utime, curr, true));

        latestDensePoseId++;

        unsigned short * firstDepth = new unsigned short[Resolution::get().numPixels()];
        memcpy(firstDepth, lastDepthData, Resolution::get().numPixels() * 2);

        unsigned char * firstImg = new unsigned char[Resolution::get().numPixels() * 3];
        memcpy(firstImg, lastRgbImage, Resolution::get().numPixels() * 3);

        firstDepthData.assignValue(firstDepth);
        firstRgbImage.assignValue(firstImg);

        if(ConfigArgs::get().vocabFile.size())
        {
            addToPlaceRecognition(depthSize, imageSize, compression, (unsigned short *)lastCompressedDepth, (unsigned char *)lastCompressedImage);
        }

        boost::mutex::scoped_lock lock(cloudMutex);

        cloudSignal.notify_all();

        return;
    }

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rmats_[rmats_.size() - 1];
    Eigen::Vector3f tprev = tvecs_[tvecs_.size() - 1];
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
    Eigen::Vector3f tcurr = tprev;

    TICK("Odometry");
    lastOdometry = odometryProvider->getIncrementalTransformation(tcurr,
                                                                  Rcurr,
                                                                  depth_raw,
                                                                  colors,
                                                                  timestamp,
                                                                  rgbImage,
                                                                  depthData);
    TOCK("Odometry");

    last_utime = current_utime;
    current_utime = timestamp;

    //save tranform
    rmats_.push_back(Rcurr);
    tvecs_.push_back(tcurr);

    Eigen::Vector3f initialTrans = volumeBasis;
    initialTrans(0) -= Volume::get().getVolumeSize() * 0.5;
    initialTrans(1) -= Volume::get().getVolumeSize() * 0.5;
    initialTrans(2) -= Volume::get().getVolumeSize() * 0.5;

    const float3 & voxelSizeMeters = Volume::get().getVoxelSizeMeters();

    currentGlobalCamera = initialTrans;
    currentGlobalCamera(0) += voxelWrap.x * voxelSizeMeters.x;
    currentGlobalCamera(1) += voxelWrap.y * voxelSizeMeters.y;
    currentGlobalCamera(2) += voxelWrap.z * voxelSizeMeters.z;

    currentGlobalCamera(0) += tcurr(0) - volumeBasis(0);
    currentGlobalCamera(1) += tcurr(1) - volumeBasis(1);
    currentGlobalCamera(2) += tcurr(2) - volumeBasis(2);

    if(ConfigArgs::get().dynamicCube)
    {
        repositionCube(Rcurr);
    }

    bool shiftSend = false;
    bool isLoopPose = false;

    if(ConfigArgs::get().vocabFile.size())
    {
        float rnorm = rodrigues2(Rcurr.inverse() * lastPlaceRecognitionRot).norm();
        float tnorm = (currentGlobalCamera - lastPlaceRecognitionTrans).norm();
        const float alpha = 1.f;

        if((rnorm + alpha * tnorm)/2 >= place_recognition_movement)
        {
            lastPlaceRecognitionRot = Rcurr;
            lastPlaceRecognitionTrans = currentGlobalCamera;

            addToPlaceRecognition(depthSize, imageSize, compression, (unsigned short *)lastCompressedDepth, (unsigned char *)lastCompressedImage);

            isLoopPose = true;
        }
        else
        {
            shiftSend = true;
        }
    }

    float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr_inv = Rcurr.inverse ();
    Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);
    Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
    float3& device_tcurr = device_cast<float3> (tcurr);

    Eigen::Vector3f currentTranslation = (tvecs_.at(tvecs_.size() - 1) - volumeBasis);

    Eigen::Vector3f voxelSize = tsdf_volume_->getVoxelSize();

    const int thresh = parked ? ConfigArgs::get().staticMode ? std::numeric_limits<int>::max() : std::numeric_limits<int>::max() : ConfigArgs::get().voxelShift;

    int xVoxelTrans = 0;
    int yVoxelTrans = 0;
    int zVoxelTrans = 0;

    if((int)std::floor(currentTranslation(0) / voxelSize(0)) < 0)
    {
        xVoxelTrans = std::max(-thresh, (int)std::floor(currentTranslation(0) / voxelSize(0)));
    }
    else
    {
        xVoxelTrans = std::min(thresh, (int)std::floor(currentTranslation(0) / voxelSize(0)));
    }

    if((int)std::floor(currentTranslation(1) / voxelSize(1)) < 0)
    {
        yVoxelTrans = std::max(-thresh, (int)std::floor(currentTranslation(1) / voxelSize(1)));
    }
    else
    {
        yVoxelTrans = std::min(thresh, (int)std::floor(currentTranslation(1) / voxelSize(1)));
    }

    if((int)std::floor(currentTranslation(2) / voxelSize(2)) < 0)
    {
        zVoxelTrans = std::max(-thresh, (int)std::floor(currentTranslation(2) / voxelSize(2)));
    }
    else
    {
        zVoxelTrans = std::min(thresh, (int)std::floor(currentTranslation(2) / voxelSize(2)));
    }

    vWrapCopyUpdate();

    cycled = false;

    PlaceRecognitionInput * nextPlaceRecognitionFrame = 0;

    if(xVoxelTrans >= thresh)
    {
        cloud_buffer_ = tsdf_volume_->fetchCloud(cloud_device_,
                                                 vWrapCopy,
                                                 color_volume_->data(),
                                                 0, xVoxelTrans + 1 + overlap,
                                                 0, VOLUME_Y,
                                                 0, VOLUME_Z,
                                                 voxelWrap);

        clearVolumeX(tsdf_volume_->data(), voxelWrap.x, voxelWrap.x + xVoxelTrans);
        clearVolumeXc(color_volume_->data(), voxelWrap.x, voxelWrap.x + xVoxelTrans);

        cycled = true;
    }
    else if(xVoxelTrans <= -thresh)
    {
        cloud_buffer_ = tsdf_volume_->fetchCloud(cloud_device_,
                                                 vWrapCopy,
                                                 color_volume_->data(),
                                                 VOLUME_X + (xVoxelTrans - overlap), VOLUME_X,
                                                 0, VOLUME_Y,
                                                 0, VOLUME_Z,
                                                 voxelWrap);

        clearVolumeXBack(tsdf_volume_->data(), voxelWrap.x, voxelWrap.x + xVoxelTrans);
        clearVolumeXBackc(color_volume_->data(), voxelWrap.x, voxelWrap.x + xVoxelTrans);

        cycled = true;
    }

    if(cycled)
    {
        if(shiftSend)
        {
            lastPlaceRecognitionRot = Rcurr;
            lastPlaceRecognitionTrans = currentGlobalCamera;

            nextPlaceRecognitionFrame = addToPlaceRecognition(depthSize, imageSize, compression, (unsigned short *)lastCompressedDepth, (unsigned char *)lastCompressedImage);

            isLoopPose = true;

            shiftSend = false;
        }

        int3 trans = {xVoxelTrans, 0, 0};
        mutexOutCloudBuffer(device_tcurr, trans, nextPlaceRecognitionFrame);
        nextPlaceRecognitionFrame = 0;
    }

    vWrapCopyUpdate();

    cycled = false;

    if(yVoxelTrans >= thresh)
    {
        cloud_buffer_ = tsdf_volume_->fetchCloud(cloud_device_,
                                                 vWrapCopy,
                                                 color_volume_->data(),
                                                 0, VOLUME_X,
                                                 0, yVoxelTrans + 1 + overlap,
                                                 0, VOLUME_Z,
                                                 voxelWrap);

        clearVolumeY(tsdf_volume_->data(), voxelWrap.y, voxelWrap.y + yVoxelTrans);
        clearVolumeYc(color_volume_->data(), voxelWrap.y, voxelWrap.y + yVoxelTrans);

        cycled = true;
    }
    else if(yVoxelTrans <= -thresh)
    {
        cloud_buffer_ = tsdf_volume_->fetchCloud(cloud_device_,
                                                 vWrapCopy,
                                                 color_volume_->data(),
                                                 0, VOLUME_X,
                                                 VOLUME_Y + (yVoxelTrans - overlap), VOLUME_Y,
                                                 0, VOLUME_Z,
                                                 voxelWrap);

        clearVolumeYBack(tsdf_volume_->data(), voxelWrap.y, voxelWrap.y + yVoxelTrans);
        clearVolumeYBackc(color_volume_->data(), voxelWrap.y, voxelWrap.y + yVoxelTrans);

        cycled = true;
    }

    if(cycled)
    {
        if(shiftSend)
        {
            lastPlaceRecognitionRot = Rcurr;
            lastPlaceRecognitionTrans = currentGlobalCamera;

            nextPlaceRecognitionFrame = addToPlaceRecognition(depthSize, imageSize, compression, (unsigned short *)lastCompressedDepth, (unsigned char *)lastCompressedImage);

            isLoopPose = true;

            shiftSend = false;
        }

        int3 trans = {0, yVoxelTrans, 0};
        mutexOutCloudBuffer(device_tcurr, trans, nextPlaceRecognitionFrame);
        nextPlaceRecognitionFrame = 0;
    }

    vWrapCopyUpdate();

    cycled = false;

    if(zVoxelTrans >= thresh)
    {
        cloud_buffer_ = tsdf_volume_->fetchCloud(cloud_device_,
                                                 vWrapCopy,
                                                 color_volume_->data(),
                                                 0, VOLUME_X,
                                                 0, VOLUME_Y,
                                                 0, zVoxelTrans + 1 + overlap,
                                                 voxelWrap);

        clearVolumeZ(tsdf_volume_->data(), voxelWrap.z, voxelWrap.z + zVoxelTrans);
        clearVolumeZc(color_volume_->data(), voxelWrap.z, voxelWrap.z + zVoxelTrans);

        cycled = true;
    }
    else if(zVoxelTrans <= -thresh)
    {
        cloud_buffer_ = tsdf_volume_->fetchCloud(cloud_device_,
                                                 vWrapCopy,
                                                 color_volume_->data(),
                                                 0, VOLUME_X,
                                                 0, VOLUME_Y,
                                                 VOLUME_Z + (zVoxelTrans - overlap) - 1, VOLUME_Z - 1,
                                                 voxelWrap);

        clearVolumeZBack(tsdf_volume_->data(), voxelWrap.z, voxelWrap.z + zVoxelTrans);
        clearVolumeZBackc(color_volume_->data(), voxelWrap.z, voxelWrap.z + zVoxelTrans);

        cycled = true;
    }

    if(cycled)
    {
        if(shiftSend)
        {
            lastPlaceRecognitionRot = Rcurr;
            lastPlaceRecognitionTrans = currentGlobalCamera;

            nextPlaceRecognitionFrame = addToPlaceRecognition(depthSize, imageSize, compression, (unsigned short *)lastCompressedDepth, (unsigned char *)lastCompressedImage);

            isLoopPose = true;

            shiftSend = false;
        }

        int3 trans = {0, 0, zVoxelTrans};
        mutexOutCloudBuffer(device_tcurr, trans, nextPlaceRecognitionFrame);
        nextPlaceRecognitionFrame = 0;
    }

    vWrapCopyUpdate();

    if(tsdfRequest.getValue())
    {
        bool tsdfNeeded = false;

        boost::mutex::scoped_lock tsdfLock(tsdfMutex);

        tsdfNeeded = !tsdfAvailable;

        tsdfLock.unlock();

        if(tsdfNeeded)
        {
            mutexOutLiveTsdf();
        }
    }

    bool imageNeeded = false;

    boost::mutex::scoped_lock imageLock(imageMutex);

    imageNeeded = !imageAvailable;

    imageLock.unlock();

    if(imageNeeded)
    {
        mutexOutLiveImage();
    }

    integrateTsdfVolume (depth_raw,
                         intr,
                         device_volume_size,
                         device_Rcurr_inv,
                         device_tcurr,
                         tsdf_volume_->getTsdfTruncDist(),
                         tsdf_volume_->data(),
                         depthRawScaled_,
                         vWrapCopy,
                         color_volume_->data(),
                         colors,
                         nmaps_curr_[0],
                         !ConfigArgs::get().disableColorAngleWeight);

    vWrapCopyUpdate();

    raycast(intr,
            device_Rcurr,
            device_tcurr,
            tsdf_volume_->getTsdfTruncDist(),
            device_volume_size,
            tsdf_volume_->data(),
            vmaps_g_prev_[0],
            nmaps_g_prev_[0],
            vWrapCopy,
            vmap_curr_color,
            color_volume_->data());

    if(icp || ConfigArgs::get().useRGBDICP)
    {
        for (int i = 1; i < ICPOdometry::LEVELS; ++i)
        {
            resizeVMap(vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
            resizeNMap(nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
        }
    }

    ++global_time_;

    Eigen::Matrix4f curr = Eigen::Matrix4f::Identity();
    curr.topLeftCorner(3,3) = Rcurr;
    curr.topRightCorner(3,1) = currentGlobalCamera;

    densePoseGraph.push_back(DensePose(current_utime, curr, isLoopPose));

    latestDensePoseId++;

    if(!ConfigArgs::get().onlineDeformation)
    {
        outputPose(timestamp, Rcurr);
    }
}

PlaceRecognitionInput * KintinuousTracker::addToPlaceRecognition(int depthSize,
                                                                 int imageSize,
                                                                 bool compression,
                                                                 unsigned short * lastCompressedDepth,
                                                                 unsigned char * lastCompressedImage)
{
    int nextSlot = placeRecognitionId.getValue();

    assert(nextSlot < PR_BUFFER_SIZE);

    int depthDataSize = compression ? depthSize : Resolution::get().numPixels() * 2;
    int rgbDataSize = compression ? imageSize : Resolution::get().numPixels() * 3;

    unsigned char * depthPr = new unsigned char[depthDataSize];
    unsigned char * imgPr = new unsigned char[rgbDataSize];

    if(!compression)
    {
        memcpy(depthPr, lastDepthData, depthDataSize);
        memcpy(imgPr, lastRgbImage, rgbDataSize);
    }
    else
    {
        memcpy(depthPr, lastCompressedDepth, depthDataSize);
        memcpy(imgPr, lastCompressedImage, rgbDataSize);
    }

    PlaceRecognitionInput * toReturn = &placeRecognitionBuffer[nextSlot];

    placeRecognitionBuffer[nextSlot].rgbImage = imgPr;
    placeRecognitionBuffer[nextSlot].imageSize = rgbDataSize;
    placeRecognitionBuffer[nextSlot].depthMap = (unsigned short *)depthPr;
    placeRecognitionBuffer[nextSlot].depthSize = depthDataSize;
    placeRecognitionBuffer[nextSlot].isCompressed = compression;
    placeRecognitionBuffer[nextSlot].utime = current_utime;
    placeRecognitionBuffer[nextSlot].lagTime = lagTime;
    placeRecognitionBuffer[nextSlot].trans = lastPlaceRecognitionTrans;
    placeRecognitionBuffer[nextSlot].rotation = lastPlaceRecognitionRot;
    placeRecognitionId++;

    return toReturn;
}

void KintinuousTracker::getImage()
{
    LightSource light;
    light.number = 1;
    light.pos[0].x = tsdf_volume_->getSize()(0) * (-3.f);
    light.pos[0].y = tsdf_volume_->getSize()(1) * (-3.f);
    light.pos[0].z = tsdf_volume_->getSize()(2) * (-3.f);

    generateImage(vmaps_g_prev_[0], nmaps_g_prev_[0], vmap_curr_color, light, modelSurface, modelColor);
}

void KintinuousTracker::getModelDepth()
{
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr_inv = rmats_.back().inverse();
    Mat33  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
    float3 device_tcurr = device_cast<float3>(tvecs_.back());

    generateDepth(device_Rcurr_inv, device_tcurr, vmaps_g_prev_[0], nmaps_g_prev_[0], modelDepth, Volume::get().getVolumeSize());

    int cols;
    modelDepth.download(modelDepthHost, cols);
}

Eigen::Vector3f KintinuousTracker::getVolumeOffset() const
{
    return volumeBasis;
}

void KintinuousTracker::setParked(const bool park)
{
    parked = park;
}

Eigen::Vector3f KintinuousTracker::getLastTranslation() const
{
    return tvecs_.at(tvecs_.size() - 1) - volumeBasis;
}

Eigen::Vector3f KintinuousTracker::getVoxelSize() const
{
    return tsdf_volume_->getVoxelSize();
}

void KintinuousTracker::finalise()
{
    vWrapCopyUpdate();

    pcl::PointCloud<pcl::PointXYZRGB> * newCloud = 0;

    cloud_buffer_ = tsdf_volume_->fetchCloud(cloud_device_,
                                             vWrapCopy,
                                             color_volume_->data(),
                                             0, VOLUME_X,
                                             0, VOLUME_Y,
                                             0, VOLUME_Z,
                                             voxelWrap);

    newCloud = new pcl::PointCloud<pcl::PointXYZRGB>();

    cloud_buffer_.download(newCloud->points);

    boost::mutex::scoped_lock lock(cloudMutex);

    cycledMutex = true;

    sharedCloudSlices.push_back(new CloudSlice(newCloud,
                                               CloudSlice::FINAL,
                                               lastOdometry,
                                               currentGlobalCamera,
                                               rmats_.at(rmats_.size() - 1),
                                               current_utime,
                                               Stopwatch::getCurrentSystemTime(),
                                               lastRgbImage,
                                               0,
                                               0,
                                               lastDepthData));

    if(ConfigArgs::get().vocabFile.size())
    {
        lastPlaceRecognitionRot = rmats_.at(rmats_.size() - 1);
        lastPlaceRecognitionTrans = currentGlobalCamera;

        addToPlaceRecognition(Resolution::get().numPixels() * 2, Resolution::get().numPixels() * 3, false, 0, 0);

        sharedCloudSlices.back()->placeRecognitionFrame = &placeRecognitionBuffer[placeRecognitionId.getValue() - 1];
    }

    cloudSignal.notify_all();
}

Eigen::Matrix<float, 3, 3, Eigen::RowMajor> KintinuousTracker::getLastRotation() const
{
    return rmats_.at(rmats_.size() - 1);
}

std::vector<CloudSlice *> & KintinuousTracker::getCloudSlices()
{
    return sharedCloudSlices;
}

void KintinuousTracker::setOverlap(int overlap)
{
    this->overlap = overlap;
}

CloudSlice * KintinuousTracker::getLiveTsdf()
{
    return liveTsdf;
}

CloudSlice * KintinuousTracker::getLiveImage()
{
    return liveImage;
}

void KintinuousTracker::vWrapCopyUpdate()
{
	vWrapCopy = voxelWrap;

	if(vWrapCopy.x < 0)
		vWrapCopy.x = VOLUME_X - ((-vWrapCopy.x) % VOLUME_X);
	if(vWrapCopy.y < 0)
		vWrapCopy.y = VOLUME_Y - ((-vWrapCopy.y) % VOLUME_Y);
	if(vWrapCopy.z < 0)
		vWrapCopy.z = VOLUME_Z - ((-vWrapCopy.z) % VOLUME_Z);
}

void KintinuousTracker::mutexOutLiveTsdf()
{
    cloud_buffer_ = tsdf_volume_->fetchCloud(cloud_device_,
                                             vWrapCopy,
                                             color_volume_->data(),
                                             0, VOLUME_X,
                                             0, VOLUME_Y,
                                             0, VOLUME_Z,
                                             voxelWrap,
                                             1);

	pcl::PointCloud<pcl::PointXYZRGB> * newCloud = new pcl::PointCloud<pcl::PointXYZRGB>();

    cloud_buffer_.download(newCloud->points);

    boost::mutex::scoped_lock tsdfLock(tsdfMutex);

    tsdfAvailable = true;

	if(liveTsdf)
		delete liveTsdf;

	liveTsdf = new CloudSlice(newCloud,
							  CloudSlice::TSDF,
							  lastOdometry,
							  currentGlobalCamera,
							  rmats_.at(rmats_.size() - 1),
							  current_utime,
							  Stopwatch::getCurrentSystemTime(),
							  0);

    tsdfLock.unlock();
}

void KintinuousTracker::mutexOutLiveImage()
{
    int cols;
    getImage();

    modelColor.download(modelHost, cols);
    unsigned char * tsdfImageColor = new unsigned char[Resolution::get().numPixels() * 3];
    memcpy(tsdfImageColor, &modelHost[0], Resolution::get().numPixels() * 3);

    modelSurface.download(modelHost, cols);
    unsigned char * tsdfImage = new unsigned char[Resolution::get().numPixels() * 3];
    memcpy(tsdfImage, &modelHost[0], Resolution::get().numPixels() * 3);

    boost::mutex::scoped_lock imageLock(imageMutex);

    imageAvailable = true;

    if(liveImage)
        delete liveImage;

    liveImage = new CloudSlice(0,
                               CloudSlice::TSDF,
                               lastOdometry,
                               currentGlobalCamera,
                               rmats_.at(rmats_.size() - 1),
                               current_utime,
                               Stopwatch::getCurrentSystemTime(),
                               lastRgbImage,
                               tsdfImageColor,
                               tsdfImage,
                               lastDepthData);

    imageLock.unlock();
}

void KintinuousTracker::mutexOutCloudBuffer(float3 & device_tcurr,
                                            int3 voxelTrans,
                                            PlaceRecognitionInput * placeRecognitionFrame)
{
	Eigen::Vector3f voxelSize = tsdf_volume_->getVoxelSize();

	pcl::PointCloud<pcl::PointXYZRGB> * newCloud = 0;

    newCloud = new pcl::PointCloud<pcl::PointXYZRGB>();

    cloud_buffer_.download(newCloud->points);

	boost::mutex::scoped_lock lock(cloudMutex);

	Eigen::Vector3f voxelTransSize;
	voxelTransSize(0) = voxelSize(0) * voxelTrans.x;
	voxelTransSize(1) = voxelSize(1) * voxelTrans.y;
	voxelTransSize(2) = voxelSize(2) * voxelTrans.z;

	tvecs_.at(tvecs_.size() - 1) -= voxelTransSize;

    cycledMutex = true;

    CloudSlice::Dimension direction = voxelTrans.x > 0 ? CloudSlice::XPlus  :
                                      voxelTrans.x < 0 ? CloudSlice::XMinus :
                                      voxelTrans.y > 0 ? CloudSlice::YPlus  :
                                      voxelTrans.y < 0 ? CloudSlice::YMinus :
                                      voxelTrans.z > 0 ? CloudSlice::ZPlus  :
                                                         CloudSlice::ZMinus;

    sharedCloudSlices.push_back(new CloudSlice(newCloud,
                                               direction,
                                               lastOdometry,
                                               currentGlobalCamera,
                                               rmats_.at(rmats_.size() - 1),
                                               current_utime,
                                               lagTime,
                                               0,
                                               0,
                                               0,
                                               0,
                                               placeRecognitionFrame));

	voxelWrap.x += voxelTrans.x;
	voxelWrap.y += voxelTrans.y;
	voxelWrap.z += voxelTrans.z;

	cloudSignal.notify_all();

	device_tcurr.x -= voxelTransSize(0);
	device_tcurr.y -= voxelTransSize(1);
	device_tcurr.z -= voxelTransSize(2);
}

Eigen::Vector3f KintinuousTracker::rodrigues2(const Eigen::Matrix3f& matrix)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

    double rx = R(2, 1) - R(1, 2);
    double ry = R(0, 2) - R(2, 0);
    double rz = R(1, 0) - R(0, 1);

    double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
    double c = (R.trace() - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;

    double theta = acos(c);

    if( s < 1e-5 )
    {
        double t;

        if( c > 0 )
            rx = ry = rz = 0;
        else
        {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
                rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    }
    else
    {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
    }
    return Eigen::Vector3d(rx, ry, rz).cast<float>();
}
