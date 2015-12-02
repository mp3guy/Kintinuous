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

#include "DepthCamera.h"

DepthCamera::DepthCamera(cv::Mat * calibration)
 : depthIntrinsics(calibration),
   principalPoint(depthIntrinsics->at<double>(0, 2), depthIntrinsics->at<double>(1, 2))
{
}

DepthCamera::~DepthCamera()
{
}

void DepthCamera::projectInlierMatches(std::vector<std::pair<int2, int2> > inliers,
                                       std::vector<Eigen::Vector3d> & inl1,
                                       std::vector<Eigen::Vector3d> & inl2,
                                       unsigned short * depth1,
                                       unsigned short * depth2)
{
    for(unsigned int i = 0; i < inliers.size(); i++)
    {
        float depthFirst = (float)depth1[inliers.at(i).first.y * Resolution::get().width() + inliers.at(i).first.x] / 1000.f;
        float depthSecond = (float)depth2[inliers.at(i).second.y * Resolution::get().width() + inliers.at(i).second.x] / 1000.f;

        if(!depthFirst || !depthSecond)
        {
            continue;
        }

        Eigen::Vector3d firstProj(depthFirst * (inliers.at(i).first.x - principalPoint.x) * (1 / depthIntrinsics->at<double>(0, 0)),
                                  depthFirst * (inliers.at(i).first.y - principalPoint.y) * (1 / depthIntrinsics->at<double>(1, 1)),
                                  depthFirst);

        Eigen::Vector3d secondProj(depthSecond * (inliers.at(i).second.x - principalPoint.x) * (1 / depthIntrinsics->at<double>(0, 0)),
                                   depthSecond * (inliers.at(i).second.y - principalPoint.y) * (1 / depthIntrinsics->at<double>(1, 1)),
                                   depthSecond);

        inl1.push_back(firstProj);
        inl2.push_back(secondProj);
    }
}

void DepthCamera::computeVertexMap(cv::Mat & depthMap, cv::Mat & vertexMap)
{
    vertexMap.create(depthMap.rows, depthMap.cols, CV_32FC3);

    Eigen::Isometry3f m;
    m.setIdentity();
    m(0,0) = depthIntrinsics->at<double>(0,0);
    m(0,2) = depthIntrinsics->at<double>(0,2);
    m(1,1) = depthIntrinsics->at<double>(1,1);
    m(1,2) = depthIntrinsics->at<double>(1,2);

    Eigen::Projective3f K = m;
    Eigen::Projective3f Kinv = K.inverse();

    for(int row = 0; row < depthMap.rows; row++)
    {
        unsigned short * depthRow = (unsigned short *)depthMap.ptr(row);

        float * vertexRow = (float *)vertexMap.ptr(row);

        for(int col = 0; col < depthMap.cols; col++)
        {
            if(depthRow[col] == 0)
            {
                vertexRow[col * 3] = 100000;
                vertexRow[col * 3 + 1] = 100000;
                vertexRow[col * 3 + 2] = 100000;
            }
            else
            {
                Eigen::Vector4f vertex;
                vertex(0) = col * (float)depthRow[col];
                vertex(1) = row * (float)depthRow[col];
                vertex(2) = (float)depthRow[col];
                vertex(3) = 1;

                Eigen::Vector4f pixel;

                pixel = Kinv * vertex;

                vertexRow[col * 3] = pixel(0) / 1000.0;
                vertexRow[col * 3 + 1] = pixel(1) / 1000.0;
                vertexRow[col * 3 + 2] = pixel(2) / 1000.0;
            }
        }
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr DepthCamera::convertToXYZPointCloud(unsigned short * depth_image, float maxDist)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZ>);

    for(int i = 0; i < Resolution::get().width(); i++)
    {
        for(int j = 0; j < Resolution::get().height(); j++)
        {
            if(depth_image[j * Resolution::get().width() + i] != 0 && depth_image[j * Resolution::get().width() + i] < maxDist * 1000)
            {
                pcl::PointXYZ pt;
                pt.z = depth_image[j * Resolution::get().width() + i] * 0.001f;
                pt.x = (static_cast<float>(i) - depthIntrinsics->at<double>(0, 2)) * pt.z * (1.0f / depthIntrinsics->at<double>(0, 0));
                pt.y = (static_cast<float>(j) - depthIntrinsics->at<double>(1, 2)) * pt.z * (1.0f / depthIntrinsics->at<double>(1, 1));
                cloud->push_back(pt);
            }
        }
    }

    return cloud;
}
