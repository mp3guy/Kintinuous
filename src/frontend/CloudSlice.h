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

#ifndef CLOUDSLICE_H_
#define CLOUDSLICE_H_

#include "Resolution.h"
#include "PlaceRecognitionInput.h"
#include "../utils/ThreadMutexObject.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class CloudSlice
{
	public:

		enum Dimension
		{
			XPlus, XMinus, YPlus, YMinus, ZPlus, ZMinus, FIRST, FINAL, TSDF
		};

		enum Odometry
		{
			ICP, GROUNDTRUTH, RGBD, FAIL
		};

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		CloudSlice(pcl::PointCloud<pcl::PointXYZRGB> * cloud,
				   Dimension dimension,
				   Odometry odometry,
				   Eigen::Vector3f & cameraTranslation,
				   Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & cameraRotation,
				   uint64_t utime,
				   uint64_t lagTime,
				   unsigned char * rgbImage,
				   unsigned char * tsdfImageColor = 0,
				   unsigned char * tsdfImage = 0,
				   unsigned short * depthData = 0,
				   PlaceRecognitionInput * placeRecognitionFrame = 0)
		 : cloud(cloud),
		   processedCloud(0),
		   dimension(dimension),
		   odometry(odometry),
		   cameraTranslation(cameraTranslation),
		   cameraRotation(cameraRotation),
		   poseIsam(false),
		   utime(utime),
		   lagTime(lagTime),
		   tsdfImageColor(tsdfImageColor),
		   tsdfImage(tsdfImage),
		   depthData(depthData),
		   placeRecognitionFrame(placeRecognitionFrame)
		{
			if(rgbImage != 0)
			{
				this->rgbImage = new unsigned char[Resolution::get().numPixels() * 3];
				memcpy(this->rgbImage, rgbImage, sizeof(unsigned char) * Resolution::get().numPixels() * 3);
			}
			else
			{
				this->rgbImage = 0;
			}

            if(depthData != 0)
            {
                this->depthData = new unsigned short[Resolution::get().numPixels()];
                memcpy(this->depthData, depthData, sizeof(unsigned short) * Resolution::get().numPixels());
            }
            else
            {
                this->depthData = 0;
            }
		}

		virtual ~CloudSlice()
		{
			if(cloud)
				delete cloud;

			if(processedCloud)
			    delete processedCloud;

			if(rgbImage)
				delete [] rgbImage;

			if(tsdfImageColor)
				delete [] tsdfImageColor;

			if(tsdfImage)
				delete [] tsdfImage;

			if(depthData)
				delete [] depthData;
		}

		pcl::PointCloud<pcl::PointXYZRGB> * cloud;
		pcl::PointCloud<pcl::PointXYZRGBNormal> * processedCloud;
		Dimension dimension;
		Odometry odometry;
		Eigen::Vector3f cameraTranslation;
		Eigen::Matrix<float, 3, 3, Eigen::RowMajor> cameraRotation;
		ThreadMutexObject<bool> poseIsam;
		uint64_t utime;
		uint64_t lagTime;
		unsigned char * rgbImage;
		unsigned char * tsdfImageColor;
		unsigned char * tsdfImage;
		unsigned short * depthData;
		PlaceRecognitionInput * placeRecognitionFrame;

	private:
		CloudSlice()
		{}
};

#endif /* CLOUDSLICE_H_ */
