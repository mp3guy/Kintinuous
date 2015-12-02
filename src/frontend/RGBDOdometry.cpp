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

#include "RGBDOdometry.h"

RGBDOdometry::RGBDOdometry(std::vector<Eigen::Vector3f> & tvecs_,
                           std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor> > & rmats_,
                           std::vector<DeviceArray2D<float> > & vmaps_g_prev_,
                           std::vector<DeviceArray2D<float> > & nmaps_g_prev_,
                           std::vector<DeviceArray2D<float> > & vmaps_curr_,
                           std::vector<DeviceArray2D<float> > & nmaps_curr_,
                           Intr & intr,
                           float distThresh,
                           float angleThresh)
: tvecs_(tvecs_),
  rmats_(rmats_),
  vmaps_g_prev_(vmaps_g_prev_),
  nmaps_g_prev_(nmaps_g_prev_),
  vmaps_curr_(vmaps_curr_),
  nmaps_curr_(nmaps_curr_),
  intr(intr),
  SOBEL_SIZE(3),
  SOBEL_SCALE(1.0 / pow(2.0, SOBEL_SIZE)),
  MAX_DEPTH_DELTA(0.07),
  MAX_DEPTH(6.0),
  lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
  distThres_(distThresh),
  angleThres_ (angleThresh)
{
    sumDataSE3.create(MAX_THREADS);
    outDataSE3.create(1);
    sumResidualRGB.create(MAX_THREADS);

    for(int i = 0; i < NUM_PYRS; i++)
    {
        int2 nextDim = {Resolution::get().rows() >> i, Resolution::get().cols() >> i};
        pyrDims.push_back(nextDim);
    }

    for(int i = 0; i < NUM_PYRS; i++)
    {
        lastDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        lastImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextdIdx[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextdIdy[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        pointClouds[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        corresImg[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    }

    intrinsics.fx = intr.fx;
    intrinsics.fy = intr.fy;
    intrinsics.cx = intr.cx;
    intrinsics.cy = intr.cy;

    iterations.reserve(NUM_PYRS);

    if(!ConfigArgs::get().useRGBDICP)
    {
        iterations[0] = 10;
        iterations[1] = 7;
        iterations[2] = 7;
        iterations[3] = 7;

        if(ConfigArgs::get().fastOdometry)
        {
            iterations[0] = 0;
            iterations[1] = 10;
            iterations[2] = 7;
            iterations[3] = 0;
        }
    }
    else
    {
        iterations[0] = 10;
        iterations[1] = 5;
        iterations[2] = 4;
        iterations[3] = 0;

        if(ConfigArgs::get().fastOdometry)
        {
            iterations[0] = 0;
            iterations[1] = 10;
            iterations[2] = 7;
            iterations[3] = 0;
        }
    }

    minimumGradientMagnitudes.reserve(NUM_PYRS);
    minimumGradientMagnitudes[0] = 12;
    minimumGradientMagnitudes[1] = 5;
    minimumGradientMagnitudes[2] = 3;
    minimumGradientMagnitudes[3] = 1;
}

RGBDOdometry::~RGBDOdometry()
{
    for(int i = 0; i < NUM_PYRS; i++)
    {
        lastDepth[i].release();
        lastImage[i].release();

        nextDepth[i].release();
        nextImage[i].release();

        nextdIdx[i].release();
        nextdIdy[i].release();

        pointClouds[i].release();

        corresImg[i].release();
    }
}

void RGBDOdometry::reset()
{
    return;
}

void RGBDOdometry::populateRGBDData(const DeviceArray2D<unsigned short> & depth,
                                    const DeviceArray2D<PixelRGB> & image,
                                    DeviceArray2D<float> * destDepths,
                                    DeviceArray2D<unsigned char> * destImages)
{
    shortDepthToMetres(depth, destDepths[0], MAX_DEPTH * 1000);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownGaussF(destDepths[i], destDepths[i + 1]);
    }

    imageBGRToIntensity(image, destImages[0]);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownUcharGauss(destImages[i], destImages[i + 1]);
    }
}

void RGBDOdometry::firstRun(const DeviceArray2D<unsigned short> & depth, const DeviceArray2D<PixelRGB> & image)
{
    populateRGBDData(depth, image, &lastDepth[0], &lastImage[0]);
}

CloudSlice::Odometry RGBDOdometry::getIncrementalTransformation(Eigen::Vector3f & trans,
                                                                Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                                                const DeviceArray2D<unsigned short> & depth,
                                                                const DeviceArray2D<PixelRGB> & image,
                                                                uint64_t timestamp,
                                                                unsigned char * rgbImage,
                                                                unsigned short * depthData)
{
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rmats_[rmats_.size() - 1];
    Eigen::Vector3f tprev = tvecs_[tvecs_.size() - 1];
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
    Eigen::Vector3f tcurr = tprev;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
    Mat33 & device_Rprev_inv = device_cast<Mat33>(Rprev_inv);
    float3& device_tprev = device_cast<float3>(tprev);

    populateRGBDData(depth, image, &nextDepth[0], &nextImage[0]);

    for(int i = 0; i < NUM_PYRS; i++)
    {
        sobelGaussian(nextImage[i], nextdIdx[i], nextdIdy[i]);
    }

    cv::Mat resultRt = cv::Mat::eye(4, 4, CV_64FC1);
    resultRtICP = cv::Mat::eye(4, 4, CV_64FC1);
    resultRtRGBD = cv::Mat::eye(4, 4, CV_64FC1);

    cv::Mat currRt;

    for(int i = NUM_PYRS - 1; i >= 0; i--)
    {
        projectToPointCloud(lastDepth[i], pointClouds[i], intrinsics, i);

        IntrDoublePrecision intrinsicsLevel = intrinsics(i);

        cv::Mat K = cv::Mat::zeros(3, 3, CV_64FC1);

        K.at<double>(0, 0) = intrinsicsLevel.fx;
        K.at<double>(1, 1) = intrinsicsLevel.fy;
        K.at<double>(0, 2) = intrinsicsLevel.cx;
        K.at<double>(1, 2) = intrinsicsLevel.cy;
        K.at<double>(2, 2) = 1;

        cv::Mat Kinv = K.inv();

        for(int j = 0; j < iterations[i]; j++)
        {
            cv::Mat Rt = resultRt.inv(cv::DECOMP_SVD);

            cv::Mat R = Rt(cv::Rect(0, 0, 3, 3)).clone();

            cv::Mat KRK_inv = K * R * K.inv();

            cv::Mat Kt = Rt(cv::Rect(3, 0, 1, 3)).clone();
            Kt = K * Kt;
            const double * Kt_ptr = reinterpret_cast<const double *>(Kt.ptr());

            float3 kt = {(float)Kt_ptr[0], (float)Kt_ptr[1], (float)Kt_ptr[2]};

            const double * KRK_inv_ptr = reinterpret_cast<const double *>(KRK_inv.ptr());

            Mat33 krkInv;
            for(int n = 0; n < 9; n++)
            {
                ((float *)krkInv.data)[n] = KRK_inv_ptr[n];
            }

            int sigma = 0;
            int rgbSize = 0;

            computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(SOBEL_SCALE, 2.0),
                               nextdIdx[i],
                               nextdIdy[i],
                               lastDepth[i],
                               nextDepth[i],
                               lastImage[i],
                               nextImage[i],
                               corresImg[i],
                               sumResidualRGB,
                               MAX_DEPTH_DELTA,
                               kt,
                               krkInv,
                               sigma,
                               rgbSize,
                               128,
                               256);

            float sigmaVal = std::sqrt((float)sigma / rgbSize == 0 ? 1 : rgbSize);

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            float residual[2];

            if(ConfigArgs::get().useRGBDICP)
            {
                Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
                float3& device_tcurr = device_cast<float3>(tcurr);

                DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
                DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];

                DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
                DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

                icpStep(device_Rcurr,
                        device_tcurr,
                        vmap_curr,
                        nmap_curr,
                        device_Rprev_inv,
                        device_tprev,
                        intr(i),
                        vmap_g_prev,
                        nmap_g_prev,
                        distThres_,
                        angleThres_,
                        sumDataSE3,
                        outDataSE3,
                        A_icp.data(),
                        b_icp.data(),
                        &residual[0],
                        128,
                        64);
            }

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
            Eigen::Matrix<float, 6, 1> b_rgbd;

            rgbStep(corresImg[i],
                    sigmaVal,
                    pointClouds[i],
                    intr(i).fx,
                    intr(i).fy,
                    nextdIdx[i],
                    nextdIdy[i],
                    SOBEL_SCALE,
                    sumDataSE3,
                    outDataSE3,
                    A_rgbd.data(),
                    b_rgbd.data(),
                    128,
                    64);

            Eigen::Matrix<double, 6, 1> result;

            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd = A_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
            Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

            if(ConfigArgs::get().useRGBDICP)
            {
                double w = 10;
                lastA = dA_rgbd + w * w * dA_icp;
                result = lastA.ldlt().solve(db_rgbd + w * db_icp);
            }
            else
            {
                lastA = dA_rgbd;
                result = lastA.ldlt().solve(db_rgbd);
            }

            cv::Mat sln = cv::Mat::zeros(6, 1, CV_64FC1);

            sln.at<double>(0, 0) = result(0, 0);
            sln.at<double>(1, 0) = result(1, 0);
            sln.at<double>(2, 0) = result(2, 0);
            sln.at<double>(3, 0) = result(3, 0);
            sln.at<double>(4, 0) = result(4, 0);
            sln.at<double>(5, 0) = result(5, 0);

            OdometryProvider::computeProjectiveMatrix(sln, currRt);

            resultRt = currRt * resultRt;

            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotation;
            rotation(0, 0) = resultRt.at<double>(0, 0);
            rotation(0, 1) = resultRt.at<double>(0, 1);
            rotation(0, 2) = resultRt.at<double>(0, 2);

            rotation(1, 0) = resultRt.at<double>(1, 0);
            rotation(1, 1) = resultRt.at<double>(1, 1);
            rotation(1, 2) = resultRt.at<double>(1, 2);

            rotation(2, 0) = resultRt.at<double>(2, 0);
            rotation(2, 1) = resultRt.at<double>(2, 1);
            rotation(2, 2) = resultRt.at<double>(2, 2);

            Eigen::Vector3f translation;
            translation(0) = resultRt.at<double>(0, 3);
            translation(1) = resultRt.at<double>(1, 3);
            translation(2) = resultRt.at<double>(2, 3);

            Eigen::Isometry3f rgbOdom;

            rgbOdom.setIdentity();
            rgbOdom.rotate(rotation);
            rgbOdom.translation() = translation;

            Eigen::Isometry3f currentT;
            currentT.setIdentity();
            currentT.rotate(Rprev);
            currentT.translation() = tprev;

            currentT = currentT * rgbOdom.inverse();

            tcurr = currentT.translation();
            Rcurr = currentT.rotation();
        }
    }

    for(int i = 0; i < NUM_PYRS; i++)
    {
        std::swap(lastDepth[i], nextDepth[i]);
        std::swap(lastImage[i], nextImage[i]);
    }

    if((tcurr - tprev).norm() > 0.3)
    {
        Rcurr = Rprev;
        tcurr = tprev;
    }

    trans = tcurr;
    rot = Rcurr;

    return CloudSlice::RGBD;
}

Eigen::MatrixXd RGBDOdometry::getCovariance()
{
    return lastA.cast<double>().lu().inverse();
}
