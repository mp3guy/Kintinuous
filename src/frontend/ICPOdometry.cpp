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

#include "ICPOdometry.h"

ICPOdometry::ICPOdometry(std::vector<Eigen::Vector3f> & tvecs_,
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
   lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
   distThres_(distThresh),
   angleThres_ (angleThresh)
{
    sumDataSE3.create(MAX_THREADS);
    outDataSE3.create(1);

    icp_iterations_[0] = 10;
    icp_iterations_[1] = 5;
    icp_iterations_[2] = 4;
    icp_iterations_[3] = 0;

    if(ConfigArgs::get().fastOdometry)
    {
        icp_iterations_[0] = 0;
        icp_iterations_[1] = 10;
        icp_iterations_[2] = 5;
        icp_iterations_[3] = 0;
    }
}

ICPOdometry::~ICPOdometry()
{

}

void ICPOdometry::reset()
{
    return;
}

CloudSlice::Odometry ICPOdometry::getIncrementalTransformation(Eigen::Vector3f & trans,
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

    cv::Mat resultRt = cv::Mat::eye(4, 4, CV_64FC1);
    cv::Mat currRt;

    for (int level_index = ICPOdometry::LEVELS - 1; level_index >= 0; --level_index)
    {
        int iter_num = icp_iterations_[level_index];

        DeviceArray2D<float> & vmap_curr = vmaps_curr_[level_index];
        DeviceArray2D<float> & nmap_curr = nmaps_curr_[level_index];

        DeviceArray2D<float> & vmap_g_prev = vmaps_g_prev_[level_index];
        DeviceArray2D<float> & nmap_g_prev = nmaps_g_prev_[level_index];

        for(int iter = 0; iter < iter_num; ++iter)
        {
            Mat33 &  device_Rcurr = device_cast<Mat33> (Rcurr);
            float3 & device_tcurr = device_cast<float3>(tcurr);

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            float residual[2];

            icpStep(device_Rcurr,
                    device_tcurr,
                    vmap_curr,
                    nmap_curr,
                    device_Rprev_inv,
                    device_tprev,
                    intr(level_index),
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

            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
            Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

            lastA = dA_icp;
            Eigen::Matrix<double, 6, 1> result = dA_icp.ldlt().solve(db_icp);

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

    trans = tcurr;
    rot = Rcurr;

    return CloudSlice::ICP;
}

Eigen::MatrixXd ICPOdometry::getCovariance()
{
    return lastA.cast<double>().lu().inverse();
}
