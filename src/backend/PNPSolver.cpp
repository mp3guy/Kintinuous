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

#include "PNPSolver.h"

PNPSolver::PNPSolver(DepthCamera * depthCamera)
 : camera(depthCamera)
{

}

PNPSolver::~PNPSolver()
{

}

void PNPSolver::getRelativePose(isam::Pose3d &pose,
                                std::vector<std::pair<int2, int2> > & inliers,
                                std::vector<InterestPoint *> & scene,
                                std::vector<InterestPoint *> & model)
{
    assert(scene.size() == model.size());

    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;

    for(size_t i = 0; i < scene.size(); i++)
    {
        points2d.push_back(cv::Point2f(model.at(i)->u, model.at(i)->v));
        points3d.push_back(cv::Point3f(scene.at(i)->X, scene.at(i)->Y, scene.at(i)->Z));
    }

    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1); //Zero distortion: Deal with it

    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);

    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

    cv::Mat inliersCv;

    cv::solvePnPRansac(points3d,
                       points2d,
                       *camera->depthIntrinsics,
                       distCoeffs,
                       rvec,
                       tvec,
                       false,
                       500,
                       2.0f,
                       0.85,
                       inliersCv);

    cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
    cv::Rodrigues(rvec, R_matrix);

    Eigen::MatrixXd isamM(3, 3);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            isamM(i, j) = R_matrix.at<double>(i, j);
        }
    }

    isam::Rot3d R(isamM);

    pose = isam::Pose3d(tvec.at<double>(0),
                        tvec.at<double>(1),
                        tvec.at<double>(2),
                        R.yaw(),
                        R.pitch(),
                        R.roll());

    for(int i = 0; i < inliersCv.rows; ++i)
    {
        int n = inliersCv.at<int>(i);
        int2 corresp1 = {int(scene.at(n)->u), int(scene.at(n)->v)};
        int2 corresp2 = {int(model.at(n)->u), int(model.at(n)->v)};
        inliers.push_back(std::pair<int2, int2>(corresp1, corresp2));
    }
}

