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

#ifndef RGBDODOMETRY_H_
#define RGBDODOMETRY_H_

#include "../utils/Stopwatch.h"
#include "OdometryProvider.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vector_types.h>
#include "cuda/internal.h"
#include "CloudSlice.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include "ColorVolume.h"
#include "TSDFVolume.h"

class RGBDOdometry : public OdometryProvider
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        RGBDOdometry(std::vector<Eigen::Vector3f> & tvecs_,
                     std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor> > & rmats_,
                     std::vector<DeviceArray2D<float> > & vmaps_g_prev_,
                     std::vector<DeviceArray2D<float> > & nmaps_g_prev_,
                     std::vector<DeviceArray2D<float> > & vmaps_curr_,
                     std::vector<DeviceArray2D<float> > & nmaps_curr_,
                     Intr & intr,
                     float distThresh = 0.10f,
                     float angleThresh = sin(20.f * 3.14159254f / 180.f));

        virtual ~RGBDOdometry();

        CloudSlice::Odometry getIncrementalTransformation(Eigen::Vector3f & trans,
                                                          Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                                          const DeviceArray2D<unsigned short> & depth,
                                                          const DeviceArray2D<PixelRGB> & image,
                                                          uint64_t timestamp,
                                                          unsigned char * rgbImage,
                                                          unsigned short * depthData);

        Eigen::MatrixXd getCovariance();

        void firstRun(const DeviceArray2D<unsigned short> & depth,
                      const DeviceArray2D<PixelRGB> & image);

        void reset();

    private:
        void populateRGBDData(const DeviceArray2D<unsigned short> & depth,
                              const DeviceArray2D<PixelRGB> & image,
                              DeviceArray2D<float> * destDepths,
                              DeviceArray2D<unsigned char> * destImages);

        std::vector<Eigen::Vector3f> & tvecs_;
        std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor> > & rmats_;

        std::vector<DeviceArray2D<float> > & vmaps_g_prev_;
        std::vector<DeviceArray2D<float> > & nmaps_g_prev_;

        std::vector<DeviceArray2D<float> > & vmaps_curr_;
        std::vector<DeviceArray2D<float> > & nmaps_curr_;

        Intr & intr;

        DeviceArray<JtJJtrSE3> sumDataSE3;
        DeviceArray<JtJJtrSE3> outDataSE3;
        DeviceArray<int2> sumResidualRGB;

        const int SOBEL_SIZE;
        const double SOBEL_SCALE;
        const double MAX_DEPTH_DELTA;
        const double MAX_DEPTH;

        std::vector<int2> pyrDims;

        static const int NUM_PYRS = 4;

        DeviceArray2D<float> lastDepth[NUM_PYRS];
        DeviceArray2D<unsigned char> lastImage[NUM_PYRS];

        DeviceArray2D<float> nextDepth[NUM_PYRS];
        DeviceArray2D<unsigned char> nextImage[NUM_PYRS];

        DeviceArray2D<short> nextdIdx[NUM_PYRS];
        DeviceArray2D<short> nextdIdy[NUM_PYRS];

        DeviceArray2D<DataTerm> corresImg[NUM_PYRS];

        DeviceArray2D<float3> pointClouds[NUM_PYRS];

        IntrDoublePrecision intrinsics;

        std::vector<int> iterations;
        std::vector<float> minimumGradientMagnitudes;

        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> lastA;

        float distThres_;
        float angleThres_;

        Eigen::Matrix<double, 6, 6> lastCov;
        cv::Mat resultRtICP;
        cv::Mat resultRtRGBD;
};

#endif /* RGBDODOMETRY_H_ */
