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


#ifndef GROUNDTRUTHODOMETRY_H_
#define GROUNDTRUTHODOMETRY_H_

#include "OdometryProvider.h"

class GroundTruthOdometry : public OdometryProvider
{
    public:
        GroundTruthOdometry(std::vector<Eigen::Vector3f> & tvecs_,
                            std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor> > & rmats_,
                            std::map<uint64_t, Eigen::Isometry3f, std::less<int>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Isometry3f> > > & camera_trajectory,
                            uint64_t & last_utime);

        virtual ~GroundTruthOdometry();

        CloudSlice::Odometry getIncrementalTransformation(Eigen::Vector3f & trans,
                                                          Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                                          const DeviceArray2D<unsigned short> & depth,
                                                          const DeviceArray2D<PixelRGB> & image,
                                                          uint64_t timestamp,
                                                          unsigned char * rgbImage,
                                                          unsigned short * depthData);

        Eigen::MatrixXd getCovariance();

        void reset();

        bool preRun(unsigned char * rgbImage,
                    unsigned short * depthDatam,
                    uint64_t timestamp);

    private:
        std::vector<Eigen::Vector3f> & tvecs_;
        std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor> > & rmats_;

        std::map<uint64_t, Eigen::Isometry3f, std::less<int>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Isometry3f> > > & camera_trajectory;
        uint64_t & last_utime;
};

#endif /* GROUNDTRUTHODOMETRY_H_ */
