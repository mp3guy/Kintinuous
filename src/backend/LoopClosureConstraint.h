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

#ifndef LOOPCLOSURECONSTRAINT_H_
#define LOOPCLOSURECONSTRAINT_H_

#include <Eigen/Core>

class LoopClosureConstraint
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        LoopClosureConstraint(uint64_t time1,
                              uint64_t time2,
                              Eigen::Matrix4f & constraint,
                              std::vector<Eigen::Vector3d> & inliers1Proj,
                              std::vector<Eigen::Vector3d> & inliers2Proj,
                              uint64_t lagTime)
         : time1(time1),
           time2(time2),
           constraint(constraint.cast<double>()),
           inliers1Proj(inliers1Proj),
           inliers2Proj(inliers2Proj),
           lagTime(lagTime)
        {}

        uint64_t time1;
        uint64_t time2;
        Eigen::Matrix4d constraint; //Expected to NOT be in iSAM basis
        std::vector<Eigen::Vector3d> inliers1Proj;
        std::vector<Eigen::Vector3d> inliers2Proj;
        uint64_t lagTime;
};

#endif /* LOOPCLOSURECONSTRAINT_H_ */
