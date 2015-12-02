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

#ifndef ISAMINTERFACE_H_
#define ISAMINTERFACE_H_

#include <string>
#include <map>
#include <Eigen/Dense>
#include <isam.h>
#include <stdint.h>
#include "../utils/ConfigArgs.h"

class iSAMInterface
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        iSAMInterface();
        virtual ~iSAMInterface();

        void addCameraCameraConstraint(uint64_t time1, uint64_t time2,
                                       const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & Rprev, const Eigen::Vector3f & tprev,
                                       const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & Rcurr, const Eigen::Vector3f & tcurr);

        isam::Pose3d_Pose3d_Factor * addLoopConstraint(uint64_t time1,
                                                       uint64_t time2,
                                                       Eigen::Matrix4d & loopConstraint);

        const std::list<isam::Factor* > & getFactors();

        void getCameraPositions(std::vector<std::pair<uint64_t, Eigen::Vector3d> > & positions);

        void getCameraPoses(std::vector<std::pair<uint64_t, Eigen::Matrix4f> > & poses);

        double optimise();

        Eigen::Matrix4f getCameraPose(uint64_t id);

        void removeFactor(isam::Pose3d_Pose3d_Factor * factor);

    private:
        isam::Pose3d_Node * cameraNode(uint64_t time);

        isam::Slam * _slam;
        std::map<uint64_t, isam::Pose3d_Node*> _camera_nodes;

        Eigen::Matrix4f transformation2isam;
        std::map<std::pair<uint64_t, uint64_t>, bool> cameraCameraConstraints;
        std::map<std::pair<uint64_t, uint64_t>, bool> cameraPlaneConstraints;
};

#endif /* ISAMINTERFACE_H_ */
