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

#ifndef GRAPHNODE_H_
#define GRAPHNODE_H_

#include <Eigen/Dense>

class GraphNode
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        GraphNode()
        {}

        int id;
        Eigen::Vector3d position;
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        std::vector<GraphNode *> neighbours;
};

#endif /* GRAPHNODE_H_ */
