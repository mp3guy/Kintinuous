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

#ifndef BACKEND_PNPSOLVER_H_
#define BACKEND_PNPSOLVER_H_

#include <isam/Pose3d.h>

#include "DepthCamera.h"
#include "Surf3DTools.h"

class PNPSolver
{
    public:
        PNPSolver(DepthCamera * depthCamera);
        virtual ~PNPSolver();

        void getRelativePose(isam::Pose3d &pose,
                             std::vector<std::pair<int2, int2> > & inliers,
                             std::vector<InterestPoint *> & scene,
                             std::vector<InterestPoint *> & model);

    private:
        DepthCamera * camera;
};

#endif /* BACKEND_PNPSOLVER_H_ */
