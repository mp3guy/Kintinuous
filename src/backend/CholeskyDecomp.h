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

#ifndef UTILS_CHOLESKYDECOMP_H_
#define UTILS_CHOLESKYDECOMP_H_

#include <cholmod.h>
#include <Eigen/Core>

#include "Jacobian.h"

class CholeskyDecomp
{
    public:
        CholeskyDecomp();
        virtual ~CholeskyDecomp();

        void freeFactor();

        Eigen::VectorXd solve(const Jacobian & jacobian, const Eigen::VectorXd & residual, const bool firstRun);

    private:
        cholmod_common Common;
        cholmod_factor * L;
};

#endif /* UTILS_CHOLESKYDECOMP_H_ */
