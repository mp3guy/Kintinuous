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

#ifndef UTILS_JACOBIAN_H_
#define UTILS_JACOBIAN_H_

#include <vector>

#include "OrderedJacobianRow.h"

class Jacobian
{
    public:
        Jacobian()
         : columns(0)
        {}

        virtual ~Jacobian()
        {
            reset();
        }

        void assign(std::vector<OrderedJacobianRow*> & rows, const int columns)
        {
            reset();
            this->rows = rows;
            this->columns = columns;
        }

        int cols() const
        {
            return columns;
        }

        int nonZero() const
        {
            int count = 0;
            for(size_t i = 0; i < rows.size(); i++)
            {
                count += rows[i]->nonZeros();
            }
            return count;
        }

        std::vector<OrderedJacobianRow*> rows;

    private:
        int columns;

        void reset()
        {
            for(size_t i = 0; i < rows.size(); i++)
            {
                delete rows[i];
            }
            rows.clear();
        }
};



#endif /* UTILS_JACOBIAN_H_ */
