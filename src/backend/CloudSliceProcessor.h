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

#ifndef CLOUDSLICEPROCESSOR_H_
#define CLOUDSLICEPROCESSOR_H_

#include "../frontend/Volume.h"
#include "../utils/ThreadObject.h"

#include <sstream>
#include <map>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/filesystem.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>

class CloudSliceProcessor : public ThreadObject
{
    public:
        CloudSliceProcessor();

        virtual ~CloudSliceProcessor();

        void reset();

        void save();

    private:
        bool inline process();

        int latestPushedCloud;
        int numClouds;
        bool cycledMutex;
};

#endif /* CLOUDSLICEPROCESSOR_H_ */
