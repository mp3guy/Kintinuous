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

#ifndef VOLUME_H_
#define VOLUME_H_

#include "cuda/internal.h"
#include <assert.h>
#include <vector_types.h>

class Volume
{
    public:
        static Volume & get(float volumeSize = 0)
        {
            static Volume instance(volumeSize);
            return instance;
        }

        const float & getVolumeSize()
        {
            return volumeSize;
        }

        const float3 & getVoxelSizeMeters()
        {
            return voxelSizeMeters;
        }

    private:
        Volume(float inVolumeSize)
         : volumeSize(inVolumeSize)
        {
            assert(volumeSize > 0);
            voxelSizeMeters.x = volumeSize / float(VOLUME_X);
            voxelSizeMeters.y = volumeSize / float(VOLUME_Y);
            voxelSizeMeters.z = volumeSize / float(VOLUME_Z);
        }

        const float volumeSize;
        float3 voxelSizeMeters;
};

#endif /* VOLUME_H_ */
