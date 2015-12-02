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

#ifndef LOGREADER_H_
#define LOGREADER_H_

#include <string>
#include <zlib.h>
#include <poll.h>
#include <opencv2/opencv.hpp>

#include "ConfigArgs.h"
#include "../frontend/Resolution.h"
#include "ThreadDataPack.h"

class LogReader
{
    public:
        LogReader()
         : decompressedDepth(0),
           decompressedImage(0),
           compressedDepth(0),
           compressedImage(0),
           decompressionBuffer(0),
           deCompImage(0)
        {}

        virtual ~LogReader()
        {}

        virtual bool grabNext(bool & returnVal, int & currentFrame) = 0;

        unsigned short * decompressedDepth;
        unsigned char * decompressedImage;

        unsigned char * compressedDepth;
        unsigned char * compressedImage;
        int32_t compressedDepthSize;
        int32_t compressedImageSize;

        int64_t timestamp;
        bool isCompressed;

        Bytef * decompressionBuffer;
        IplImage * deCompImage;
};

#endif /* LOGREADER_H_ */
