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

#include "RawLogReader.h"

RawLogReader::RawLogReader()
{
    assert(boost::filesystem::exists(ConfigArgs::get().logFile.c_str()));

    fp = fopen(ConfigArgs::get().logFile.c_str(), "rb");

    currentFrame = 0;

    assert(fread(&numFrames, sizeof(int32_t), 1, fp));

    compressedDepth = new unsigned char[Resolution::get().numPixels() * 2];
    compressedImage = new unsigned char[Resolution::get().numPixels() * 3];

    decompressionBuffer = new Bytef[Resolution::get().numPixels() * 2];

    if(!ConfigArgs::get().totalNumFrames)
    {
        *const_cast<int *>(&ConfigArgs::get().totalNumFrames) = numFrames;
    }

    isCompressed = true;
}

RawLogReader::~RawLogReader()
{
    delete [] compressedDepth;
    delete [] compressedImage;
    delete [] decompressionBuffer;
    fclose(fp);
}

void RawLogReader::readNext()
{
    assert(fread(&timestamp, sizeof(int64_t), 1, fp));

    assert(fread(&compressedDepthSize, sizeof(int32_t), 1, fp));
    assert(fread(&compressedImageSize, sizeof(int32_t), 1, fp));

    assert(fread(compressedDepth, compressedDepthSize, 1, fp));

    if(compressedImageSize > 0)
    {
        assert(fread(compressedImage, compressedImageSize, 1, fp));
    }

    if(deCompImage != 0)
    {
        cvReleaseImage(&deCompImage);
    }

    CvMat tempMat = cvMat(1, compressedImageSize, CV_8UC1, (void *)compressedImage);

    if(compressedImageSize == Resolution::get().numPixels() * 3)
    {
        isCompressed = false;

        deCompImage = cvCreateImage(cvSize(Resolution::get().width(), Resolution::get().height()), IPL_DEPTH_8U, 3);

        memcpy(deCompImage->imageData, compressedImage, Resolution::get().numPixels() * 3);
    }
    else if(compressedImageSize > 0)
    {
        isCompressed = true;

        deCompImage = cvDecodeImage(&tempMat);
    }
    else
    {
        isCompressed = false;

        deCompImage = cvCreateImage(cvSize(Resolution::get().width(), Resolution::get().height()), IPL_DEPTH_8U, 3);

        memset(deCompImage->imageData, 0, Resolution::get().numPixels() * 3);
    }

    if(compressedDepthSize == Resolution::get().numPixels() * 2)
    {
        //RGB should not be compressed in this case
        assert(!isCompressed);

        memcpy(&decompressionBuffer[0], compressedDepth, Resolution::get().numPixels() * 2);
    }
    else if(compressedDepthSize > 0)
    {
        //RGB should also be compressed
        assert(isCompressed);

        unsigned long decompLength = Resolution::get().numPixels() * 2;

        uncompress(&decompressionBuffer[0], (unsigned long *)&decompLength, (const Bytef *)compressedDepth, compressedDepthSize);
    }
    else
    {
        isCompressed = false;

        memset(&decompressionBuffer[0], 0, Resolution::get().numPixels() * 2);
    }

    decompressedDepth = (unsigned short *)&decompressionBuffer[0];
    decompressedImage = (unsigned char *)deCompImage->imageData;

    if(ConfigArgs::get().flipColors)
    {
        cv::Mat3b rgb(Resolution::get().rows(),
                      Resolution::get().cols(),
                      (cv::Vec<unsigned char, 3> *)deCompImage->imageData,
                      Resolution::get().width() * 3);

        cv::cvtColor(rgb, rgb, CV_RGB2BGR);
    }

    currentFrame++;
}

bool RawLogReader::grabNext(bool & returnVal, int & currentFrame)
{
    if(hasMore() && currentFrame < ConfigArgs::get().totalNumFrames)
    {
        readNext();
        ThreadDataPack::get().trackerFrame.assignAndNotifyAll(currentFrame);
        return true;
    }
	returnVal = false;
	return false;
}

bool RawLogReader::hasMore()
{
    return currentFrame + 1 < numFrames;
}
