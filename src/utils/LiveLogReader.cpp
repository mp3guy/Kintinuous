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

#include "LiveLogReader.h"

LiveLogReader::LiveLogReader()
 : lastFrameTime(0),
   lastGot(-1)
{
	decompressionBuffer = new Bytef[Resolution::get().numPixels() * 2];
	deCompImage = cvCreateImage(cvSize(Resolution::get().width(), Resolution::get().height()), IPL_DEPTH_8U, 3);

    std::cout << "Creating live capture... "; std::cout.flush();

    asus = new OpenNI2Interface(Resolution::get().width(), Resolution::get().height());

    if(!asus->ok())
    {
        std::cout << "failed!" << std::endl;
        std::cout << asus->error();
        exit(0);
    }
    else
    {
        std::cout << "success!" << std::endl;

        std::cout << "Waiting for first frame"; std::cout.flush();

        int lastDepth = asus->latestDepthIndex.getValue();

        do
        {
            usleep(33333);
            std::cout << "."; std::cout.flush();
            lastDepth = asus->latestDepthIndex.getValue();
        } while(lastDepth == -1);

        std::cout << " got it!" << std::endl;
    }
}

LiveLogReader::~LiveLogReader()
{
	delete asus;
	delete [] decompressionBuffer;
	cvReleaseImage(&deCompImage);
}

bool LiveLogReader::grabNext(bool & returnVal, int & currentFrame)
{
    int lastDepth = asus->latestDepthIndex.getValue();

    if(lastDepth == -1)
    {
        return true;
    }

    int bufferIndex = lastDepth % 10;

    if(bufferIndex == lastGot)
    {
        return true;
    }

    if(lastFrameTime == asus->frameBuffers[bufferIndex].second)
    {
        return true;
    }

    memcpy(&decompressionBuffer[0], asus->frameBuffers[bufferIndex].first.first, Resolution::get().numPixels() * 2);
    memcpy(deCompImage->imageData, asus->frameBuffers[bufferIndex].first.second, Resolution::get().numPixels() * 3);

    lastFrameTime = asus->frameBuffers[bufferIndex].second;

    timestamp = lastFrameTime;
    isCompressed = false;

    decompressedImage = (unsigned char *)deCompImage->imageData;
    decompressedDepth = (unsigned short *)&decompressionBuffer[0];

    compressedImage = 0;
    compressedDepth = 0;

    compressedImageSize = Resolution::get().numPixels() * 3;
    compressedDepthSize = Resolution::get().numPixels() * 2;

    if(ConfigArgs::get().flipColors)
    {
        cv::Mat3b rgb(Resolution::get().rows(),
                      Resolution::get().cols(),
                      (cv::Vec<unsigned char, 3> *)decompressedImage,
                      Resolution::get().width() * 3);

        cv::cvtColor(rgb, rgb, CV_RGB2BGR);
    }

    ThreadDataPack::get().trackerFrame.assignAndNotifyAll(currentFrame);

    return true;
}
