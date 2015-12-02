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


#include "TrackerInterface.h"

TrackerInterface::TrackerInterface(LogReader * logRead, cv::Mat * depthIntrinsics)
 : ThreadObject("TrackerInterfaceThread"),
   endRequested(false),
   logRead(logRead),
   currentFrame(0),
   firstRun(true)
{
    frontend = new KintinuousTracker(depthIntrinsics);
    reset();
}

TrackerInterface::~TrackerInterface()
{
    delete frontend;
}

void TrackerInterface::reset()
{
    currentFrame = 0;
    frontend->reset();
}

bool inline TrackerInterface::process()
{
    if(firstRun)
    {
        cudaSafeCall(cudaSetDevice(ConfigArgs::get().gpu));
        firstRun = false;
    }

    if(!threadPack.pauseCapture.getValue())
    {
        TICK(threadIdentifier);

        uint64_t start = Stopwatch::getCurrentSystemTime();

        bool returnVal = true;

        bool shouldEnd = endRequested.getValue();

        if(!logRead->grabNext(returnVal, currentFrame) || shouldEnd)
        {
            threadPack.pauseCapture.assignValue(true);
            threadPack.finalised.assignValue(true);

            finalise();

            while(!threadPack.cloudSliceProcessorFinished.getValueWait())
            {
                frontend->cloudSignal.notify_all();
            }

            return shouldEnd ? false : returnVal;
        }

        depth.data = (unsigned short *)logRead->decompressedDepth;
        rgb24.data = (PixelRGB *)logRead->decompressedImage;
        
        currentFrame++;

        depth.step = Resolution::get().width() * 2;
        depth.rows = Resolution::get().rows();
        depth.cols = Resolution::get().cols();

        rgb24.step = Resolution::get().width() * 3;
        rgb24.rows = Resolution::get().rows();
        rgb24.cols = Resolution::get().cols();

        depth_device.upload(depth.data, depth.step, depth.rows, depth.cols);
        colors_device.upload(rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);

        TICK("processFrame");
        frontend->processFrame(depth_device,
                               colors_device,
                               logRead->decompressedImage,
                               logRead->decompressedDepth,
                               logRead->timestamp,
                               logRead->isCompressed,
                               logRead->compressedDepth,
                               logRead->compressedDepthSize,
                               logRead->compressedImage,
                               logRead->compressedImageSize);
        TOCK("processFrame");

        uint64_t duration = Stopwatch::getCurrentSystemTime() - start;

        if(threadPack.limit.getValue() && duration < 33333)
        {
            int sleepTime = std::max(int(33333 - duration), 0);
            usleep(sleepTime);
        }

        TOCK(threadIdentifier);
    }
    
    return true;
}
