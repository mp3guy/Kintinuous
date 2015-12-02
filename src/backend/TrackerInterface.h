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

#ifndef TRACKERINTERFACE_H_
#define TRACKERINTERFACE_H_

#include <poll.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "../frontend/Volume.h"
#include "../frontend/KintinuousTracker.h"

#include "../utils/ThreadObject.h"
#include "../utils/LogReader.h"

class TrackerInterface : public ThreadObject
{
    public:
        TrackerInterface(LogReader * logRead, cv::Mat * depthIntrinsics);

        virtual ~TrackerInterface();

        void reset();

        KintinuousTracker * getFrontend()
        {
            return frontend;
        }

        void finalise()
        {
            frontend->finalise();
        }

        void setPark(const bool park)
        {
            frontend->setParked(park);
        }

        void enableOverlap()
        {
            frontend->setOverlap(2);
        }

        void loadTrajectory(const std::string & filename)
        {
            frontend->loadTrajectory(filename);
        }

        ThreadMutexObject<bool> endRequested;

    private:
        bool inline process();

        PtrStepSz<const unsigned short> depth;
        PtrStepSz<const PixelRGB> rgb24;

        LogReader * logRead;

        KintinuousTracker * frontend;
        DeviceArray2D<unsigned short> depth_device;
        DeviceArray2D<PixelRGB> colors_device;
        
        int currentFrame;
        bool firstRun;
};

#endif /* TRACKERINTERFACE_H_ */
