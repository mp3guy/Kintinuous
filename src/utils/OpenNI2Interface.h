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

#ifndef OPENNI2INTERFACE_H_
#define OPENNI2INTERFACE_H_

#include <OpenNI.h>
#include <PS1080.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

#include "ThreadMutexObject.h"

class OpenNI2Interface
{
    public:
        OpenNI2Interface(int inWidth = 640, int inHeight = 480, int fps = 30);
        virtual ~OpenNI2Interface();

        const int width, height, fps;

        void printModes();
        bool findMode(int x, int y, int fps);
        void setAutoExposure(bool value);
        void setAutoWhiteBalance(bool value);
        bool getAutoExposure();
        bool getAutoWhiteBalance();

        bool ok()
        {
            return initSuccessful;
        }

        std::string error()
        {
            errorText.erase(std::remove_if(errorText.begin(), errorText.end(), &OpenNI2Interface::isTab), errorText.end());
            return errorText;
        }

        static const int numBuffers = 10;
        ThreadMutexObject<int> latestDepthIndex;
        std::pair<std::pair<uint8_t *, uint8_t *>, int64_t> frameBuffers[numBuffers];

        class RGBCallback : public openni::VideoStream::NewFrameListener
        {
            public:
                RGBCallback(int64_t & lastRgbTime,
                            ThreadMutexObject<int> & latestRgbIndex,
                            std::pair<uint8_t *, int64_t> * rgbBuffers)
                 : lastRgbTime(lastRgbTime),
                   latestRgbIndex(latestRgbIndex),
                   rgbBuffers(rgbBuffers)
                {}

                virtual ~RGBCallback() {}

                void onNewFrame(openni::VideoStream& stream)
                {
                    stream.readFrame(&frame);

                    boost::posix_time::ptime time = boost::posix_time::microsec_clock::local_time();
                    boost::posix_time::time_duration duration(time.time_of_day());
                    lastRgbTime = duration.total_microseconds();

                    int bufferIndex = (latestRgbIndex.getValue() + 1) % numBuffers;

                    memcpy(rgbBuffers[bufferIndex].first, frame.getData(), frame.getWidth() * frame.getHeight() * 3);

                    rgbBuffers[bufferIndex].second = lastRgbTime;

                    latestRgbIndex++;
                }

            private:
                openni::VideoFrameRef frame;
                int64_t & lastRgbTime;
                ThreadMutexObject<int> & latestRgbIndex;
                std::pair<uint8_t *, int64_t> * rgbBuffers;
        };

        class DepthCallback : public openni::VideoStream::NewFrameListener
        {
            public:
                DepthCallback(int64_t & lastDepthTime,
                              ThreadMutexObject<int> & latestDepthIndex,
                              ThreadMutexObject<int> & latestRgbIndex,
                              std::pair<uint8_t *, int64_t> * rgbBuffers,
                              std::pair<std::pair<uint8_t *, uint8_t *>, int64_t> * frameBuffers)
                 : lastDepthTime(lastDepthTime),
                   latestDepthIndex(latestDepthIndex),
                   latestRgbIndex(latestRgbIndex),
                   rgbBuffers(rgbBuffers),
                   frameBuffers(frameBuffers)
                {}

                virtual ~DepthCallback() {}

                void onNewFrame(openni::VideoStream& stream)
                {
                    stream.readFrame(&frame);

                    boost::posix_time::ptime time = boost::posix_time::microsec_clock::local_time();
                    boost::posix_time::time_duration duration(time.time_of_day());
                    lastDepthTime = duration.total_microseconds();

                    int bufferIndex = (latestDepthIndex.getValue() + 1) % numBuffers;

                    memcpy(frameBuffers[bufferIndex].first.first, frame.getData(), frame.getWidth() * frame.getHeight() * 2);

                    frameBuffers[bufferIndex].second = lastDepthTime;

                    int lastImageVal = latestRgbIndex.getValue();

                    if(lastImageVal == -1)
                    {
                        return;
                    }

                    lastImageVal %= numBuffers;

                    memcpy(frameBuffers[bufferIndex].first.second, rgbBuffers[lastImageVal].first, frame.getWidth() * frame.getHeight() * 3);

                    latestDepthIndex++;
                }

            private:
                openni::VideoFrameRef frame;
                int64_t & lastDepthTime;
                ThreadMutexObject<int> & latestDepthIndex;
                ThreadMutexObject<int> & latestRgbIndex;

                std::pair<uint8_t *, int64_t> * rgbBuffers;
                std::pair<std::pair<uint8_t *, uint8_t *>, int64_t> * frameBuffers;
        };

    private:
        openni::Device device;

        openni::VideoStream depthStream;
        openni::VideoStream rgbStream;

        //Map for formats from OpenNI2
        std::map<int, std::string> formatMap;

        int64_t lastRgbTime;
        int64_t lastDepthTime;

        ThreadMutexObject<int> latestRgbIndex;
        std::pair<uint8_t *, int64_t> rgbBuffers[numBuffers];

        RGBCallback * rgbCallback;
        DepthCallback * depthCallback;

        bool initSuccessful;
        std::string errorText;

        //For removing tabs from OpenNI's error messages
        static bool isTab(char c)
        {
            switch(c)
            {
                case '\t':
                    return true;
                default:
                    return false;
            }
        }
};

#endif /* OPENNI2INTERFACE_H_ */
