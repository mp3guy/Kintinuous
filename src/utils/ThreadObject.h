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

#ifndef THREADOBJECT_H_
#define THREADOBJECT_H_

#include <assert.h>
#include "ThreadDataPack.h"
#include "Stopwatch.h"

class ThreadObject
{
    public:
        ThreadObject(std::string threadIdentifier)
         : threadPack(ThreadDataPack::get()),
           threadIdentifier(threadIdentifier)
        {
            //Heartbeat
            Stopwatch::get().pulse(threadIdentifier);
            Stopwatch::get().sendAll();
            haltSignal.assignValue(false);
            isRunning.assignValue(false);
            lagTime.assignValue(0);
        }

        virtual ~ThreadObject()
        {}

        virtual void reset()
        {}

        void stop()
        {
            haltSignal.assignValue(true);
        }

        void start()
        {
            haltSignal.assignValue(false);
            run();
        }

        std::string getThreadIdentifier()
        {
            return threadIdentifier;
        }

        bool running()
        {
            return isRunning.getValue();
        }

        ThreadDataPack & threadPack;
        ThreadMutexObject<uint64_t> lagTime;

    protected:
        void run()
        {
            std::cout << threadIdentifier << " started" << std::endl;

            isRunning.assignValue(true);

            while(process() && !haltSignal.getValue())
            {
                Stopwatch::get().sendAll();
            }

            isRunning.assignValue(false);

            std::cout << threadIdentifier << " ended" << std::endl;
        }

        virtual bool inline process()
        {
            assert(false);
            return false;
        }

        std::string threadIdentifier;
        ThreadMutexObject<bool> haltSignal;
        ThreadMutexObject<bool> isRunning;
};

#endif /* THREADOBJECT_H_ */
