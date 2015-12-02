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

#ifndef THREADMUTEXOBJECT_H_
#define THREADMUTEXOBJECT_H_

#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/condition_variable.hpp>

template <class T>
class ThreadMutexObject
{
    public:
        ThreadMutexObject()
        {}

        ThreadMutexObject(T initialValue)
         : object(initialValue),
           lastCopy(initialValue)
        {}

        void assignValue(T newValue)
        {
            boost::mutex::scoped_lock lock(mutex);

            object = lastCopy = newValue;
        }

        boost::mutex & getMutex()
        {
            return mutex;
        }

        T & getReference()
        {
            return object;
        }

        void assignAndNotifyAll(T newValue)
        {
            boost::mutex::scoped_lock lock(mutex);

            object = newValue;

            signal.notify_all();
        }
        
        void notifyAll()
        {
            boost::mutex::scoped_lock lock(mutex);

            signal.notify_all();
        }

        T getValue()
        {
            boost::mutex::scoped_lock lock(mutex);

            lastCopy = object;

            return lastCopy;
        }

        T waitForSignal()
        {
            boost::mutex::scoped_lock lock(mutex);

            signal.wait(mutex);

            lastCopy = object;

            return lastCopy;
        }

        T getValueWait(int wait = 33000)
        {
            boost::this_thread::sleep(boost::posix_time::microseconds(wait));

            boost::mutex::scoped_lock lock(mutex);

            lastCopy = object;

            return lastCopy;
        }

        T & getReferenceWait(int wait = 33000)
        {
            boost::this_thread::sleep(boost::posix_time::microseconds(wait));

            boost::mutex::scoped_lock lock(mutex);

            lastCopy = object;

            return lastCopy;
        }

        void operator++(int)
        {
            boost::mutex::scoped_lock lock(mutex);

            object++;
        }

        void operator+=(const uint64_t & other)
        {
            boost::mutex::scoped_lock lock(mutex);

            object+=other;
        }

    private:
        T object;
        T lastCopy;
        boost::mutex mutex;
        boost::condition_variable_any signal;
};

#endif /* THREADMUTEXOBJECT_H_ */
