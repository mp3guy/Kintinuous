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

#ifndef LIVELOGREADER_H_
#define LIVELOGREADER_H_

#include <stdio.h>
#include <stdlib.h>
#include <poll.h>
#include <signal.h>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "LogReader.h"
#include "OpenNI2Interface.h"

class LiveLogReader : public LogReader
{
	public:
		LiveLogReader();
		virtual ~LiveLogReader();

		bool grabNext(bool & returnVal, int & currentFrame);

	private:
		OpenNI2Interface * asus;
		int64_t lastFrameTime;
		int lastGot;
};

#endif /* LIVELOGREADER_H_ */
