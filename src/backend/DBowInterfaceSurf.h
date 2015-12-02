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

#ifndef DBOWINTERFACESURF_H_
#define DBOWINTERFACESURF_H_

#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <DVision/DVision.h>
#include <DBoW2/FSurf64.h>
#include <DLoopDetector/TemplatedLoopDetector.h>

class DBowInterfaceSurf
{
    public:
        DBowInterfaceSurf(int width, int height, int mode, const std::string & filename);
        virtual ~DBowInterfaceSurf();

        void detectSURF(const cv::Mat & im, std::vector<float> & imageDescriptor, std::vector<cv::KeyPoint> & imageKeyPoints);
        DLoopDetector::DetectionResult detectLoop();
        void computeExportVocab();

        const int width;
        const int height;

        const int currentMode;
        const std::string filename;

        enum Mode
        {
            VOCAB_CREATION = 0,
            LOOP_DETECTION = 1
        };

        const static int surfDescriptorLength = 64;

        void reset();

    private:
        std::vector<cv::KeyPoint> keys;
        std::vector<FSurf64::TDescriptor> descriptors;
        std::vector<std::vector<FSurf64::TDescriptor> > descriptorsCollection;
        DLoopDetector::DetectionResult result;

        DVision::SurfSet surfExtractor;
        DBoW2::TemplatedVocabulary<FSurf64::TDescriptor, FSurf64> * vocab;
        DLoopDetector::TemplatedLoopDetector<FSurf64::TDescriptor, FSurf64> * surfLoopDetector;
};

#endif /* DBOWINTERFACESURF_H_ */
