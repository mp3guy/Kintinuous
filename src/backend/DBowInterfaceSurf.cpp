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


#include "DBowInterfaceSurf.h"

DBowInterfaceSurf::DBowInterfaceSurf(int width, int height, int mode, const std::string & filename)
 : width(width), height(height), currentMode(mode), filename(filename)
{
    assert(currentMode != -1);

    if(currentMode == LOOP_DETECTION)
    {
        std::cout << "Loading vocabularly for loop detection... ";
        std::cout.flush();
        vocab = new DBoW2::TemplatedVocabulary<FSurf64::TDescriptor, FSurf64>(filename);
        std::cout << "vocabularly loaded" << std::endl;

        DLoopDetector::TemplatedLoopDetector<FSurf64::TDescriptor, FSurf64>::Parameters params(height, width);

        params.use_nss = true; // use normalized similarity score instead of raw score
        params.alpha = 0.3; // nss threshold
        params.k = 1; // a loop must be consistent with 1 previous matches
        params.geom_check = DLoopDetector::GEOM_DI; // use direct index for geometrical checking
        params.di_levels = 2; // use two direct index levels

        std::cout << "Initialising loop detector... ";
        std::cout.flush();
        surfLoopDetector = new DLoopDetector::TemplatedLoopDetector<FSurf64::TDescriptor, FSurf64>(*vocab, params);
        surfLoopDetector->allocate(1000, 1000);
        std::cout << "loop detector initialised" << std::endl;
    }
    else if(currentMode == VOCAB_CREATION)
    {
        vocab = new DBoW2::TemplatedVocabulary<FSurf64::TDescriptor, FSurf64>();
    }
}

DBowInterfaceSurf::~DBowInterfaceSurf()
{
    if(currentMode == LOOP_DETECTION)
    {
        delete surfLoopDetector;
    }

    delete vocab;
}

void DBowInterfaceSurf::reset()
{
    keys.clear();
    descriptors.clear();
    descriptorsCollection.clear();
    surfLoopDetector->clear();
}

void DBowInterfaceSurf::detectSURF(const cv::Mat & im, std::vector<float> & imageDescriptor, std::vector<cv::KeyPoint> & imageKeyPoints)
{
    // extract surfs with opencv
    static cv::SURF surf_detector(400, 4, 2, false);

    vector<float> plain;
    surf_detector(im, cv::Mat(), keys, plain);

    imageDescriptor.insert(imageDescriptor.end(), plain.begin(), plain.end());
    imageKeyPoints.insert(imageKeyPoints.end(), keys.begin(), keys.end());

    // change descriptor format
    const int L = surf_detector.descriptorSize();

    descriptors.resize(plain.size() / L);

    unsigned int j = 0;
    for(unsigned int i = 0; i < plain.size(); i += L, ++j)
    {
        descriptors[j].resize(L);
        std::copy(plain.begin() + i, plain.begin() + i + L, descriptors[j].begin());
    }

    if(currentMode == VOCAB_CREATION)
    {
        descriptorsCollection.push_back(descriptors);
    }
}

void DBowInterfaceSurf::computeExportVocab()
{
    assert(currentMode == VOCAB_CREATION);

    std::cout << "Creating vocabulary..." << std::endl;
    vocab->create(descriptorsCollection);

    std::cout << "Saving to disk as " << filename << "..." << std::endl;
    vocab->save(filename);
    std::cout << "New vocabularly created" << std::endl;
}

DLoopDetector::DetectionResult DBowInterfaceSurf::detectLoop()
{
    assert(currentMode == LOOP_DETECTION);

    surfLoopDetector->detectLoop(keys, descriptors, result);

    return result;
}
