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

#include "MainController.h"

MainController * MainController::controller = 0;

MainController::MainController(int argc, char * argv[])
 : depthIntrinsics(0),
   pangoVis(0),
   trackerInterface(0),
   meshGenerator(0),
   placeRecognition(0),
   cloudSliceProcessor(0),
   deformation(0),
   rawRead(0),
   liveRead(0),
   logRead(0)
{
    ConfigArgs::get(argc, argv);

    assert(!MainController::controller);

    MainController::controller = this;
}

MainController::~MainController()
{
    if(depthIntrinsics)
    {
        delete depthIntrinsics;
    }
}

int MainController::start()
{
    if(setup())
    {
        return mainLoop();
    }
    else
    {
        return -1;
    }
}

bool MainController::setup()
{
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    Resolution::get(640, 480);

    Volume::get(ConfigArgs::get().volumeSize);

    Stopwatch::get().setCustomSignature(43543534);

    cudaSafeCall(cudaSetDevice(ConfigArgs::get().gpu));

    loadCalibration();

    std::cout << "Point resolution: " << ((int)((Volume::get().getVoxelSizeMeters().x * 1000.0f) * 10.0f)) / 10.0f << " millimetres" << std::endl;

    if(ConfigArgs::get().logFile.size())
    {
        rawRead = new RawLogReader;
        logRead = static_cast<LogReader *>(rawRead);
    }
    else
    {
        liveRead = new LiveLogReader;
        logRead = static_cast<LogReader *>(liveRead);
    }

    ThreadDataPack::get();

    trackerInterface = new TrackerInterface(logRead, depthIntrinsics);

    if(ConfigArgs::get().trajectoryFile.size())
    {
        std::cout << "Load trajectory: " << ConfigArgs::get().trajectoryFile << std::endl;
        trackerInterface->loadTrajectory(ConfigArgs::get().trajectoryFile);
    }

    systemComponents.push_back(trackerInterface);

    ThreadDataPack::get().assignFrontend(trackerInterface->getFrontend());

    cloudSliceProcessor = new CloudSliceProcessor();
    systemComponents.push_back(cloudSliceProcessor);

    if(ConfigArgs::get().extractOverlap)
    {
        trackerInterface->enableOverlap();
    }

    if(!ConfigArgs::get().incrementalMesh && ConfigArgs::get().enableMeshGenerator)
    {
        meshGenerator = new MeshGenerator();
        systemComponents.push_back(meshGenerator);
    }
    else
    {
        ThreadDataPack::get().meshGeneratorFinished.assignValue(true);
    }

    if(ConfigArgs::get().vocabFile.size() && ConfigArgs::get().onlineDeformation)
    {
        deformation = new Deformation;
        placeRecognition = new PlaceRecognition(depthIntrinsics);

        systemComponents.push_back(deformation);
        systemComponents.push_back(placeRecognition);
    }
    else
    {
        ThreadDataPack::get().deformationFinished.assignValue(true);
        ThreadDataPack::get().placeRecognitionFinished.assignValue(true);
    }

    pangoVis = new PangoVis(depthIntrinsics);

    return true;
}

int MainController::mainLoop()
{
    timeval start;
    gettimeofday(&start, 0);
    uint64_t beginning = start.tv_sec * 1000000 + start.tv_usec;

    for(unsigned int i = 0; i < systemComponents.size(); i++)
    {
        threads.add_thread(new boost::thread(boost::bind(&ThreadObject::start, systemComponents.at(i))));
    }

    if(pangoVis)
    {
        pangoVis->start();
    }

    threads.join_all();

    for(unsigned int i = 0; i < systemComponents.size(); i++)
    {
        delete systemComponents.at(i);
    }

    if(pangoVis)
    {
        pangoVis->stop();
        delete pangoVis;
    }

    if(rawRead)
    {
        delete rawRead;
    }

    if(liveRead)
    {
        delete liveRead;
    }

    return 0;
}

void MainController::loadCalibration()
{
    if(ConfigArgs::get().calibrationFile.length() > 0)
    {
        cv::FileStorage calibrationFile(ConfigArgs::get().calibrationFile.c_str(), cv::FileStorage::READ);
        depthIntrinsics = new cv::Mat((CvMat *) calibrationFile["depth_intrinsics"].readObj(), true);
    }
    else
    {
        depthIntrinsics = new cv::Mat(cv::Mat::zeros(3, 3, CV_64F));
        depthIntrinsics->at<double>(0, 2) = 320;
        depthIntrinsics->at<double>(1, 2) = 267;
        depthIntrinsics->at<double>(0, 0) = 528.01442863461716;
        depthIntrinsics->at<double>(1, 1) = 528.01442863461716;
        depthIntrinsics->at<double>(2, 2) = 1;
    }
}

void MainController::complete()
{
    trackerInterface->endRequested.assignValue(true);
}

void MainController::save()
{
    if(ThreadDataPack::get().finalised.getValue())
    {
        if(!ConfigArgs::get().onlineDeformation)
        {
            boost::thread * cloudSaveThread = new boost::thread(boost::bind(&CloudSliceProcessor::save, cloudSliceProcessor));
            assert(cloudSaveThread);

            if(controller->meshGenerator)
            {
                boost::thread * meshSaveThread = new boost::thread(boost::bind(&MeshGenerator::save, meshGenerator));
                assert(meshSaveThread);
            }
        }
        else
        {
            boost::thread * cloudSaveThread = new boost::thread(boost::bind(&Deformation::saveCloud, deformation));
            assert(cloudSaveThread);

            if(ConfigArgs::get().enableMeshGenerator)
            {
                boost::thread * meshSaveThread = new boost::thread(boost::bind(&Deformation::saveMesh, deformation));
                assert(meshSaveThread);
            }
        }
    }
}

void MainController::reset()
{
    if(!ThreadDataPack::get().finalised.getValue())
    {
        for(unsigned int i = 0; i < systemComponents.size(); i++)
        {
            systemComponents.at(i)->stop();
        }

        while(true)
        {
            bool stillRunning = false;

            for(unsigned int i = 0; i < systemComponents.size(); i++)
            {
                if(systemComponents.at(i)->running())
                {
                    stillRunning = true;
                    break;
                }
            }

            if(!stillRunning)
            {
                break;
            }
            else
            {
                ThreadDataPack::get().notifyVariables();
                ThreadDataPack::get().tracker->cloudSignal.notify_all();
            }
        }

        threads.join_all();

        if(pangoVis)
        {
            pangoVis->reset();
        }

        for(unsigned int i = 0; i < systemComponents.size(); i++)
        {
            systemComponents.at(i)->reset();
        }

        ThreadDataPack::get().reset();

        for(unsigned int i = 0; i < systemComponents.size(); i++)
        {
            threads.add_thread(new boost::thread(boost::bind(&ThreadObject::start, systemComponents.at(i))));
        }
    }
}

void MainController::setPark(const bool park)
{
    trackerInterface->setPark(park);
}

void MainController::shutdown()
{
    for(size_t i = 0; i < systemComponents.size(); i++)
    {
        systemComponents.at(i)->stop();
    }

    while(true)
    {
        bool stillRunning = false;

        for(size_t i = 0; i < systemComponents.size(); i++)
        {
            if(systemComponents.at(i)->running())
            {
                stillRunning = true;
                break;
            }
        }

        if(!stillRunning)
        {
            break;
        }
        else
        {
            ThreadDataPack::get().notifyVariables();
            ThreadDataPack::get().tracker->cloudSignal.notify_all();
        }
    }

    if(pangoVis)
    {
        pangoVis->stop();
    }
}

uint64_t MainController::getMaxLag()
{
    uint64_t maxLag = 0;

    for(size_t i = 0; i < systemComponents.size(); i++)
    {
        maxLag = std::max(systemComponents.at(i)->lagTime.getValue(), maxLag);
    }

    return maxLag;
}
