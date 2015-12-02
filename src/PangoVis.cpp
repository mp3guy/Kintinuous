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

//This includes "PangoVis.h" already
#include "MainController.h"

PangoVis::PangoVis(cv::Mat * depthIntrinsics)
 : ThreadObject("VisualisationThread"),
   liveTSDF(0),
   incMesh(0),
   complete("ui.Complete", false, false),
   pause("ui.Pause", false, true),
   save("ui.Save", false, false),
   resetAll("ui.Reset", false, false),
   volumeShifting("ui.Volume Shifting", !ConfigArgs::get().staticMode, true),
   limitFrontend("ui.Limit Frontend", threadPack.limit.getValue(), true),
   followPose("ui.Follow Pose", true, true),
   drawTSDF("ui.Draw TSDF", false, true),
   drawCloud("ui.Draw Cloud", true, true),
   drawMesh("ui.Draw Mesh", ConfigArgs::get().enableMeshGenerator || ConfigArgs::get().incrementalMesh, true),
   drawMeshNormals("ui.Draw Mesh Normals", false, true),
   totalPoints("ui.Points:", "0"),
   totalTriangles("ui.Triangles:", "0"),
   frame("ui.Frame:", "0"),
   frontendFps("ui.Frontend:", ""),
   backendLag("ui.Backend Lag:", ""),
   status("ui.Status:", ""),
   depthBuffer(new unsigned short[Resolution::get().numPixels()])
{
    pangolin::CreateWindowAndBind("Kintinuous", 1280 + 180, 960);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    rgbTex.Reinitialise(Resolution::get().width(), Resolution::get().height()),
    depthTex.Reinitialise(Resolution::get().width(), Resolution::get().height()),
    tsdfRgbTex.Reinitialise(Resolution::get().width(), Resolution::get().height()),
    tsdfTex.Reinitialise(Resolution::get().width(), Resolution::get().height()),

    rgbImg.Alloc(Resolution::get().width(), Resolution::get().height(), pangolin::VideoFormatFromString("RGB24"));
    tsdfImg.Alloc(Resolution::get().width(), Resolution::get().height(), pangolin::VideoFormatFromString("RGB24"));
    tsdfImgColor.Alloc(Resolution::get().width(), Resolution::get().height(), pangolin::VideoFormatFromString("RGB24"));
    depthImg.Alloc(Resolution::get().width(), Resolution::get().height(), pangolin::VideoFormatFromString("RGB24"));

    glEnable(GL_DEPTH_TEST);

    s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
                                        pangolin::ModelViewLookAt(-0.35, -2.3, -6.4, 0, 0, 0, 0, -1, 0));

    pangolin::Display("cam").SetBounds(0, 1.0f, 0, 1.0f, -640 / 480.0)
                            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::Display("Img").SetAspect(640.0f / 480.0f);
    pangolin::Display("Depth").SetAspect(640.0f / 480.0f);
    pangolin::Display("ModelImg").SetAspect(640.0f / 480.0f);
    pangolin::Display("Model").SetAspect(640.0f / 480.0f);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(180));

    pangolin::Display("multi").SetBounds(pangolin::Attach::Pix(0), 1 / 4.0f, pangolin::Attach::Pix(180), 1.0)
                              .SetLayout(pangolin::LayoutEqualHorizontal)
                              .AddDisplay(pangolin::Display("Img"))
                              .AddDisplay(pangolin::Display("Depth"))
                              .AddDisplay(pangolin::Display("ModelImg"))
                              .AddDisplay(pangolin::Display("Model"));

    K = Eigen::Matrix3f::Identity();
    K(0, 0) = depthIntrinsics->at<double>(0,0);
    K(1, 1) = depthIntrinsics->at<double>(1,1);
    K(0, 2) = depthIntrinsics->at<double>(0,2);
    K(1, 2) = depthIntrinsics->at<double>(1,2);

    Kinv = K.inverse();

    depthBuffer = new unsigned short[Resolution::get().numPixels()];

    reset();
}

PangoVis::~PangoVis()
{
    reset();

    delete [] depthBuffer;

    pangolin::FreeImage(rgbImg);
    pangolin::FreeImage(tsdfImg);
    pangolin::FreeImage(tsdfImgColor);
    pangolin::FreeImage(depthImg);
}

void PangoVis::removeAllClouds()
{
    for(size_t i = 0; i < clouds.size(); i++)
    {
        delete clouds.at(i);
    }

    clouds.clear();
}

void PangoVis::removeAllShapes()
{
    lines.clear();
    poses.clear();
}

void PangoVis::removeAllMeshes()
{
    for(size_t i = 0; i < meshes.size(); i++)
    {
        delete meshes.at(i);
    }

    meshes.clear();
}

void PangoVis::reset()
{
    latestDrawnMeshId = 0;
    numPoints = 0;
    numTriangles = 0;
    numTrianglePoints = 0;
    latestDrawnPoseCloudId = 0;

    removeAllClouds();
    removeAllShapes();
    removeAllMeshes();

    if(liveTSDF)
    {
        delete liveTSDF;
        liveTSDF = 0;
    }

    if(incMesh)
    {
        delete incMesh;
        incMesh = 0;
    }
}

void PangoVis::preCall()
{
    glClearColor(0.25, 0.25, 0.25, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    pangolin::Display("cam").Activate(s_cam);
}

bool inline PangoVis::process()
{
    TICK(threadIdentifier);

    if(pangolin::ShouldQuit())
    {
        MainController::controller->shutdown();
        return false;
    }

    preCall();

    processTsdf();

    processClouds();

    processMeshes();

    processImages();

    render();

    postCall();

    handleInput();

    TOCK(threadIdentifier);

    usleep(std::max(1, int(33000 - (Stopwatch::get().getTimings().at(threadIdentifier) * 1000))));

    return true;
}

void PangoVis::processClouds()
{
    int latestDensePoseIdCopy = threadPack.tracker->latestDensePoseId.getValue();

    if(latestDensePoseIdCopy > 0)
    {
        const float3 & voxelSizeMeters = Volume::get().getVoxelSizeMeters();

        pose = threadPack.isamOffset.getValue() * threadPack.loopOffset.getValue() * threadPack.tracker->densePoseGraph.at(latestDensePoseIdCopy - 1).pose;

        Eigen::Vector3f translation = pose.topRightCorner(3, 1);

        Eigen::Vector3f initialTrans = Eigen::Vector3f::Constant(Volume::get().getVolumeSize() * 0.5) - threadPack.tracker->getVolumeOffset();

        Eigen::Vector3f currentCubeTranslation = initialTrans;
        currentCubeTranslation(0) += std::floor(translation(0) / voxelSizeMeters.x) * voxelSizeMeters.x;
        currentCubeTranslation(1) += std::floor(translation(1) / voxelSizeMeters.y) * voxelSizeMeters.y;
        currentCubeTranslation(2) += std::floor(translation(2) / voxelSizeMeters.z) * voxelSizeMeters.z;

        //Kinda hacky
        if(volumeShifting)
        {
            tsdfCube.setEmpty();
            tsdfCube.extend(currentCubeTranslation + Eigen::Vector3f::Constant(Volume::get().getVolumeSize() / 2.0f));
            tsdfCube.extend(currentCubeTranslation - Eigen::Vector3f::Constant(Volume::get().getVolumeSize() / 2.0f));
        }

        pangolin::glDrawAlignedBox(tsdfCube);

        glColor3f(0, 1, 0);
        pangolin::glDrawFrustrum(Kinv, Resolution::get().width(), Resolution::get().height(), pose, 0.1f);
        glColor3f(1, 1, 1);
    }

    int latestPoseIdCopy = threadPack.latestPoseId.getValue();

    if(ConfigArgs::get().onlineDeformation)
    {
        boost::mutex::scoped_lock lock(threadPack.poolMutex, boost::try_to_lock);

        if(lock)
        {
            if(threadPack.poolLooped.getValue())
            {
                removeAllClouds();
                numPoints = 0;
                threadPack.poolLooped.assignValue(false);
            }

            if((int)threadPack.pointPool->size() != numPoints)
            {
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

                tempCloud->points.insert(tempCloud->end(), threadPack.pointPool->begin() + numPoints, threadPack.pointPool->end());

                lock.unlock();

                numPoints = threadPack.pointPool->size();

                static int i = 0;
                std::stringstream strs;
                strs << "pool" << i++;

                clouds.push_back(new PangoCloud(tempCloud.get()));
            }
            else
            {
                lock.unlock();
            }
        }

        removeAllShapes();

        for(int i = 2; i < latestPoseIdCopy; i++)
        {
            if(ConfigArgs::get().dynamicCube && !threadPack.cloudSlices.at(i)->poseIsam.getValue())
                break;

            Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
            pose.topLeftCorner(3, 3) = threadPack.cloudSlices.at(i)->cameraRotation;
            pose.topRightCorner(3, 1) = threadPack.cloudSlices.at(i)->cameraTranslation;
            poses.push_back(pose);

            if(i > 2)
            {
                lines.push_back(std::pair<Eigen::Vector3f, Eigen::Vector3f>(threadPack.cloudSlices.at(i)->cameraTranslation,
                                                                            threadPack.cloudSlices.at(i - 1)->cameraTranslation));
            }
        }
    }
    else
    {
        while(latestDrawnPoseCloudId < latestPoseIdCopy)
        {
            clouds.push_back(new PangoCloud(threadPack.cloudSlices.at(latestDrawnPoseCloudId)->processedCloud));

            numPoints += threadPack.cloudSlices.at(latestDrawnPoseCloudId)->processedCloud->size();

            if(latestDrawnPoseCloudId > 1)
            {
                Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
                pose.topLeftCorner(3, 3) = threadPack.cloudSlices.at(latestDrawnPoseCloudId)->cameraRotation;
                pose.topRightCorner(3, 1) = threadPack.cloudSlices.at(latestDrawnPoseCloudId)->cameraTranslation;
                poses.push_back(pose);

                if(latestDrawnPoseCloudId > 2)
                {
                    lines.push_back(std::pair<Eigen::Vector3f, Eigen::Vector3f>(threadPack.cloudSlices.at(latestDrawnPoseCloudId)->cameraTranslation,
                                                                                threadPack.cloudSlices.at(latestDrawnPoseCloudId - 1)->cameraTranslation));
                }
            }
            latestDrawnPoseCloudId++;
        }
    }
}

void PangoVis::processTsdf()
{
    if(threadPack.finalised.getValue())
    {
        if(liveTSDF)
        {
            delete liveTSDF;
            liveTSDF = 0;
        }
    }
    else if(drawTSDF)
    {
        boost::mutex::scoped_lock tsdfLock(threadPack.tracker->tsdfMutex);

        if(threadPack.tracker->tsdfAvailable)
        {
            if(liveTSDF)
            {
                delete liveTSDF;
                liveTSDF = 0;
            }

            if(ConfigArgs::get().dynamicCube)
            {
                pcl::transformPointCloud(*threadPack.tracker->getLiveTsdf()->cloud, *threadPack.tracker->getLiveTsdf()->cloud, threadPack.isamOffset.getValue());
            }

            liveTSDF = new PangoCloud(threadPack.tracker->getLiveTsdf()->cloud);

            threadPack.tracker->tsdfAvailable = false;
        }
    }
}

void PangoVis::processMeshes()
{
    int latestMeshIdCopy = threadPack.latestMeshId.getValue();

    if(ConfigArgs::get().incrementalMesh)
    {
        boost::mutex::scoped_lock lock(threadPack.incMeshMutex, boost::try_to_lock);

        if(lock)
        {
            int numIncPoints = threadPack.incrementalMesh->mesh->cloud.width * threadPack.incrementalMesh->mesh->cloud.height;

            bool looped = threadPack.incMeshLooped.getValue();

            if((int)threadPack.incrementalMesh->mesh->polygons.size() != numTriangles || numIncPoints != numTrianglePoints || looped)
            {
                removeAllMeshes();

                numTriangles = threadPack.incrementalMesh->mesh->polygons.size();
                numTrianglePoints = numIncPoints;

                if(incMesh)
                {
                    delete incMesh;
                    incMesh = 0;
                }

                incMesh = new PangoMesh(threadPack.incrementalMesh->mesh.get());

                if(looped)
                {
                    threadPack.incMeshLooped.assignValue(false);
                }
            }
        }
    }
    else
    {
        while(latestDrawnMeshId < latestMeshIdCopy)
        {
            if(threadPack.triangles.at(latestDrawnMeshId)->polygons.size() > 0)
            {
                meshes.push_back(new PangoMesh(threadPack.triangles.at(latestDrawnMeshId)));
                numTriangles += threadPack.triangles.at(latestDrawnMeshId)->polygons.size();
            }
            latestDrawnMeshId++;
        }
    }
}

void PangoVis::render()
{
    if((ConfigArgs::get().incrementalMesh || ConfigArgs::get().enableMeshGenerator) && drawMesh)
    {
        for(size_t i = 0; i < meshes.size(); i++)
        {
            meshes.at(i)->drawTriangles(drawMeshNormals);
        }

        if(incMesh)
        {
            incMesh->drawTriangles(drawMeshNormals);
        }
    }
    else if(drawCloud)
    {
        for(size_t i = 0; i < clouds.size(); i++)
        {
            clouds.at(i)->drawPoints();
        }

    }

    if(liveTSDF)
    {
        liveTSDF->drawPoints();
    }

    glColor3f(1, 0, 1);
    for(size_t i = 0; i < lines.size(); i++)
    {
        pangolin::glDrawLine(lines.at(i).first(0),
                             lines.at(i).first(1),
                             lines.at(i).first(2),
                             lines.at(i).second(0),
                             lines.at(i).second(1),
                             lines.at(i).second(2));
    }

    glColor3f(1, 1, 1);
    for(size_t i = 0; i < poses.size(); i++)
    {
        pangolin::glDrawFrustrum(Kinv, Resolution::get().width(), Resolution::get().height(), poses.at(i), 0.05f);
    }

    glDisable(GL_DEPTH_TEST);

    pangolin::Display("Img").Activate();
    rgbTex.RenderToViewport(true);

    pangolin::Display("Depth").Activate();
    depthTex.RenderToViewport(true);

    pangolin::Display("ModelImg").Activate();
    tsdfRgbTex.RenderToViewport(true);

    pangolin::Display("Model").Activate();
    tsdfTex.RenderToViewport(true);

    glEnable(GL_DEPTH_TEST);
}

void PangoVis::processImages()
{
    boost::mutex::scoped_lock imageLock(threadPack.tracker->imageMutex, boost::try_to_lock);

    if(imageLock && threadPack.tracker->imageAvailable)
    {
        threadPack.tracker->imageAvailable = false;

        memcpy(tsdfImg.ptr, threadPack.tracker->getLiveImage()->tsdfImage, Resolution::get().numPixels() * 3);
        memcpy(tsdfImgColor.ptr, threadPack.tracker->getLiveImage()->tsdfImageColor, Resolution::get().numPixels() * 3);
        memcpy(rgbImg.ptr, threadPack.tracker->getLiveImage()->rgbImage, Resolution::get().numPixels() * 3);
        memcpy(&depthBuffer[0], threadPack.tracker->getLiveImage()->depthData, Resolution::get().numPixels() * 2);

        imageLock.unlock();

        float max = 0;

        for(int i = 0; i < Resolution::get().numPixels(); i++)
        {
            if(depthBuffer[i] > max)
            {
                max = depthBuffer[i];
            }
        }

        int innerPtr = 0;
        for(int i = 0; i < Resolution::get().numPixels(); i++, innerPtr+=3)
        {
            depthImg.ptr[innerPtr + 0] = ((float)depthBuffer[i] / max) * 255.0f;
            depthImg.ptr[innerPtr + 1] = ((float)depthBuffer[i] / max) * 255.0f;
            depthImg.ptr[innerPtr + 2] = ((float)depthBuffer[i] / max) * 255.0f;
        }

        rgbTex.Upload(rgbImg.ptr, GL_RGB, GL_UNSIGNED_BYTE);
        depthTex.Upload(depthImg.ptr, GL_RGB, GL_UNSIGNED_BYTE);
        tsdfRgbTex.Upload(tsdfImgColor.ptr, GL_RGB, GL_UNSIGNED_BYTE);
        tsdfTex.Upload(tsdfImg.ptr, GL_BGR, GL_UNSIGNED_BYTE);

        //For a minimal "TSDF" visualisation
        if(!drawTSDF)
        {
            if(liveTSDF)
            {
                delete liveTSDF;
                liveTSDF = 0;
            }

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZRGB>);

            for(int i = 0; i < Resolution::get().width(); i++)
            {
                for(int j = 0; j < Resolution::get().height(); j++)
                {
                    if(depthBuffer[j * Resolution::get().width() + i])
                    {
                        pcl::PointXYZRGB pt;
                        pt.z = depthBuffer[j * Resolution::get().width() + i] * 0.001f;
                        pt.x = (static_cast<float>(i) - K(0, 2)) * pt.z * (1.0f / K(0, 0));
                        pt.y = (static_cast<float>(j) - K(1, 2)) * pt.z * (1.0f / K(1, 1));
                        pt.b = rgbImg.ptr[3 * (j * Resolution::get().width() + i) + 0];
                        pt.g = rgbImg.ptr[3 * (j * Resolution::get().width() + i) + 1];
                        pt.r = rgbImg.ptr[3 * (j * Resolution::get().width() + i) + 2];
                        cloud->push_back(pt);
                    }
                }
            }

            pcl::transformPointCloud(*cloud, *cloud, pose);

            liveTSDF = new PangoCloud(cloud.get());
        }
    }
    else if(imageLock)
    {
        imageLock.unlock();
    }
}

void PangoVis::handleInput()
{
    //So this is some hilarious access/control!
    if(pangolin::Pushed(complete))
    {
        MainController::controller->complete();
        followPose = false;
    }

    if(pause.GuiChanged())
    {
        threadPack.pauseCapture.assignValue(pause);
    }

    if(pangolin::Pushed(save))
    {
        MainController::controller->save();
    }

    if(pangolin::Pushed(resetAll))
    {
        MainController::controller->reset();
        MainController::controller->setPark(!volumeShifting);
        threadPack.pauseCapture.assignValue(pause);
    }

    if(volumeShifting.GuiChanged())
    {
        MainController::controller->setPark(!volumeShifting);
    }

    if(drawTSDF.GuiChanged())
    {
        threadPack.tracker->tsdfRequest.assignValue(drawTSDF);
    }

    if(limitFrontend.GuiChanged())
    {
        threadPack.limit.assignValue(limitFrontend);
    }

    if(pause)
    {
        status = "Paused";
    }
    else if(threadPack.cloudSliceProcessorFinished.getValue() &&
            threadPack.meshGeneratorFinished.getValue() &&
            threadPack.placeRecognitionFinished.getValue() &&
            threadPack.deformationFinished.getValue())
    {
        status = "Finished";
    }
    else
    {
        status = "Running";
    }

    if(followPose)
    {
        pangolin::OpenGlMatrix mv;

        Eigen::Matrix4f currPose = pose;
        Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

        Eigen::Quaternionf currQuat(currRot);
        Eigen::Vector3f forwardVector(0, 0, 1);
        Eigen::Vector3f upVector(0, -1, 0);

        Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
        Eigen::Vector3f up = (currQuat * upVector).normalized();

        Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

        eye -= forward * 20;

        Eigen::Vector3f at = eye + forward;

        Eigen::Vector3f z = (eye - at).normalized();  // Forward
        Eigen::Vector3f x = up.cross(z).normalized(); // Right
        Eigen::Vector3f y = z.cross(x);

        Eigen::Matrix4d m;
        m << x(0),  x(1),  x(2),  -(x.dot(eye)),
             y(0),  y(1),  y(2),  -(y.dot(eye)),
             z(0),  z(1),  z(2),  -(z.dot(eye)),
                0,     0,     0,              1;

        memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

        s_cam.SetModelViewMatrix(mv);
    }

    std::stringstream strs;
    strs << numPoints;
    totalPoints = strs.str();

    std::stringstream strst;
    strst << numTriangles;
    totalTriangles = strst.str();

    std::stringstream strsf;
    strsf << int(std::ceil(1.0f / (Stopwatch::get().getTimings().at("TrackerInterfaceThread") / 1000.0f))) << "Hz";
    frontendFps = strsf.str();

    std::stringstream strsb;
    strsb << int(float(MainController::controller->getMaxLag()) / 1000.0f) << "ms";
    backendLag = strsb.str();

    std::stringstream strsfr;
    strsfr << threadPack.trackerFrame.getValue();
    frame = strsfr.str();
}

void PangoVis::postCall()
{
    pangolin::FinishFrame();
}
