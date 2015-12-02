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

#ifndef PANGOVIS_H_
#define PANGOVIS_H_

#include "utils/ThreadObject.h"
#include "frontend/Volume.h"
#include "utils/PangoCloud.h"
#include "utils/PangoMesh.h"

class PangoVis : public ThreadObject
{
    public:
        PangoVis(cv::Mat * depthIntrinsics);

        virtual ~PangoVis();

        void reset();

    private:
        bool inline process();

        void preCall();

        void postCall();

        void render();

        void handleInput();

        void removeAllClouds();

        void removeAllShapes();

        void removeAllMeshes();

        void processClouds();

        void processTsdf();

        void processMeshes();

        void processImages();

        pangolin::OpenGlRenderState s_cam;

        int latestDrawnPoseCloudId;
        int latestDrawnMeshId;
        int numPoints;
        int numTriangles;
        int numTrianglePoints;

        Eigen::Matrix4f pose;
        Eigen::Matrix3f K, Kinv;
        Eigen::AlignedBox3f tsdfCube;

        PangoCloud * liveTSDF;
        PangoMesh * incMesh;
        std::vector<PangoCloud*> clouds;
        std::vector<PangoMesh*> meshes;
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses;
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>, Eigen::aligned_allocator<std::pair<Eigen::Vector3f, Eigen::Vector3f>>> lines;

        pangolin::Var<bool> complete;
        pangolin::Var<bool> pause;
        pangolin::Var<bool> save;
        pangolin::Var<bool> resetAll;
        pangolin::Var<bool> volumeShifting;
        pangolin::Var<bool> limitFrontend;

        pangolin::Var<bool> followPose;
        pangolin::Var<bool> drawTSDF;
        pangolin::Var<bool> drawCloud;
        pangolin::Var<bool> drawMesh;
        pangolin::Var<bool> drawMeshNormals;

        pangolin::Var<std::string> totalPoints;
        pangolin::Var<std::string> totalTriangles;
        pangolin::Var<std::string> frame;
        pangolin::Var<std::string> frontendFps;
        pangolin::Var<std::string> backendLag;
        pangolin::Var<std::string> status;

        pangolin::GlTexture rgbTex,
                            depthTex,
                            tsdfRgbTex,
                            tsdfTex;

        pangolin::TypedImage rgbImg;
        pangolin::TypedImage tsdfImg;
        pangolin::TypedImage tsdfImgColor;
        pangolin::TypedImage depthImg;

        unsigned short * depthBuffer;
};

#endif /* PANGOVIS_H_ */
