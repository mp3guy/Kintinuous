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

#ifndef SURF3DTOOLS_H_
#define SURF3DTOOLS_H_

#include "../utils/ConfigArgs.h"
#include "DBowInterfaceSurf.h"
#include <opencv2/opencv.hpp>
#include <sstream>
#include "DepthCamera.h"

class InterestPoint
{
    public:
        InterestPoint(float X, float Y, float Z, float u, float v)
             : X(X), Y(Y), Z(Z), u(u), v(v)
            {}
            float X, Y, Z, u, v;
};

class Surf3DTools
{
    public:
        class Surf3DImage
        {
            public:
                Surf3DImage(std::vector<float> & imageDescriptor,
                            std::vector<cv::KeyPoint> & imageKeyPoints)
                 : descriptor(imageDescriptor),
                   keyPoints(imageKeyPoints)
                {}

                class PointCorrespondence
                {
                    public:
                        PointCorrespondence(CvPoint3D32f point3d,
                                            CvPoint2D32f coordIm)
                         : point3d(point3d),
                           coordIm(coordIm)
                        {}

                        CvPoint3D32f point3d;
                        CvPoint2D32f coordIm;
                };

                std::vector<float> & descriptor;
                std::vector<cv::KeyPoint> & keyPoints;
                std::vector<PointCorrespondence> pointCorrespondences;
        };

        static Surf3DImage * calculate3dPointsSURF(DepthCamera * depthCamera,
                                                   cv::Mat * depthMap,
                                                   std::vector<float> & imageDescriptor,
                                                   std::vector<cv::KeyPoint> & imageKeyPoints)
        {
            Surf3DImage * newSurf3DImage = new Surf3DImage(imageDescriptor, imageKeyPoints);

            cv::Mat * image3D = new cv::Mat();

            depthCamera->computeVertexMap(*depthMap, *image3D);

            const double maxZ = 10.0;

            for(int y = 0; y < image3D->rows; y++)
            {
                for(int x = 0; x < image3D->cols; x++)
                {
                    cv::Vec3f point = image3D->at<cv::Vec3f> (y, x);

                    if(fabs(point[2] - maxZ) < FLT_EPSILON || fabs(point[2]) > maxZ)
                    {
                        continue;
                    }
                    else
                    {
                        newSurf3DImage->pointCorrespondences.push_back(Surf3DImage::PointCorrespondence(cvPoint3D32f(point[0],
                                                                                                                     point[1],
                                                                                                                     point[2]),
                                                                                                        cvPoint2D32f(x, y)));
                    }
                }
            }

            delete image3D;

            return newSurf3DImage;
        }

        static void surfMatch3D(Surf3DImage * one,
                                Surf3DImage * two,
                                std::vector<std::vector<float> > & matches1,
                                std::vector<std::vector<float> > & matches2)
        {
            vector<int> ind;
            compare_descriptors(one->descriptor, two->descriptor, ind);

            int inn1, inn2, is1, is2;

            float tol = 0.5;

            float mindist2, dist2;

            vector<int> j1, j2, i1, i2;

            for(int is = 0; is < (int) ind.size(); is++)
            {
                if(ind[is] > 0)
                {
                    inn1 = -1;
                    is1 = -1;
                    mindist2 = 1;

                    for(int ipc = 0; ipc < (int) one->pointCorrespondences.size(); ipc++)
                    {
                        if(one->pointCorrespondences.at(ipc).coordIm.y > (one->keyPoints.at(is).pt.y - tol) &&
                           one->pointCorrespondences.at(ipc).coordIm.y < (one->keyPoints.at(is).pt.y + tol) &&
                           one->pointCorrespondences.at(ipc).coordIm.x > (one->keyPoints.at(is).pt.x - tol) &&
                           one->pointCorrespondences.at(ipc).coordIm.x < (one->keyPoints.at(is).pt.x + tol))
                        {
                            dist2 = pow(one->pointCorrespondences.at(ipc).coordIm.x - one->keyPoints.at(is).pt.x, 2) +
                                    pow(one->pointCorrespondences.at(ipc).coordIm.y - one->keyPoints.at(is).pt.y, 2);
                            if(dist2 < mindist2)
                            {
                                mindist2 = dist2;
                                inn1 = ipc;
                                is1 = is;
                            }
                        }
                    }

                    inn2 = -1;
                    is2 = -1;
                    mindist2 = 1;

                    for(int ipc = 0; ipc < (int) two->pointCorrespondences.size(); ipc++)
                    {
                        if(two->pointCorrespondences.at(ipc).coordIm.y > (two->keyPoints.at(ind[is]).pt.y - tol) &&
                           two->pointCorrespondences.at(ipc).coordIm.y < (two->keyPoints.at(ind[is]).pt.y + tol) &&
                           two->pointCorrespondences.at(ipc).coordIm.x > (two->keyPoints.at(ind[is]).pt.x - tol) &&
                           two->pointCorrespondences.at(ipc).coordIm.x < (two->keyPoints.at(ind[is]).pt.x + tol))
                        {
                            dist2 = pow(two->pointCorrespondences.at(ipc).coordIm.x - two->keyPoints.at(ind[is]).pt.x, 2) +
                                    pow(two->pointCorrespondences.at(ipc).coordIm.y - two->keyPoints.at(ind[is]).pt.y, 2);
                            if(dist2 < mindist2)
                            {
                                mindist2 = dist2;
                                inn2 = ipc;
                                is2 = ind[is];
                            }
                        }
                    }

                    if(inn1 >= 0 and inn2 >= 0)
                    {
                        j1.push_back(inn1);
                        j2.push_back(inn2);
                        i1.push_back(is1);
                        i2.push_back(is2);
                    }
                }
            }

            matches1.resize(j1.size());
            matches2.resize(j2.size());

            for(unsigned int i = 0; i < j1.size(); i++)
            {
                matches1[i].resize(5);
                matches1[i][0] = one->pointCorrespondences.at(j1[i]).point3d.x;
                matches1[i][1] = one->pointCorrespondences.at(j1[i]).point3d.y;
                matches1[i][2] = one->pointCorrespondences.at(j1[i]).point3d.z;
                matches1[i][3] = one->keyPoints.at(i1[i]).pt.x;
                matches1[i][4] = one->keyPoints.at(i1[i]).pt.y;
                matches2[i].resize(5);
                matches2[i][0] = two->pointCorrespondences.at(j2[i]).point3d.x;
                matches2[i][1] = two->pointCorrespondences.at(j2[i]).point3d.y;
                matches2[i][2] = two->pointCorrespondences.at(j2[i]).point3d.z;
                matches2[i][3] = two->keyPoints.at(i2[i]).pt.x;
                matches2[i][4] = two->keyPoints.at(i2[i]).pt.y;
            }
        }

        static void compare_descriptors(vector<float> & des1, vector<float> & des2, vector<int> & ind)
        {
            vector<int> ptpairs;
            int i;

            if((int)des1.size() > 4 * DBowInterfaceSurf::surfDescriptorLength)
            {
                flannFindPairs(des1, des2, ptpairs);
            }

            ind.resize(des1.size() / DBowInterfaceSurf::surfDescriptorLength);

            for(i = 0; i < (int)ptpairs.size(); i += 2)
            {
                ind[ptpairs[i]] = ptpairs[i + 1];
            }
        }

        static void flannFindPairs(vector<float> & des1, vector<float> & des2, vector<int> & ptpairs)
        {
            ptpairs.clear();

            if(des1.size() == 0 || des2.size() == 0)
            {
                return;
            }

            float A[(int)des1.size()];
            float B[(int)des2.size()];

            int k = 0;
            k = min((int)des1.size(), (int)des2.size());

            for(int i = 0; i < k; i++ )
            {
                A[i] = des1[i];
                B[i] = des2[i];
            }

            if(k == (int)des1.size())
            {
                for(int i = k; i < (int)des2.size(); i++)
                {
                    B[i] = des2[i];
                }
            }
            else
            {
                for(int i = k; i < (int)des1.size(); i++)
                {
                    A[i] = des1[i];
                }
            }

            cv::Mat m_image((int)des1.size() / DBowInterfaceSurf::surfDescriptorLength, DBowInterfaceSurf::surfDescriptorLength, CV_32F, A);
            cv::Mat m_object((int)des2.size() / DBowInterfaceSurf::surfDescriptorLength, DBowInterfaceSurf::surfDescriptorLength, CV_32F, B);

            // find nearest neighbors using FLANN
            cv::Mat m_indices((int)des2.size() / DBowInterfaceSurf::surfDescriptorLength, 2, CV_32S);
            cv::Mat m_dists((int)des2.size() / DBowInterfaceSurf::surfDescriptorLength, 2, CV_32F);

            cv::flann::Index flann_index(m_image, cv::flann::KDTreeIndexParams(1));  // using 1 randomized kdtrees

            flann_index.knnSearch(m_object, m_indices, m_dists, 2, cv::flann::SearchParams(64)); // maximum number of leafs checked
            int * indices_ptr = m_indices.ptr<int>(0);
            float * dists_ptr = m_dists.ptr<float>(0);

            for(int i = 0; i < m_indices.rows; i++)
            {
                if (dists_ptr[2 * i] < 0.49 * dists_ptr[2 * i + 1])
                {
                    ptpairs.push_back(indices_ptr[2 * i]);
                    ptpairs.push_back(i);
                }
            }
        }

    private:
        Surf3DTools()
        {}
};

#endif /* SURF3DTOOLS_H_ */
