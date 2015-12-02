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

#ifndef UTILS_PANGOCLOUD_H_
#define UTILS_PANGOCLOUD_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//Because PCL and Pangolin don't play nice
#ifdef HAVE_OPENNI
#undef HAVE_OPENNI
#endif

#ifdef HAVE_OPENNI2
#undef HAVE_OPENNI2
#endif

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>

class PangoCloud
{
    public:
        PangoCloud(pcl::PointCloud<pcl::PointXYZRGB> * cloud)
         : numPoints(cloud->size()),
           offset(4),
           stride(sizeof(pcl::PointXYZRGB))
        {
            glGenBuffers(1, &vbo);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, cloud->points.size() * stride, cloud->points.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        PangoCloud(pcl::PointCloud<pcl::PointXYZRGBNormal> * cloud)
         : numPoints(cloud->size()),
           offset(8),
           stride(sizeof(pcl::PointXYZRGBNormal))
        {
            glGenBuffers(1, &vbo);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, cloud->points.size() * stride, cloud->points.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        virtual ~PangoCloud()
        {
            glDeleteBuffers(1, &vbo);
        }

        void drawPoints()
        {
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glVertexPointer(3, GL_FLOAT, stride, 0);
            glColorPointer(3, GL_UNSIGNED_BYTE, stride, (void *)(sizeof(float) * offset));

            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);

            glDrawArrays(GL_POINTS, 0, numPoints);

            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);

            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        const int numPoints;

    private:
        const int offset;
        const int stride;
        GLuint vbo;

};

#endif /* UTILS_PANGOCLOUD_H_ */
