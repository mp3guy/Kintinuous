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

#ifndef UTILS_PANGOMESH_H_
#define UTILS_PANGOMESH_H_

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

class PangoMesh
{
    public:
        PangoMesh(pcl::PolygonMesh * mesh)
         : numPoints(mesh->cloud.width * mesh->cloud.height),
           numTriangles(mesh->polygons.size())
        {
            glGenBuffers(1, &vbo);
            glGenBuffers(1, &ibo);

            std::vector<uint32_t> indices;

            pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
            pcl::fromPCLPointCloud2(mesh->cloud, cloud);

            for(size_t i = 0; i < mesh->polygons.size(); i++)
            {
                indices.insert(indices.end(), mesh->polygons.at(i).vertices.begin(), mesh->polygons.at(i).vertices.end());
            }

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, cloud.points.size() * sizeof(pcl::PointXYZRGBNormal), cloud.points.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), &indices[0], GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        }

        virtual ~PangoMesh()
        {
            glDeleteBuffers(1, &vbo);
            glDeleteBuffers(1, &ibo);
        }

        void drawTriangles(const bool normals)
        {
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glVertexPointer(3, GL_FLOAT, sizeof(pcl::PointXYZRGBNormal), 0);

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            if(normals)
            {
                glColorPointer(3, GL_FLOAT, sizeof(pcl::PointXYZRGBNormal), (void *)(sizeof(float) * 4));
            }
            else
            {
                glColorPointer(3, GL_UNSIGNED_BYTE, sizeof(pcl::PointXYZRGBNormal), (void *)(sizeof(float) * 8));
            }

            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

            glDrawElements(GL_TRIANGLES, numTriangles * 3, GL_UNSIGNED_INT, 0);

            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);

            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        }

        GLuint vbo;
        GLuint ibo;
        const int numPoints;
        const int numTriangles;
};

#endif /* UTILS_PANGOMESH_H_ */
