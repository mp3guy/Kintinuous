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


#ifndef DEFORMATIONGRAPH_H_
#define DEFORMATIONGRAPH_H_

#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <vector>
#include <pcl/PolygonMesh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include "../utils/Stopwatch.h"
#include "GraphNode.h"
#include "Jacobian.h"
#include "CholeskyDecomp.h"

/**
 * This is basically and object-oriented type approach. Using an array based approach would be faster...
 */

class DeformationGraph
{
    public:
        DeformationGraph(int k);
        virtual ~DeformationGraph();

        //Naive initialisation
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr initialiseGraphNN(pcl::PointCloud<pcl::PointXYZRGBNormal> * vertices,
                                                                       const double targetSpacing = 1.0f);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr initialiseGraphPosesNN(pcl::PointCloud<pcl::PointXYZRGBNormal> * vertices,
                                                                            const double targetSpacing,
                                                                            std::vector<uint64_t> * vertexTimeMap,
                                                                            unsigned int originalPointEnd);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr initialiseGraphPoses(pcl::PointCloud<pcl::PointXYZRGBNormal> * vertices,
                                                                          float poseDist,
                                                                          pcl::PointCloud<pcl::PointXYZRGBNormal> * customGraph,
                                                                          std::vector<uint64_t> * graphTimeMap,
                                                                          std::vector<uint64_t> * vertexTimeMap,
                                                                          unsigned int originalPointEnd);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr appendGraphPoses(float poseDist,
                                                                      pcl::PointCloud<pcl::PointXYZRGBNormal> * customGraph,
                                                                      std::vector<uint64_t> * graphTimeMap,
                                                                      std::vector<uint64_t> * vertexTimeMap,
                                                                      unsigned int originalPointEnd);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr appendVertices(std::vector<uint64_t> * vertexTimeMap,
                                                                    unsigned int originalPointEnd);

        //Stores a weight and node pointer for a vertex
        class VertexWeightMap
        {
            public:
                VertexWeightMap(double weight, int node)
                : weight(weight),
                  node(node)
                {}

                double weight;
                int node;

                /**
                 * BubblesortLOL
                 * @param list
                 * @param graph
                 */
                static void sort(std::vector<VertexWeightMap> & list, std::vector<GraphNode *> & graph)
                {
                    bool done = false;

                    int size = list.size();

                    while(!done)
                    {
                        done = true;
                        for(int i = 0; i < size - 1; i++)
                        {
                            if(graph.at(list[i].node)->id > graph.at(list[i + 1].node)->id)
                            {
                                done = false;
                                std::swap(list[i], list[i + 1]);
                            }
                        }
                        size--;
                    }
                }
        };

        std::vector<GraphNode *> & getGraph();
        std::vector<std::vector<VertexWeightMap> > & getVertexMap();

        void addConstraint(int vertexId, Eigen::Vector3d & target);
        void removeConstraint(int vertexId);
        void clearConstraints();

        void applyGraphToVertices(int numThreads);
        void optimiseGraphSparse();
        void resetGraph();

    private:
        bool initialised;

        //Number of neighbours
        const int k;

        //From paper
        const double wRot;
        const double wReg;
        const double wCon;

        static const int numVariables = 12;
        static const int eRotRows = 6;
        static const int eRegRows = 3;
        static const int eConRows = 3;

        //Graph itself
        GraphNode * graphNodes;
        std::vector<GraphNode *> graph;

        //Maps vertex indices to neighbours and weights
        std::vector<std::vector<VertexWeightMap> > vertexMap;
        pcl::PointCloud<pcl::PointXYZRGBNormal> * sourceVertices;

        //Stores a vertex constraint
        class Constraint
        {
            public:
                Constraint(int vertexId,
                           Eigen::Vector3d & targetPosition)
                 : vertexId(vertexId),
                   targetPosition(targetPosition)
                {}

                int vertexId;
                Eigen::Vector3d targetPosition;
        };

        std::vector<Constraint> constraints;

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud;
        std::vector<uint64_t> sampledGraphTimes;
        unsigned int lastPointCount;

        void connectGraphSeq(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud);

        void weightVerticesSeq(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud,
                               std::vector<uint64_t> * vertexTimeMap,
                               std::vector<uint64_t> & sampledGraphTimes);

        void radiusSample(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud, const double targetSpacing);

        void radiusSampleTemporal(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud,
                                  std::vector<uint64_t> * vertexTimeMap,
                                  std::vector<uint64_t> * graphTimeMap,
                                  const double targetSpacing);

        void connectGraphNN(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud,
                            pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> & graphTree);

        void connectGraphNNTemporal(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud,
                                    std::vector<uint64_t> * graphTimeMap,
                                    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> & graphTree);

        void weightVerticesNN(pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> & graphTree);

        void weightVerticesNNTemporal(pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> & graphTree,
                                      std::vector<uint64_t> * vertexTimeMap,
                                      std::vector<uint64_t> * graphTimeMap);

        void computeVertexPosition(int vertexId, Eigen::Vector3d & position, Eigen::Vector3d & normal);

        void sparseJacobian(Jacobian & jacobian, const int numRows, const int numCols);

        Eigen::VectorXd sparseResidual(const int numRows);

        void applyDeltaSparse(Eigen::VectorXd & delta);

        CholeskyDecomp cholesky;

        void applyGraphThread(int id, int numThreads);

        Eigen::VectorXd sparseResidualRot(const int numRows)
        {
            //Now the residual
            Eigen::VectorXd residual(numRows);

            int lastRow = 0;

            for(unsigned int j = 0; j < graph.size(); j++)
            {
                //No weights for rotation as rotation weight = 1
                const Eigen::Matrix3d & rotation = graph.at(j)->rotation;

                //ab + de + gh
                residual(lastRow) = rotation.col(0).dot(rotation.col(1));

                //ac + df + gi
                residual(lastRow + 1) = rotation.col(0).dot(rotation.col(2));

                //bc + ef + hi
                residual(lastRow + 2) = rotation.col(1).dot(rotation.col(2));

                //a^2 + d^2 + g^2 - 1
                residual(lastRow + 3) = (rotation.col(0).dot(rotation.col(0)) - 1.0);

                //b^2 + e^2 + h^2 - 1
                residual(lastRow + 4) = (rotation.col(1).dot(rotation.col(1)) - 1.0);

                //c^2 + f^2 + i^2 - 1
                residual(lastRow + 5) = (rotation.col(2).dot(rotation.col(2)) - 1.0);

                lastRow += eRotRows;
            }

            return residual;
        }

        Eigen::VectorXd sparseResidualReg(const int numRows)
        {
            //Now the residual
            Eigen::VectorXd residual(numRows);

            int lastRow = 0;

            for(unsigned int j = 0; j < graph.size(); j++)
            {
                for(unsigned int n = 0; n < graph.at(j)->neighbours.size(); n++)
                {
                    residual.segment(lastRow, 3) = (graph.at(j)->rotation * (graph.at(j)->neighbours.at(n)->position - graph.at(j)->position) +
                                                                           graph.at(j)->position + graph.at(j)->translation -
                                                                          (graph.at(j)->neighbours.at(n)->position + graph.at(j)->neighbours.at(n)->translation)) * sqrt(wReg);

                    lastRow += eRegRows;
                }
            }

            return residual;
        }

        Eigen::VectorXd sparseResidualCons(const int numRows)
        {
            //Now the residual
            Eigen::VectorXd residual(numRows);

            int lastRow = 0;

            Eigen::Vector3d position;
            Eigen::Vector3d normal;
            for(unsigned int l = 0; l < constraints.size(); l++)
            {
                //Compute desired position for cost
                computeVertexPosition(constraints.at(l).vertexId, position, normal);

                residual.segment(lastRow, 3) = (position - constraints.at(l).targetPosition) * sqrt(wCon);

                lastRow += eConRows;
            }

            return residual;
        }

};

#endif /* DEFORMATIONGRAPH_H_ */
