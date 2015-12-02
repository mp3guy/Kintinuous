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

#include "DeformationGraph.h"

DeformationGraph::DeformationGraph(int k)
 : initialised(false),
   k(k),
   wRot(1),
   wReg(10),
   wCon(100),
   graphNodes(0),
   sourceVertices(0),
   graphCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
   lastPointCount(0)
{}

DeformationGraph::~DeformationGraph()
{
    if(initialised)
    {
        delete [] graphNodes;
    }
}

std::vector<GraphNode *> & DeformationGraph::getGraph()
{
    return graph;
}

std::vector<std::vector<DeformationGraph::VertexWeightMap> > & DeformationGraph::getVertexMap()
{
    return vertexMap;
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr DeformationGraph::initialiseGraphPoses(pcl::PointCloud<pcl::PointXYZRGBNormal> * vertices,
                                                                                    float poseDist,
                                                                                    pcl::PointCloud<pcl::PointXYZRGBNormal> * customGraph,
                                                                                    std::vector<uint64_t> * graphTimeMap,
                                                                                    std::vector<uint64_t> * vertexTimeMap,
                                                                                    unsigned int originalPointEnd)
{
    assert(!initialised);

    sourceVertices = vertices;

    graphCloud->push_back(customGraph->at(0));
    sampledGraphTimes.push_back(graphTimeMap->at(0));

    //NOTE: Skipping poses here is crucial, oversampling looks bad
    for(unsigned int i = 1; i < customGraph->size(); i++)
    {
        if((graphCloud->at(graphCloud->size() - 1).getVector3fMap() - customGraph->at(i).getVector3fMap()).norm() > poseDist)
        {
            graphCloud->push_back(customGraph->at(i));
            sampledGraphTimes.push_back(graphTimeMap->at(i));
        }
    }

    graphNodes = new GraphNode[graphCloud->size()];

    connectGraphSeq(graphCloud);

    weightVerticesSeq(graphCloud, vertexTimeMap, sampledGraphTimes);

    initialised = true;

    lastPointCount = originalPointEnd;

    return graphCloud;
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr DeformationGraph::initialiseGraphPosesNN(pcl::PointCloud<pcl::PointXYZRGBNormal> * vertices,
                                                                                      const double targetSpacing,
                                                                                      std::vector<uint64_t> * vertexTimeMap,
                                                                                      unsigned int originalPointEnd)
{
    assert(!initialised);

    sourceVertices = vertices;

    std::vector<uint64_t> graphTimeMap;

    TICK("sample");
    radiusSampleTemporal(graphCloud, vertexTimeMap, &graphTimeMap, targetSpacing);
    TOCK("sample");

    graphNodes = new GraphNode[graphCloud->size()];

    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> graphTree(true);
    graphTree.setInputCloud(graphCloud);

    TICK("connect");
    connectGraphNNTemporal(graphCloud, &graphTimeMap, graphTree);
    TOCK("connect");

    TICK("vertices");
    weightVerticesNNTemporal(graphTree, vertexTimeMap, &graphTimeMap);
    TOCK("vertices");

    initialised = true;

    lastPointCount = originalPointEnd;

    return graphCloud;
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr DeformationGraph::appendGraphPoses(float poseDist,
                                                                                pcl::PointCloud<pcl::PointXYZRGBNormal> * customGraph,
                                                                                std::vector<uint64_t> * graphTimeMap,
                                                                                std::vector<uint64_t> * vertexTimeMap,
                                                                                unsigned int originalPointEnd)
{
    assert(initialised);

    //Find appendage and update poses based on iSAM
    unsigned int startIndex = 0;
    unsigned int currentNodeIndex = 0;
    for(startIndex = 0; startIndex < graphTimeMap->size(); startIndex++)
    {
        if(graphTimeMap->at(startIndex) == sampledGraphTimes.at(currentNodeIndex))
        {
            graphCloud->at(currentNodeIndex).x = customGraph->at(startIndex).x;
            graphCloud->at(currentNodeIndex).y = customGraph->at(startIndex).y;
            graphCloud->at(currentNodeIndex).z = customGraph->at(startIndex).z;
            currentNodeIndex++;
        }

        if(graphTimeMap->at(startIndex) == sampledGraphTimes.back())
        {
            //We've reached where we were last time, break;
            break;
        }
    }

    assert(currentNodeIndex == graphCloud->size());

    //Append new graph pose nodes
    for(unsigned int i = startIndex + 1; i < customGraph->size(); i++)
    {
        if((graphCloud->at(graphCloud->size() - 1).getVector3fMap() - customGraph->at(i).getVector3fMap()).norm() > poseDist)
        {
            graphCloud->push_back(customGraph->at(i));
            sampledGraphTimes.push_back(graphTimeMap->at(i));
        }
    }

    assert(graphNodes);

    delete [] graphNodes;

    graph.clear();

    graphNodes = new GraphNode[graphCloud->size()];

    connectGraphSeq(graphCloud);

    vertexMap.resize(lastPointCount);

    weightVerticesSeq(graphCloud, vertexTimeMap, sampledGraphTimes);

    lastPointCount = originalPointEnd;

    return graphCloud;
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr DeformationGraph::appendVertices(std::vector<uint64_t> * vertexTimeMap,
                                                                              unsigned int originalPointEnd)
{
    vertexMap.resize(lastPointCount);

    weightVerticesSeq(graphCloud, vertexTimeMap, sampledGraphTimes);

    lastPointCount = originalPointEnd;

    return graphCloud;
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr DeformationGraph::initialiseGraphNN(pcl::PointCloud<pcl::PointXYZRGBNormal> * vertices,
                                                                                 const double targetSpacing)
{
    assert(!initialised);

    sourceVertices = vertices;

    radiusSample(graphCloud, targetSpacing);

    graphNodes = new GraphNode[graphCloud->size()];

    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> graphTree;
    graphTree.setInputCloud(graphCloud);

    connectGraphNN(graphCloud, graphTree);

    weightVerticesNN(graphTree);

    initialised = true;

    return graphCloud;
}

void DeformationGraph::connectGraphSeq(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud)
{
    //Initialises graph nodes and connects neighbours
    for(unsigned int i = 0; i < graphCloud->size(); i++)
    {
        graphNodes[i].id = i;

        graphNodes[i].position(0) = graphCloud->at(i).x;
        graphNodes[i].position(1) = graphCloud->at(i).y;
        graphNodes[i].position(2) = graphCloud->at(i).z;

        graphNodes[i].translation(0) = 0;
        graphNodes[i].translation(1) = 0;
        graphNodes[i].translation(2) = 0;

        graphNodes[i].rotation.setIdentity();

        graph.push_back(&graphNodes[i]);
    }

    for(int i = 0; i < k / 2; i++)
    {
        for(int n = 0; n < k + 1; n++)
        {
            if(i == n)
            {
                continue;
            }

            graphNodes[i].neighbours.push_back(&graphNodes[n]);
        }
    }

    for(unsigned int i = k / 2; i < graphCloud->size() - (k / 2); i++)
    {
        for(int n = 0; n < k / 2; n++)
        {
            graphNodes[i].neighbours.push_back(&graphNodes[i - (n + 1)]);
            graphNodes[i].neighbours.push_back(&graphNodes[i + (n + 1)]);
        }
    }

    for(unsigned int i = graphCloud->size() - (k / 2); i < graphCloud->size(); i++)
    {
        for(unsigned int n = graphCloud->size() - (k + 1); n < graphCloud->size(); n++)
        {
            if(i == n)
            {
                continue;
            }

            graphNodes[i].neighbours.push_back(&graphNodes[n]);
        }
    }
}

void DeformationGraph::connectGraphNN(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud,
                                      pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> & graphTree)
{
    //k + 1 because the nearest neighbour is a node itself
    std::vector<int> pointIdxNKNSearch(k + 1);
    std::vector<float> pointNKNSquaredDistance(k + 1);

    //Initialises graph nodes and connects neighbours
    for(unsigned int i = 0; i < graphCloud->size(); i++)
    {
        graphNodes[i].id = i;

        graphNodes[i].position(0) = graphCloud->at(i).x;
        graphNodes[i].position(1) = graphCloud->at(i).y;
        graphNodes[i].position(2) = graphCloud->at(i).z;

        graphNodes[i].translation(0) = 0;
        graphNodes[i].translation(1) = 0;
        graphNodes[i].translation(2) = 0;

        graphNodes[i].rotation.setIdentity();

        assert(graphTree.nearestKSearch(graphCloud->at(i), k + 1, pointIdxNKNSearch, pointNKNSquaredDistance) == k + 1);

        unsigned int j = 0;
        int numNeighboursAdded = 0;

        while(numNeighboursAdded < k)
        {
            if(pointIdxNKNSearch.at(j) == (int)i)
            {
                j++;
                continue;
            }

            graphNodes[i].neighbours.push_back(&graphNodes[pointIdxNKNSearch.at(j)]);

            j++;
            numNeighboursAdded++;
        }

        graph.push_back(&graphNodes[i]);
    }
}

void DeformationGraph::connectGraphNNTemporal(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud,
                                              std::vector<uint64_t> * graphTimeMap,
                                              pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> & graphTree)
{
    int sampleCount = k * 4;
    //Lets have redundancy
    std::vector<int> pointIdxNKNSearch(sampleCount);
    std::vector<float> pointNKNSquaredDistance(sampleCount);

    assert(graphCloud->size() == graphTimeMap->size());

    //Initialises graph nodes and connects neighbours
    for(unsigned int i = 0; i < graphCloud->size(); i++)
    {
        graphNodes[i].id = i;

        graphNodes[i].position(0) = graphCloud->at(i).x;
        graphNodes[i].position(1) = graphCloud->at(i).y;
        graphNodes[i].position(2) = graphCloud->at(i).z;

        graphNodes[i].translation(0) = 0;
        graphNodes[i].translation(1) = 0;
        graphNodes[i].translation(2) = 0;

        graphNodes[i].rotation.setIdentity();

        assert(graphTree.nearestKSearch(graphCloud->at(i), sampleCount, pointIdxNKNSearch, pointNKNSquaredDistance) == sampleCount);

        unsigned int j = 0;
        int numNeighboursAdded = 0;

        while(numNeighboursAdded < k)
        {
            if(pointIdxNKNSearch.at(j) == (int)i)
            {
                j++;
                continue;
            }

            //Only connect if less than a minute apart
            if(abs(graphTimeMap->at(pointIdxNKNSearch.at(j)) - graphTimeMap->at(i)) < 60000000)
            {
                graphNodes[i].neighbours.push_back(&graphNodes[pointIdxNKNSearch.at(j)]);
                numNeighboursAdded++;
            }

            j++;
        }

        graph.push_back(&graphNodes[i]);
    }
}

void DeformationGraph::radiusSample(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud, const double targetSpacing)
{
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>sourceTree(false);

    sourceTree.setInputCloud(sourceVertices->makeShared());

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    bool * used = new bool[sourceVertices->size()];

    memset(&used[0], 0, sizeof(bool) * sourceVertices->size());

    for(unsigned int i = 0; i < sourceVertices->size(); i++)
    {
        if(!used[i])
        {
            if(sourceTree.radiusSearch(sourceVertices->at(i), targetSpacing, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
            {
                graphCloud->push_back(sourceVertices->at(i));

                for(unsigned int j = 0; j < pointIdxRadiusSearch.size(); j++)
                {
                    used[pointIdxRadiusSearch.at(j)] = true;
                }
            }
        }
    }

    delete [] used;
}

void DeformationGraph::radiusSampleTemporal(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud,
                                            std::vector<uint64_t> * vertexTimeMap,
                                            std::vector<uint64_t> * graphTimeMap,
                                            const double targetSpacing)
{
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>sourceTree(false);

    sourceTree.setInputCloud(sourceVertices->makeShared());

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    assert(vertexTimeMap->size() == sourceVertices->size());

    bool * used = new bool[sourceVertices->size()];

    memset(&used[0], 0, sizeof(bool) * sourceVertices->size());

    for(unsigned int i = 0; i < sourceVertices->size(); i++)
    {
        if(!used[i])
        {
            if(sourceTree.radiusSearch(sourceVertices->at(i), targetSpacing, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
            {
                graphCloud->push_back(sourceVertices->at(i));
                graphTimeMap->push_back(vertexTimeMap->at(i));

                for(unsigned int j = 0; j < pointIdxRadiusSearch.size(); j++)
                {
                    used[pointIdxRadiusSearch.at(j)] = true;
                }
            }
        }
    }

    delete [] used;
}

void DeformationGraph::weightVerticesSeq(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr graphCloud,
                                         std::vector<uint64_t> * vertexTimeMap,
                                         std::vector<uint64_t> & sampledGraphTimes)
{
    const unsigned int lookBack = 20;

    std::vector<int> pointIdxNKNSearch(k + 1);
    std::vector<float> pointNKNSquaredDistance(k + 1);

    for(unsigned int i = lastPointCount; i < sourceVertices->size(); i++)
    {
        uint64_t vertexTime = vertexTimeMap->at(i);

        unsigned int foundIndex = 0;

        int imin = 0;
        int imax = sampledGraphTimes.size() - 1;
        int imid = (imin + imax) / 2;

        while(imax >= imin)
        {
            imid = (imin + imax) / 2;

            if (sampledGraphTimes[imid] < vertexTime)
            {
                imin = imid + 1;
            }
            else if(sampledGraphTimes[imid] > vertexTime)
            {
                imax = imid - 1;
            }
            else
            {
                break;
            }
        }

        if(abs(sampledGraphTimes[imin] - vertexTime) <= abs(sampledGraphTimes[imid] - vertexTime) &&
           abs(sampledGraphTimes[imin] - vertexTime) <= abs(sampledGraphTimes[imax] - vertexTime))
        {
            foundIndex = imin;
        }
        else if(abs(sampledGraphTimes[imid] - vertexTime) <= abs(sampledGraphTimes[imin] - vertexTime) &&
                abs(sampledGraphTimes[imid] - vertexTime) <= abs(sampledGraphTimes[imax] - vertexTime))
        {
            foundIndex = imid;
        }
        else
        {
            foundIndex = imax;
        }

        std::vector<std::pair<float, int> > nearNodes;

        if(foundIndex == graphCloud->size())
        {
            foundIndex = graphCloud->size() - 1;
        }

        unsigned int distanceBack = 0;
        for(int j = (int)foundIndex; j >= 0; j--)
        {
            std::pair<float, int> newNode;
            newNode.first = (graphCloud->at(j).getVector3fMap() - sourceVertices->at(i).getVector3fMap()).norm();
            newNode.second = j;

            nearNodes.push_back(newNode);

            if(++distanceBack == lookBack)
            {
                break;
            }
        }

        if(distanceBack != lookBack)
        {
            for(unsigned int j = foundIndex + 1; j < sampledGraphTimes.size(); j++)
            {
                std::pair<float, int> newNode;
                newNode.first = (graphCloud->at(j).getVector3fMap() - sourceVertices->at(i).getVector3fMap()).norm();
                newNode.second = j;

                nearNodes.push_back(newNode);

                if(++distanceBack == lookBack)
                {
                    break;
                }
            }
        }

        std::sort(nearNodes.begin(), nearNodes.end(), boost::bind(&std::pair<float, int>::first, _1) < boost::bind(&std::pair<float, int>::first, _2));

        Eigen::Vector3d vertexPosition(sourceVertices->at(i).x, sourceVertices->at(i).y, sourceVertices->at(i).z);
        double dMax = nearNodes.at(k).first;

        std::vector<VertexWeightMap> newMap;

        double weightSum = 0;

        for(unsigned int j = 0; j < (unsigned int)k; j++)
        {
            newMap.push_back(VertexWeightMap(pow(1.0f - (vertexPosition - graphNodes[nearNodes.at(j).second].position).norm() / dMax, 2), nearNodes.at(j).second));
            weightSum += newMap.back().weight;
        }

        for(unsigned int j = 0; j < newMap.size(); j++)
        {
            newMap.at(j).weight /= weightSum;
        }

        VertexWeightMap::sort(newMap, graph);

        vertexMap.push_back(newMap);
    }
}

void DeformationGraph::weightVerticesNN(pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> & graphTree)
{
    std::vector<int> pointIdxNKNSearch(k + 1);
    std::vector<float> pointNKNSquaredDistance(k + 1);

    for(unsigned int i = 0; i < sourceVertices->size(); i++)
    {
        assert(graphTree.nearestKSearch(sourceVertices->at(i), k + 1, pointIdxNKNSearch, pointNKNSquaredDistance) == k + 1);

        Eigen::Vector3d vertexPosition(sourceVertices->at(i).x, sourceVertices->at(i).y, sourceVertices->at(i).z);
        double dMax = sqrt(pointNKNSquaredDistance.back());

        std::vector<VertexWeightMap> newMap;

        double weightSum = 0;

        for(unsigned int j = 0; j < pointIdxNKNSearch.size() - 1; j++)
        {
            newMap.push_back(VertexWeightMap(pow(1.0f - (vertexPosition - graphNodes[pointIdxNKNSearch.at(j)].position).norm() / dMax, 2), pointIdxNKNSearch.at(j)));
            weightSum += newMap.back().weight;
        }

        for(unsigned int j = 0; j < newMap.size(); j++)
        {
            newMap.at(j).weight /= weightSum;
        }

        VertexWeightMap::sort(newMap, graph);

        vertexMap.push_back(newMap);
    }
}

void DeformationGraph::weightVerticesNNTemporal(pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> & graphTree,
                                                std::vector<uint64_t> * vertexTimeMap,
                                                std::vector<uint64_t> * graphTimeMap)
{
    int sampleCount = k * 4;

    std::vector<int> pointIdxNKNSearch(sampleCount);
    std::vector<float> pointNKNSquaredDistance(sampleCount);

    std::vector<int> validIndices;
    std::vector<float> validDistances;

    for(unsigned int i = 0; i < sourceVertices->size(); i++)
    {
        validIndices.clear();
        validDistances.clear();

        assert(graphTree.nearestKSearch(sourceVertices->at(i), sampleCount, pointIdxNKNSearch, pointNKNSquaredDistance) == sampleCount);

        //Assume sorted kNN
        for(unsigned int j = 0; j < pointIdxNKNSearch.size() && (int)validIndices.size() < k + 1; j++)
        {
            if(abs(graphTimeMap->at(pointIdxNKNSearch.at(j)) - vertexTimeMap->at(i)) < 60000000)
            {
                validIndices.push_back(pointIdxNKNSearch.at(j));
                validDistances.push_back(pointNKNSquaredDistance.at(j));
            }
        }

        Eigen::Vector3d vertexPosition(sourceVertices->at(i).x, sourceVertices->at(i).y, sourceVertices->at(i).z);
        double dMax = sqrt(validDistances.back());

        std::vector<VertexWeightMap> newMap;

        double weightSum = 0;

        for(unsigned int j = 0; j < validIndices.size() - 1; j++)
        {
            newMap.push_back(VertexWeightMap(pow(1.0f - (vertexPosition - graphNodes[validIndices.at(j)].position).norm() / dMax, 2), validIndices.at(j)));
            weightSum += newMap.back().weight;
        }

        for(unsigned int j = 0; j < newMap.size(); j++)
        {
            newMap.at(j).weight /= weightSum;
        }

        VertexWeightMap::sort(newMap, graph);

        vertexMap.push_back(newMap);
    }
}

void DeformationGraph::applyGraphToVertices(int numThreads)
{
    assert(initialised);

    TICK("apply");
    boost::thread_group threads;
    for(int i = 0; i < numThreads; i++)
    {
        threads.add_thread(new boost::thread(boost::bind(&DeformationGraph::applyGraphThread, this, i, numThreads)));
    }
    threads.join_all();
    TOCK("apply");

    Stopwatch::get().sendAll();
}

void DeformationGraph::applyGraphThread(int id, int numThreads)
{
    Eigen::Vector3d position;
    Eigen::Vector3d normal;

    for(unsigned int i = id; i < sourceVertices->size(); i += numThreads)
    {
        computeVertexPosition(i, position, normal);

        sourceVertices->at(i).x = position(0);
        sourceVertices->at(i).y = position(1);
        sourceVertices->at(i).z = position(2);

        sourceVertices->at(i).normal_x = normal(0);
        sourceVertices->at(i).normal_y = normal(1);
        sourceVertices->at(i).normal_z = normal(2);
    }
}

void DeformationGraph::addConstraint(int vertexId, Eigen::Vector3d & target)
{
    assert(initialised);

    for(unsigned int i = 0; i < constraints.size(); i++)
    {
        if(constraints.at(i).vertexId == vertexId)
        {
            constraints.at(i) = Constraint(vertexId, target);
            return;
        }
    }

    constraints.push_back(Constraint(vertexId, target));
}

void DeformationGraph::removeConstraint(int vertexId)
{
    assert(initialised);

    for(unsigned int i = 0; i < constraints.size(); i++)
    {
        if(constraints.at(i).vertexId == vertexId)
        {
            constraints.erase(constraints.begin() + i);
            break;
        }
    }
}

void DeformationGraph::clearConstraints()
{
    constraints.clear();
}

void DeformationGraph::optimiseGraphSparse()
{
    assert(initialised);

    TICK("opt");
    //6 per E_rot, 3 * k per E_reg and 3 * p per E_con
    const int rows = (eRotRows + eRegRows * k) * graph.size() + eConRows * constraints.size();

    Eigen::VectorXd rCon = sparseResidualCons(eConRows * constraints.size());

    float graphError = rCon.norm() / constraints.size();

    if(graphError < 0.1)
    {
        std::cout << "Not deforming, constraint error insignificant (" << graphError << ")" << std::endl;
        return;
    }

    Eigen::VectorXd residual = sparseResidual(rows);

    Jacobian jacobian;

    sparseJacobian(jacobian, residual.rows(), numVariables * graph.size());

    double error = residual.squaredNorm();
    double lastError = error;
    double errorDiff = 0;

    std::cout << "Initial error: " << error << " (" << graphError << ")" << std::endl;

    int iter = 0;

    while(iter++ < 10)
    {
        Eigen::VectorXd delta = cholesky.solve(jacobian, -residual, iter == 1);

        applyDeltaSparse(delta);

        residual = sparseResidual(rows);

        error = residual.squaredNorm();

        errorDiff = error - lastError;

        std::cout << "Iteration " << iter << ": " << error << std::endl;

        if(delta.norm() < 1e-2 || error < 1e-3 || fabs(errorDiff) < 1e-5 * error)
        {
            break;
        }

        lastError = error;

        sparseJacobian(jacobian, residual.rows(), numVariables * graph.size());
    }

    cholesky.freeFactor();

    TOCK("opt");
    Stopwatch::get().sendAll();
}

void DeformationGraph::sparseJacobian(Jacobian & jacobian, const int numRows, const int numCols)
{
    std::vector<OrderedJacobianRow*> rows(numRows);

    //We know exact counts per row...
    int lastRow = 0;

    for(unsigned int j = 0; j < graph.size(); j++)
    {
        //No weights for rotation as rotation weight = 1
        const Eigen::Matrix3d & rotation = graph.at(j)->rotation;

        int colOffset = graph.at(j)->id * numVariables;

        rows[lastRow] = new OrderedJacobianRow(6);
        rows[lastRow + 1] = new OrderedJacobianRow(6);
        rows[lastRow + 2] = new OrderedJacobianRow(6);
        rows[lastRow + 3] = new OrderedJacobianRow(3);
        rows[lastRow + 4] = new OrderedJacobianRow(3);
        rows[lastRow + 5] = new OrderedJacobianRow(3);

        rows[lastRow]->append(colOffset, rotation(0, 1));
        rows[lastRow]->append(colOffset + 1, rotation(1, 1));
        rows[lastRow]->append(colOffset + 2, rotation(2, 1));
        rows[lastRow]->append(colOffset + 3, rotation(0, 0));
        rows[lastRow]->append(colOffset + 4, rotation(1, 0));
        rows[lastRow]->append(colOffset + 5, rotation(2, 0));

        rows[lastRow + 1]->append(colOffset, rotation(0, 2));
        rows[lastRow + 1]->append(colOffset + 1, rotation(1, 2));
        rows[lastRow + 1]->append(colOffset + 2, rotation(2, 2));
        rows[lastRow + 1]->append(colOffset + 6, rotation(0, 0));
        rows[lastRow + 1]->append(colOffset + 7, rotation(1, 0));
        rows[lastRow + 1]->append(colOffset + 8, rotation(2, 0));

        rows[lastRow + 2]->append(colOffset + 3, rotation(0, 2));
        rows[lastRow + 2]->append(colOffset + 4, rotation(1, 2));
        rows[lastRow + 2]->append(colOffset + 5, rotation(2, 2));
        rows[lastRow + 2]->append(colOffset + 6, rotation(0, 1));
        rows[lastRow + 2]->append(colOffset + 7, rotation(1, 1));
        rows[lastRow + 2]->append(colOffset + 8, rotation(2, 1));

        rows[lastRow + 3]->append(colOffset, 2*rotation(0, 0));
        rows[lastRow + 3]->append(colOffset + 1, 2*rotation(1, 0));
        rows[lastRow + 3]->append(colOffset + 2, 2*rotation(2, 0));

        rows[lastRow + 4]->append(colOffset + 3, 2*rotation(0, 1));
        rows[lastRow + 4]->append(colOffset + 4, 2*rotation(1, 1));
        rows[lastRow + 4]->append(colOffset + 5, 2*rotation(2, 1));

        rows[lastRow + 5]->append(colOffset + 6, 2*rotation(0, 2));
        rows[lastRow + 5]->append(colOffset + 7, 2*rotation(1, 2));
        rows[lastRow + 5]->append(colOffset + 8, 2*rotation(2, 2));

        lastRow += eRotRows;
    }

    for(unsigned int j = 0; j < graph.size(); j++)
    {
        int colOffset = graph.at(j)->id * numVariables;

        //For each neighbour
        for(unsigned int n = 0; n < graph.at(j)->neighbours.size(); n++)
        {
            rows[lastRow] = new OrderedJacobianRow(5);
            rows[lastRow + 1] = new OrderedJacobianRow(5);
            rows[lastRow + 2] = new OrderedJacobianRow(5);

            Eigen::Vector3d delta = graph.at(j)->neighbours.at(n)->position - graph.at(j)->position;

            int colOffsetN = graph.at(j)->neighbours.at(n)->id * numVariables;

            assert(colOffset != colOffsetN);

            if(colOffsetN < colOffset)
            {
                rows[lastRow]->append(colOffsetN + 9, -1.0 * sqrt(wReg));
                rows[lastRow + 1]->append(colOffsetN + 10, -1.0 * sqrt(wReg));
                rows[lastRow + 2]->append(colOffsetN + 11, -1.0 * sqrt(wReg));
            }

            rows[lastRow]->append(colOffset, delta(0) * sqrt(wReg));
            rows[lastRow]->append(colOffset + 3, delta(1) * sqrt(wReg));
            rows[lastRow]->append(colOffset + 6, delta(2) * sqrt(wReg));
            rows[lastRow]->append(colOffset + 9, 1.0 * sqrt(wReg));

            rows[lastRow + 1]->append(colOffset + 1, delta(0) * sqrt(wReg));
            rows[lastRow + 1]->append(colOffset + 4, delta(1) * sqrt(wReg));
            rows[lastRow + 1]->append(colOffset + 7, delta(2) * sqrt(wReg));
            rows[lastRow + 1]->append(colOffset + 10, 1.0 * sqrt(wReg));

            rows[lastRow + 2]->append(colOffset + 2, delta(0) * sqrt(wReg));
            rows[lastRow + 2]->append(colOffset + 5, delta(1) * sqrt(wReg));
            rows[lastRow + 2]->append(colOffset + 8, delta(2) * sqrt(wReg));
            rows[lastRow + 2]->append(colOffset + 11, 1.0 * sqrt(wReg));

            if(colOffsetN > colOffset)
            {
                rows[lastRow]->append(colOffsetN + 9, -1.0 * sqrt(wReg));
                rows[lastRow + 1]->append(colOffsetN + 10, -1.0 * sqrt(wReg));
                rows[lastRow + 2]->append(colOffsetN + 11, -1.0 * sqrt(wReg));
            }

            lastRow += eRegRows;
        }
    }

    for(unsigned int l = 0; l < constraints.size(); l++)
    {
        //For each k-node we have a weight
        const std::vector<VertexWeightMap> & weightMap = vertexMap.at(constraints.at(l).vertexId);

        Eigen::Vector3d sourcePosition(sourceVertices->at(constraints.at(l).vertexId).x,
                                       sourceVertices->at(constraints.at(l).vertexId).y,
                                       sourceVertices->at(constraints.at(l).vertexId).z);

        rows[lastRow] = new OrderedJacobianRow(4 * k * 2);
        rows[lastRow + 1] = new OrderedJacobianRow(4 * k * 2);
        rows[lastRow + 2] = new OrderedJacobianRow(4 * k * 2);

        assert(graph.at(weightMap.at(0).node)->id < graph.at(weightMap.at(1).node)->id);

        //Populate each column on the current Jacobian block rows
        //WARNING: Assumes weightMap is sorted by id!
        for(unsigned int i = 0; i < weightMap.size(); i++)
        {
            int colOffset = graph.at(weightMap.at(i).node)->id * numVariables;

            Eigen::Vector3d delta = (sourcePosition - graph.at(weightMap.at(i).node)->position) * weightMap.at(i).weight;

            rows[lastRow]->append(colOffset, delta(0) * sqrt(wCon));
            rows[lastRow]->append(colOffset + 3, delta(1) * sqrt(wCon));
            rows[lastRow]->append(colOffset + 6, delta(2) * sqrt(wCon));
            rows[lastRow]->append(colOffset + 9, weightMap.at(i).weight * sqrt(wCon));

            rows[lastRow + 1]->append(colOffset + 1, delta(0) * sqrt(wCon));
            rows[lastRow + 1]->append(colOffset + 4, delta(1) * sqrt(wCon));
            rows[lastRow + 1]->append(colOffset + 7, delta(2) * sqrt(wCon));
            rows[lastRow + 1]->append(colOffset + 10, weightMap.at(i).weight * sqrt(wCon));

            rows[lastRow + 2]->append(colOffset + 2, delta(0) * sqrt(wCon));
            rows[lastRow + 2]->append(colOffset + 5, delta(1) * sqrt(wCon));
            rows[lastRow + 2]->append(colOffset + 8, delta(2) * sqrt(wCon));
            rows[lastRow + 2]->append(colOffset + 11, weightMap.at(i).weight * sqrt(wCon));
        }

        lastRow += eConRows;
    }

    assert(lastRow == numRows);

    jacobian.assign(rows, numCols);
}

Eigen::VectorXd DeformationGraph::sparseResidual(const int numRows)
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

void DeformationGraph::resetGraph()
{
    for(unsigned int j = 0; j < graph.size(); j++)
    {
        graph.at(j)->rotation.setIdentity();
        graph.at(j)->translation.setIdentity();
    }
}

void DeformationGraph::applyDeltaSparse(Eigen::VectorXd & delta)
{
    assert(initialised);

    //Current row
    int z = 0;

    for(unsigned int j = 0; j < graph.size(); j++)
    {
        const_cast<double *>(graph.at(j)->rotation.data())[0] += delta(z + 0);
        const_cast<double *>(graph.at(j)->rotation.data())[1] += delta(z + 1);
        const_cast<double *>(graph.at(j)->rotation.data())[2] += delta(z + 2);

        const_cast<double *>(graph.at(j)->rotation.data())[3] += delta(z + 3);
        const_cast<double *>(graph.at(j)->rotation.data())[4] += delta(z + 4);
        const_cast<double *>(graph.at(j)->rotation.data())[5] += delta(z + 5);

        const_cast<double *>(graph.at(j)->rotation.data())[6] += delta(z + 6);
        const_cast<double *>(graph.at(j)->rotation.data())[7] += delta(z + 7);
        const_cast<double *>(graph.at(j)->rotation.data())[8] += delta(z + 8);

        const_cast<double *>(graph.at(j)->translation.data())[0] += delta(z + 9);
        const_cast<double *>(graph.at(j)->translation.data())[1] += delta(z + 10);
        const_cast<double *>(graph.at(j)->translation.data())[2] += delta(z + 11);

        z += numVariables;
    }
}

void DeformationGraph::computeVertexPosition(int vertexId, Eigen::Vector3d & position, Eigen::Vector3d & normal)
{
    assert(initialised);

    std::vector<VertexWeightMap> & weightMap = vertexMap.at(vertexId);

    position(0) = 0;
    position(1) = 0;
    position(2) = 0;

    normal(0) = 0;
    normal(1) = 0;
    normal(2) = 0;

    Eigen::Vector3d sourcePosition(sourceVertices->at(vertexId).x, sourceVertices->at(vertexId).y, sourceVertices->at(vertexId).z);
    Eigen::Vector3d sourceNormal(sourceVertices->at(vertexId).normal_x, sourceVertices->at(vertexId).normal_y, sourceVertices->at(vertexId).normal_z);

    for(unsigned int i = 0; i < weightMap.size(); i++)
    {
        position += weightMap.at(i).weight * (graph.at(weightMap.at(i).node)->rotation * (sourcePosition - graph.at(weightMap.at(i).node)->position) +
                                              graph.at(weightMap.at(i).node)->position + graph.at(weightMap.at(i).node)->translation);

        normal += weightMap.at(i).weight * (graph.at(weightMap.at(i).node)->rotation.inverse().transpose() * sourceNormal);
    }

    normal.normalize();
}
