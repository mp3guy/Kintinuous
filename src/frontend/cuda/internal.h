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
 *
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef INTERNAL_HPP_
#define INTERNAL_HPP_

#include "containers/device_array.hpp"
#include "cuda_runtime_api.h"

#include <iostream>
#include <stdlib.h>

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif


#if defined(__GNUC__)
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__)
#endif

static inline void error(const char *error_string, const char *file, const int line, const char *func)
{
    std::cout << "Error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
        error(cudaGetErrorString(err), file, line, func);
}

static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }

struct DataTerm
{
    short2 zero;
    short2 one;
    float diff;
    bool valid;
};

struct JtJJtrSE3
{
    //27 floats for each product (27)
    float aa, ab, ac, ad, ae, af, ag,
              bb, bc, bd, be, bf, bg,
                  cc, cd, ce, cf, cg,
                      dd, de, df, dg,
                          ee, ef, eg,
                              ff, fg;

    //Extra data needed (29)
    float residual, inliers;

    __device__ inline void add(const JtJJtrSE3 & a)
    {
        aa += a.aa;
        ab += a.ab;
        ac += a.ac;
        ad += a.ad;
        ae += a.ae;
        af += a.af;
        ag += a.ag;

        bb += a.bb;
        bc += a.bc;
        bd += a.bd;
        be += a.be;
        bf += a.bf;
        bg += a.bg;

        cc += a.cc;
        cd += a.cd;
        ce += a.ce;
        cf += a.cf;
        cg += a.cg;

        dd += a.dd;
        de += a.de;
        df += a.df;
        dg += a.dg;

        ee += a.ee;
        ef += a.ef;
        eg += a.eg;

        ff += a.ff;
        fg += a.fg;

        residual += a.residual;
        inliers += a.inliers;
    }
};

struct PixelRGB
{
  unsigned char r, g, b;
};

struct PointXYZRGB
{
    union __attribute__((aligned(16)))
    {
      float data[4];
      struct
      {
        float x;
        float y;
        float z;
      };
    };

    union
    {
      union
      {
        struct
        {
                unsigned char b;
                unsigned char g;
                unsigned char r;
                unsigned char a;
        };
        float rgb;
      };
      int rgba;
    };
};

struct PointXYZRGBNormal
{
    union __attribute__((aligned(16)))
    {
        float data[4];
        struct
        {
          float x;
          float y;
          float z;
        };
    };

    union __attribute__((aligned(16)))
    {
        float data_n[4];
        float normal[3];
        struct
        {
            float normal_x;
            float normal_y;
            float normal_z;
        };
    };

    union
    {
        struct
        {
            union
            {
                union
                {
                    struct
                    {
                        unsigned char b;
                        unsigned char g;
                        unsigned char r;
                        unsigned char a;
                    };
                    float rgb;
                };
                int rgba;
            };
            float curvature;
        };
        float data_c[4];
    };
};

//Tsdf fixed point divisor (if old format is enabled)
const int DIVISOR = 32767;     // SHRT_MAX;

// Color update constants
// Sets point where normal angle reaches max weight
const float RGB_VIEW_ANGLE_WEIGHT = 0.75;

#define VOL 512

enum { VOLUME_X = VOL, VOLUME_Y = VOL, VOLUME_Z = VOL};

/** \brief Camera intrinsics structure
  */
struct Intr
{
  float fx, fy, cx, cy;
  Intr () : fx(0), fy(0), cx(0), cy(0) {}
  Intr (float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

  Intr operator()(int level_index) const
  {
    int div = 1 << level_index;
    return (Intr (fx / div, fy / div, cx / div, cy / div));
  }
};

struct IntrDoublePrecision
{
    double fx, fy, cx, cy;
    IntrDoublePrecision () : fx(0), fy(0), cx(0), cy(0) {}
    IntrDoublePrecision (double fx_, double fy_, double cx_, double cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    IntrDoublePrecision operator()(int level_index) const
    {
      int div = 1 << level_index;
      return (IntrDoublePrecision (fx / div, fy / div, cx / div, cy / div));
    }
};

/** \brief 3x3 Matrix for device code
  */
struct Mat33
{
  float3 data[3];
};

struct Mat33d
{
  double data[9];
};

/** \brief Light source collection
  */
struct LightSource
{
  float3 pos[1];
  int number;
};// Maps

/** \brief Perfoms bilateral filtering of disparity map
  * \param[in] src soruce map
  * \param[out] dst output map
  */
void bilateralFilter (const DeviceArray2D<unsigned short>& src, DeviceArray2D<unsigned short>& dst);

void sobelGaussian(DeviceArray2D<unsigned char>& src, DeviceArray2D<short>& dx, DeviceArray2D<short>& dy);

/** \brief Computes depth pyramid
  * \param[in] src source
  * \param[out] dst destination
  */
void pyrDown (const DeviceArray2D<unsigned short>& src, DeviceArray2D<unsigned short>& dst);

void pyrDownGaussF(const DeviceArray2D<float> & src, DeviceArray2D<float> & dst);

void pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src, DeviceArray2D<unsigned char> & dst);

void shortDepthToMetres(const DeviceArray2D<unsigned short>& src, DeviceArray2D<float> & dst, int cutOff);

void imageBGRToIntensity(const DeviceArray2D<PixelRGB> & src, DeviceArray2D<unsigned char> & dst);

void projectToPointCloud(const DeviceArray2D<float> & depth,
        const DeviceArray2D<float3> & cloud,
        IntrDoublePrecision & intrinsics,
        const int & level);

/** \brief Computes vertex map
  * \param[in] intr depth camera intrinsics
  * \param[in] depth depth
  * \param[out] vmap vertex map
  */
void
createVMap (const Intr& intr, const DeviceArray2D<unsigned short>& depth, DeviceArray2D<float>& vmap);

/** \brief Computes normal map using cross product
  * \param[in] vmap vertex map
  * \param[out] nmap normal map
  */
void
createNMap (const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap);

/** \brief Performs affine tranform of vertex and normal maps
  * \param[in] vmap_src source vertex map
  * \param[in] nmap_src source vertex map
  * \param[in] Rmat Rotation mat
  * \param[in] tvec translation
  * \param[out] vmap_dst destination vertex map
  * \param[out] nmap_dst destination vertex map
  */
void
tranformMaps (const DeviceArray2D<float>& vmap_src, const DeviceArray2D<float>& nmap_src, const Mat33& Rmat, const float3& tvec, DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst);//   ICP

/** \brief Perform tsdf volume initialization
  *  \param[out] array volume to be initialized
  */
void
initVolume(PtrStep<short> array);

void
clearVolumeX(PtrStep<short> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeXBack(PtrStep<short> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeY(PtrStep<short> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeYBack(PtrStep<short> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeZ(PtrStep<short> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeZBack(PtrStep<short> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeXc(PtrStep<uchar4> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeXBackc(PtrStep<uchar4> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeYc(PtrStep<uchar4> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeYBackc(PtrStep<uchar4> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeZc(PtrStep<uchar4> array, const int currentVoxelWrap, const int deltaVoxelWrap);

void
clearVolumeZBackc(PtrStep<uchar4> array, const int currentVoxelWrap, const int deltaVoxelWrap);

//second version
/** \brief Function that integrates volume if volume element contains: 2 bytes for round(tsdf*SHORT_MAX) and 2 bytes for integer weight.
  * \param[in] depth_raw depth image
  * \param[in] intr camera intrinsics
  * \param[in] volume_size size of volume in mm
  * \param[in] Rcurr_inv inverse rotation for current camera pose
  * \param[in] tcurr translation for current camera pose
  * \param[in] tranc_dist tsdf truncation distance
  * \param[in] volume tsdf volume to be updated
  * \param[out] depthRawScaled Buffer for scaled depth along ray
  */
void
integrateTsdfVolume (const PtrStepSz<unsigned short>& depth_raw, const Intr& intr, const float3& volume_size,
                     const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, PtrStep<short> volume, DeviceArray2D<float>& depthRawScaled,
                     const int3 & voxelWrap, PtrStep<uchar4> color_volume, PtrStepSz<uchar3> colors,
                     const DeviceArray2D<float>& nmap_curr,
                     bool angleColor);

/** \brief Initialzied color volume
  * \param[out] color_volume color volume for initialization
  */

void
initColorVolume(PtrStep<uchar4> color_volume);// Raycast and view generation
/** \brief Generation vertex and normal maps from volume for current camera pose
  * \param[in] intr camera intrinsices
  * \param[in] Rcurr current rotation
  * \param[in] tcurr current translation
  * \param[in] tranc_dist volume truncation distance
  * \param[in] volume_size volume size in mm
  * \param[in] volume tsdf volume
  * \param[out] vmap output vertex map
  * \param[out] nmap output normals map
  */
void
raycast (const Intr& intr, const Mat33& Rcurr, const float3& tcurr, float tranc_dist, const float3& volume_size,
         const PtrStep<short>& volume, DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap, const int3 & voxelWrap, DeviceArray2D<uchar4> & vmap_curr_color, PtrStep<uchar4> color_volume);

/** \brief Renders 3D image of the scene
  * \param[in] vmap vetex map
  * \param[in] nmap normals map
  * \param[in] light poase of light source
  * \param[out] dst buffer where image is generated
  */
void
generateImage (const DeviceArray2D<float>& vmap, const DeviceArray2D<float>& nmap, const DeviceArray2D<uchar4> & vmap_curr_color, const LightSource& light, PtrStepSz<uchar3> dst, PtrStepSz<uchar3> dstColor);/** \brief Renders depth image from give pose
  * \param[in] vmap inverse camera rotation
  * \param[in] nmap camera translation
  * \param[in] light vertex map
  * \param[out] dst buffer where depth is generated
  */
void
generateDepth (const Mat33& R_inv, const float3& t, const DeviceArray2D<float>& vmap, const DeviceArray2D<float>& nmap, DeviceArray2D<unsigned short>& dst, float maxDepth);

/** \brief Performs resize of vertex map to next pyramid level by averaging each four points
  * \param[in] input vertext map
  * \param[out] output resized vertex map
  */
void
resizeVMap (const DeviceArray2D<float>& input, DeviceArray2D<float>& output);

/** \brief Performs resize of vertex map to next pyramid level by averaging each four normals
  * \param[in] input normal map
  * \param[out] output vertex map
  */
void
resizeNMap (const DeviceArray2D<float>& input, DeviceArray2D<float>& output);// Cloud extraction

int GetGridDim(int D, int B);

/** \brief Perform point cloud extraction from tsdf volume
  * \param[in] volume tsdf volume
  * \param[in] volume_size size of the volume
  * \param[out] output buffer large enought to store point cloud
  * \return number of point stored to passed buffer
  */
size_t
extractCloudSlice(const PtrStep<short>& volume,
                  const float3& volume_size,
                  PtrSz<PointXYZRGB> output,
                  int3 voxelWrap,
                  PtrStep<uchar4> & color_volume,
                  int minX,
                  int maxX,
                  int minY,
                  int maxY,
                  int minZ,
                  int maxZ,
                  int subsample,
                  int3 realVoxelWrap);

template<class D, class Matx> D&
device_cast (Matx& matx)
{
  return (*reinterpret_cast<D*>(matx.data ()));
}

void icpStep(const Mat33& Rcurr,
             const float3& tcurr,
             const DeviceArray2D<float>& vmap_curr,
             const DeviceArray2D<float>& nmap_curr,
             const Mat33& Rprev_inv,
             const float3& tprev,
             const Intr& intr,
             const DeviceArray2D<float>& vmap_g_prev,
             const DeviceArray2D<float>& nmap_g_prev,
             float distThres,
             float angleThres,
             DeviceArray<JtJJtrSE3> & sum,
             DeviceArray<JtJJtrSE3> & out,
             float * matrixA_host,
             float * vectorB_host,
             float * residual_host,
             int threads,
             int blocks);

void rgbStep(const DeviceArray2D<DataTerm> & corresImg,
             const float & sigma,
             const DeviceArray2D<float3> & cloud,
             const float & fx,
             const float & fy,
             const DeviceArray2D<short> & dIdx,
             const DeviceArray2D<short> & dIdy,
             const float & sobelScale,
             DeviceArray<JtJJtrSE3> & sum,
             DeviceArray<JtJJtrSE3> & out,
             float * matrixA_host,
             float * vectorB_host,
             int threads,
             int blocks);

void computeRgbResidual(const float & minScale,
                        const DeviceArray2D<short> & dIdx,
                        const DeviceArray2D<short> & dIdy,
                        const DeviceArray2D<float> & lastDepth,
                        const DeviceArray2D<float> & nextDepth,
                        const DeviceArray2D<unsigned char> & lastImage,
                        const DeviceArray2D<unsigned char> & nextImage,
                        DeviceArray2D<DataTerm> & corresImg,
                        DeviceArray<int2> & sumResidual,
                        const float maxDepthDelta,
                        const float3 & kt,
                        const Mat33 & krkinv,
                        int & sigmaSum,
                        int & count,
                        int threads,
                        int blocks);


#endif /* INTERNAL_HPP_ */
