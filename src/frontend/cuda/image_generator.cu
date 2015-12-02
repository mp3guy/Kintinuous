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

#include "device.hpp"

struct ImageGenerator
{
  enum
  {
    CTA_SIZE_X = 32, CTA_SIZE_Y = 8
  };

  PtrStep<float> vmap;
  PtrStep<float> nmap;
  PtrStepSz<uchar4> vmap_curr_color;

  LightSource light;

  mutable PtrStepSz<uchar3> dstColor;
  mutable PtrStepSz<uchar3> dst;

  __device__ __forceinline__ void
  getHeatMapColor(float value, int & red, int & green, int & blue) const
  {
      const int NUM_COLORS = 4;
      float color[NUM_COLORS][3] = {{0,0,1}, {0,1,0}, {1,1,0}, {1,0,0}};

      int idx1;
      int idx2;
      float fractBetween = 0;

      if(value <= 0)
      {
          idx1 = idx2 = 0;
      }
      else if(value >= 1)
      {
          idx1 = idx2 = NUM_COLORS - 1;
      }
      else
      {
          value = value * (NUM_COLORS - 1);
          idx1 = floor(value);
          idx2 = idx1 + 1;
          fractBetween = value - float(idx1);
      }

      red   = ((color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0]) * 235.0f;
      green = ((color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1]) * 235.0f;
      blue  = ((color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2]) * 235.0f;
  }

  __device__ __forceinline__ void
  operator () () const
  {
    int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
    int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

    if (x >= dstColor.cols || y >= dstColor.rows)
      return;

    float3 v, n;
    v.x = vmap.ptr (y)[x];
    n.x = nmap.ptr (y)[x];
    uchar3 color = make_uchar3 (0, 0, 0);

    if (!isnan (v.x) && !isnan (n.x))
    {
      int r = (int)(vmap_curr_color.ptr (y)[x].x);
      r = max (0, min (255, r));
      int g = (int)(vmap_curr_color.ptr (y)[x].y);
      g = max (0, min (255, g));
      int b = (int)(vmap_curr_color.ptr (y)[x].z);
      b = max (0, min (255, b));
      color = make_uchar3 (r, g, b);
    }

    dstColor.ptr (y)[x] = color;

    if (!isnan (v.x) && !isnan (n.x))
    {
        v.y = vmap.ptr (y + dst.rows)[x];
        v.z = vmap.ptr (y + 2 * dst.rows)[x];

        n.y = nmap.ptr (y + dst.rows)[x];
        n.z = nmap.ptr (y + 2 * dst.rows)[x];

        float weight = 1.f;

        for (int i = 0; i < light.number; ++i)
        {
          float3 vec = normalized (light.pos[i] - v);

          weight *= fabs (dot (vec, n));
        }

        int r, g, b;

        getHeatMapColor((float)vmap_curr_color.ptr (y)[x].w / 128.0f, r, g, b);

        color = make_uchar3 (b * weight + 20,
                             g * weight + 20,
                             r * weight + 20);
    }

    dst.ptr (y)[x] = color;
  }
};

__global__ void
generateImageKernel (const ImageGenerator ig) {
  ig ();
}

void
generateImage (const DeviceArray2D<float>& vmap, const DeviceArray2D<float>& nmap, const DeviceArray2D<uchar4> & vmap_curr_color, const LightSource& light,
                           PtrStepSz<uchar3> dst, PtrStepSz<uchar3> dstColor)
{
  ImageGenerator ig;
  ig.vmap = vmap;
  ig.nmap = nmap;
  ig.light = light;
  ig.dst = dst;
  ig.dstColor = dstColor;
  ig.vmap_curr_color = vmap_curr_color;

  dim3 block (ImageGenerator::CTA_SIZE_X, ImageGenerator::CTA_SIZE_Y);
  dim3 grid (divUp (dst.cols, block.x), divUp (dst.rows, block.y));

  generateImageKernel<<<grid, block>>>(ig);
  cudaSafeCall (cudaGetLastError ());
  cudaSafeCall (cudaDeviceSynchronize ());
} 
__global__ void generateDepthKernel(const float3 R_inv_row3, const float3 t, const PtrStep<float> vmap, const PtrStep<float> nmap, PtrStepSz<unsigned short> depth, float maxDepth)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < depth.cols && y < depth.rows)
  {
    unsigned short result = 0;

    float3 v_g, n;
    v_g.x = vmap.ptr (y)[x];
    n.x = nmap.ptr (y)[x];

    if (!isnan (v_g.x) && !isnan (n.x))
    {
      v_g.y = vmap.ptr (y +     depth.rows)[x];
      v_g.z = vmap.ptr (y + 2 * depth.rows)[x];

      float v_z = dot(R_inv_row3, v_g - t);

      result = static_cast<unsigned short>(v_z * 1000);
    }
    depth.ptr(y)[x] = result;
  }
}

void
generateDepth (const Mat33& R_inv, const float3& t, const DeviceArray2D<float>& vmap, const DeviceArray2D<float>& nmap, DeviceArray2D<unsigned short>& dst, float maxDepth)
{
  dim3 block(32, 8);
  dim3 grid(divUp(dst.cols(), block.x), divUp(dst.rows(), block.y));
  
  generateDepthKernel<<<grid, block>>>(R_inv.data[2], t, vmap, nmap, dst, maxDepth);
  cudaSafeCall (cudaGetLastError ());
  cudaSafeCall (cudaDeviceSynchronize ());  
}
