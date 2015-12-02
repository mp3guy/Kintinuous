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
#include "warp.hpp"

//////////////////////////////////////////////////////////////////////// Full Volume Scan6

enum
{
  CTA_SIZE_X = 32,
  CTA_SIZE_Y = 6,
  CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,

  MAX_LOCAL_POINTS = 3
};

__device__ int global_count = 0;
__device__ int output_count;
__device__ unsigned int blocks_done = 0;

__shared__ float storage_X[CTA_SIZE * MAX_LOCAL_POINTS];
__shared__ float storage_Y[CTA_SIZE * MAX_LOCAL_POINTS];
__shared__ float storage_Z[CTA_SIZE * MAX_LOCAL_POINTS];
__shared__ float storage_W[CTA_SIZE * MAX_LOCAL_POINTS];

//////////////////////////////////////////////////////////////////////// Full Volume Scan6 Slice

struct FullScan6Slice
{
  PtrStep<short> volume;
  float3 cell_size;
  int subsample;
  int3 voxelWrap;
  int3 realVoxelWrap;

  int minX;
  int maxX;
  int minY;
  int maxY;
  int minZ;
  int maxZ;

  int xOffset;
  int yOffset;

  mutable PtrSz<PointXYZRGB> output;

  PtrStep<uchar4> color_volume;

  __device__ __forceinline__ float
  fetch (int x, int y, int z, int& weight) const
  {
      const short * pos = &volume.ptr(0)[((x + voxelWrap.x) % VOLUME_X) + ((y + voxelWrap.y) % VOLUME_Y) * VOLUME_X + ((z + voxelWrap.z) % VOLUME_Z) * VOLUME_X * VOLUME_Y];
      float tsdf = unpack_tsdf(*pos);
      const uchar4 * ptrColor = &color_volume.ptr(0)[((x + voxelWrap.x) % VOLUME_X) + ((y + voxelWrap.y) % VOLUME_Y) * VOLUME_X + ((z + voxelWrap.z) % VOLUME_Z) * VOLUME_X * VOLUME_Y];
      weight = ptrColor->w;
      return tsdf;
  }

  __device__ __forceinline__ void
  fetchColor (int x, int y, int z, int & r, int & g, int & b) const
  {
      const uchar4 * ptrColor = &color_volume.ptr(0)[((x + voxelWrap.x) % VOLUME_X) + ((y + voxelWrap.y) % VOLUME_Y) * VOLUME_X + ((z + voxelWrap.z) % VOLUME_Z) * VOLUME_X * VOLUME_Y];
      r = ptrColor->x;
      g = ptrColor->y;
      b = ptrColor->z;
  }

  __device__ __forceinline__ void
  operator () () const
  {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) + xOffset;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) + yOffset;

    if (__all (x >= VOLUME_X) || __all (y >= VOLUME_Y))
      return;

    float3 V;
    V.x = (x + 0.5f) * cell_size.x;
    V.y = (y + 0.5f) * cell_size.y;

    int ftid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    for (int z = minZ; z < maxZ; z += subsample)
    {
      float4 points[MAX_LOCAL_POINTS];
      int local_count = 0;

      if (x < VOLUME_X && y < VOLUME_Y &&
          x >= minX && x < maxX &&
          y >= minY && y < maxY &&
          x % subsample == 0 &&
          y % subsample == 0)
      {
        int W;
        float F = fetch (x, y, z, W);

        if (W != 0 && F != 1.f)
        {
          V.z = (z + 0.5f) * cell_size.z;

          //process dx
          if (x + 1 < VOLUME_X)
          {
            int Wn;
            float Fn = fetch (x + 1, y, z, Wn);

            if (Wn != 0 && Fn != 1.f)
              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
              {
                float4 p;
                p.y = V.y;
                p.z = V.z;

                float Vnx = V.x + cell_size.x;

                float d_inv = 1.f / (fabs (F) + fabs (Fn));
                p.x = (V.x * fabs (Fn) + Vnx * fabs (F)) * d_inv;

                int3 colorsPacked;
                fetchColor(x + 1, y, z, colorsPacked.x, colorsPacked.y, colorsPacked.z);

                int rgb = ((int)colorsPacked.x << 24 | (int)colorsPacked.y << 16 | (int)colorsPacked.z << 8 | (int)W);

                p.w = *reinterpret_cast<float*>(&rgb);

                points[local_count++] = p;
              }
          }               /* if (x + 1 < VOLUME_X) */

          //process dy
          if (y + 1 < VOLUME_Y)
          {
            int Wn;
            float Fn = fetch (x, y + 1, z, Wn);

            if (Wn != 0 && Fn != 1.f)
              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
              {
                float4 p;
                p.x = V.x;
                p.z = V.z;

                float Vny = V.y + cell_size.y;

                float d_inv = 1.f / (fabs (F) + fabs (Fn));
                p.y = (V.y * fabs (Fn) + Vny * fabs (F)) * d_inv;

                int3 colorsPacked;
                fetchColor(x, y + 1, z, colorsPacked.x, colorsPacked.y, colorsPacked.z);

                int rgb = ((int)colorsPacked.x << 24 | (int)colorsPacked.y << 16 | (int)colorsPacked.z << 8 | (int)W);

                p.w = *reinterpret_cast<float*>(&rgb);

                points[local_count++] = p;
              }
          }                /*  if (y + 1 < VOLUME_Y) */

          //process dz
          //if (z + 1 < VOLUME_Z) // guaranteed by loop
          {
            int Wn;
            float Fn = fetch (x, y, z + 1, Wn);

            if (Wn != 0 && Fn != 1.f)
              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
              {
                float4 p;
                p.x = V.x;
                p.y = V.y;

                float Vnz = V.z + cell_size.z;

                float d_inv = 1.f / (fabs (F) + fabs (Fn));
                p.z = (V.z * fabs (Fn) + Vnz * fabs (F)) * d_inv;

                int3 colorsPacked;
                fetchColor(x, y, z + 1, colorsPacked.x, colorsPacked.y, colorsPacked.z);

                int rgb = ((int)colorsPacked.x << 24 | (int)colorsPacked.y << 16 | (int)colorsPacked.z << 8 | (int)W);

                p.w = *reinterpret_cast<float*>(&rgb);

                points[local_count++] = p;
              }
          }               /* if (z + 1 < VOLUME_Z) */
        }              /* if (W != 0 && F != 1.f) */
      }            /* if (x < VOLUME_X && y < VOLUME_Y) */      ///not we fulfilled points array at current iteration
      int total_warp = __popc (__ballot (local_count > 0)) + __popc (__ballot (local_count > 1)) + __popc (__ballot (local_count > 2));

      if (total_warp > 0)
      {
        int lane = Warp::laneId ();
        int storage_index = (ftid >> Warp::LOG_WARP_SIZE) * Warp::WARP_SIZE * MAX_LOCAL_POINTS;

        volatile int* cta_buffer = (int*)(storage_X + storage_index);

        cta_buffer[lane] = local_count;
        int offset = scan_warp<exclusive>(cta_buffer, lane);

        if (lane == 0)
        {
          int old_global_count = atomicAdd (&global_count, total_warp);
          cta_buffer[0] = old_global_count;
        }
        int old_global_count = cta_buffer[0];

        for (int l = 0; l < local_count; ++l)
        {
          storage_X[storage_index + offset + l] = points[l].x;
          storage_Y[storage_index + offset + l] = points[l].y;
          storage_Z[storage_index + offset + l] = points[l].z;
          storage_W[storage_index + offset + l] = points[l].w;
        }

        PointXYZRGB *pos = output.data + old_global_count + lane;
        for (int idx = lane; idx < total_warp; idx += Warp::STRIDE, pos += Warp::STRIDE)
        {
          float x = storage_X[storage_index + idx];
          float y = storage_Y[storage_index + idx];
          float z = storage_Z[storage_index + idx];
          float w = storage_W[storage_index + idx];

          int rgb = *reinterpret_cast<int*>(&w);
          int r = (rgb >> 24) & 0x0000ff;
          int g = (rgb >> 16) & 0x0000ff;
          int b = (rgb >> 8)  & 0x0000ff;
          int weight = (rgb)  & 0x0000ff;

          store_point_type (x, y, z, r, g, b, weight, pos);
        }

        bool full = (old_global_count + total_warp) >= output.size;

        if (full)
          break;
      }

    }         /* for(int z = 0; z < VOLUME_Z - 1; ++z) */    //////    // prepare for future scans
    if (ftid == 0)
    {
      unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
      unsigned int value = atomicInc (&blocks_done, total_blocks);

      //last block
      if (value == total_blocks - 1)
      {
        output_count = min ((int)output.size, global_count);
        blocks_done = 0;
        global_count = 0;
      }
    }
  }       /* operator() */

  __device__ __forceinline__ void
  store_point_type (float x, float y, float z, int r, int g, int b, int w, PointXYZRGB * ptr) const
  {
      ptr->x = x + realVoxelWrap.x * cell_size.x - ((cell_size.x * VOLUME_X) / 2);
      ptr->y = y + realVoxelWrap.y * cell_size.y - ((cell_size.y * VOLUME_Y) / 2);
      ptr->z = z + realVoxelWrap.z * cell_size.z - ((cell_size.z * VOLUME_Z) / 2);
      ptr->r = b;
      ptr->g = g;
      ptr->b = r;
      ptr->a = w;
  }
};

__global__ void
extractKernelSlice (const FullScan6Slice fs) {
  fs ();
}

size_t extractCloudSlice(const PtrStep<short>& volume,
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
                                      int3 realVoxelWrap)
{
  FullScan6Slice fs;
  fs.volume = volume;
  fs.cell_size.x = (volume_size.x / VOLUME_X);
  fs.cell_size.y = (volume_size.y / VOLUME_Y);
  fs.cell_size.z = (volume_size.z / VOLUME_Z);
  fs.output = output;
  fs.subsample = subsample;
  fs.voxelWrap = voxelWrap;
  fs.realVoxelWrap = realVoxelWrap;

  fs.minX = minX;
  fs.maxX = maxX;
  fs.minY = minY;
  fs.maxY = maxY;
  fs.minZ = minZ;
  fs.maxZ = maxZ;

  fs.xOffset = 0;
  fs.yOffset = 0;

  fs.color_volume = color_volume;

  int amountX = maxX - minX;
  int amountY = maxY - minY;
  int amountZ = maxZ - minZ;

  dim3 block (CTA_SIZE_X, CTA_SIZE_Y);

  bool fullCloud = amountX == VOLUME_X &&
			       amountY == VOLUME_Y &&
			       amountZ == VOLUME_Z;

  if(fullCloud)
  {
	  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));
	  extractKernelSlice<<<grid, block>>>(fs);
  }
  else
  {
	  int amount = amountZ;

	  bool x = false, y = false;

	  if(amountX < amountY &&
		 amountX < amountZ)
	  {
		  fs.xOffset = minX;
		  amount = amountX;
		  x = true;
	  }
	  else if(amountY < amountX &&
              amountY < amountZ)
	  {
		  fs.yOffset = minY;
		  amount = amountY;
		  y = true;
	  }

	  int remainder = amount % 16;

	  if(remainder != 0)
	  {
		  remainder = amount + 16 - remainder;
	  }
	  else
	  {
		  remainder = amount;
	  }

	  dim3 grid (x ? divUp(remainder, block.x) : divUp(VOLUME_X, block.x), y ? divUp(remainder, block.y) : divUp(VOLUME_Y, block.y));
	  extractKernelSlice<<<grid, block>>>(fs);
  }

  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());

  int size;
  cudaSafeCall(cudaMemcpyFromSymbol(&size, output_count, sizeof(size)));

  return (size_t)(size);
}

