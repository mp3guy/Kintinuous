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
#include "vector_math.hpp"

__global__ void
initColorVolumeKernel (PtrStep<uchar4> volume)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < VOLUME_X && y < VOLUME_Y)
  {
    uchar4 *pos = volume.ptr (y) + x;
    int z_step = VOLUME_Y * volume.step / sizeof(*pos);

#pragma unroll
    for(int z = 0; z < VOLUME_Z; ++z, pos += z_step)
    {
        clear_voxel(*pos);
    }
  }
}

void
initColorVolume (PtrStep<uchar4> color_volume)
{
  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = divUp (VOLUME_X, block.x);
  grid.y = divUp (VOLUME_Y, block.y);

  initColorVolumeKernel<<<grid, block>>>(color_volume);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}template<typename T>
__global__ void
clearVolumeInX (PtrStep<T> volume, int bottom, int numUp)
{
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    bottom %= VOLUME_X;
    const int cachedWrap = (bottom + numUp) % VOLUME_X;

    const bool wrap = cachedWrap != bottom + numUp;

    int x = ((threadIdx.x + blockIdx.x * blockDim.x) + bottom) % VOLUME_X;

    if(!wrap ? (x >= bottom && x <= cachedWrap) :
               (x >= bottom || x <= cachedWrap))
    {
        T * base = volume.ptr(0);
        T * pos;

        const int cachedXY = x + y * VOLUME_X;
        const int cachedProduct = VOLUME_X * VOLUME_Y;

        for(int z = 0; z < VOLUME_Z; ++z)
        {
            pos = &base[cachedXY + z * cachedProduct];
            clear_voxel(*pos);
        }
    }
}

void clearVolumeX (PtrStep<short> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
	int remainder = (deltaVoxelWrap - currentVoxelWrap) % 16;

	if(remainder != 0)
	{
		remainder = (deltaVoxelWrap - currentVoxelWrap) + 16 - remainder;
	}
	else
	{
		remainder = abs(deltaVoxelWrap - currentVoxelWrap);
	}

    dim3 block (16, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (remainder, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    int bottom = currentVoxelWrap > 0 ? currentVoxelWrap % VOLUME_X : VOLUME_X - ((-currentVoxelWrap) % VOLUME_X);
    int numUp = -(currentVoxelWrap - deltaVoxelWrap);

    clearVolumeInX<<<grid, block>>>(volume, bottom, numUp);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

void
clearVolumeXc (PtrStep<uchar4> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
	int remainder = (deltaVoxelWrap - currentVoxelWrap) % 16;

	if(remainder != 0)
	{
		remainder = (deltaVoxelWrap - currentVoxelWrap) + 16 - remainder;
	}
	else
	{
		remainder = abs(deltaVoxelWrap - currentVoxelWrap);
	}

    dim3 block (16, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (remainder, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    int bottom = currentVoxelWrap > 0 ? currentVoxelWrap % VOLUME_X : VOLUME_X - ((-currentVoxelWrap) % VOLUME_X);
    int numUp = -(currentVoxelWrap - deltaVoxelWrap);

    clearVolumeInX<<<grid, block>>>(volume, bottom, numUp);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

void
clearVolumeXBack (PtrStep<short> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
	int remainder = (deltaVoxelWrap - currentVoxelWrap) % 16;

	if(remainder != 0)
	{
		remainder = (deltaVoxelWrap - currentVoxelWrap) + 16 - remainder;
	}
	else
	{
		remainder = abs(deltaVoxelWrap - currentVoxelWrap);
	}

    dim3 block (16, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (remainder, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    int currentVoxelBase = currentVoxelWrap > 0 ? currentVoxelWrap % VOLUME_X : VOLUME_X - ((-currentVoxelWrap) % VOLUME_X);
    int top = (currentVoxelBase + VOLUME_X) % VOLUME_X;
    int numDown = currentVoxelWrap - deltaVoxelWrap;

    int bottom = top - numDown;

    if(bottom < 0)
    {
    	bottom = VOLUME_X + bottom;
    }

    clearVolumeInX<<<grid, block>>>(volume, bottom, numDown);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

void
clearVolumeXBackc (PtrStep<uchar4> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
	int remainder = (deltaVoxelWrap - currentVoxelWrap) % 16;

	if(remainder != 0)
	{
		remainder = (deltaVoxelWrap - currentVoxelWrap) + 16 - remainder;
	}
	else
	{
		remainder = abs(deltaVoxelWrap - currentVoxelWrap);
	}

    dim3 block (16, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (remainder, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    int currentVoxelBase = currentVoxelWrap > 0 ? currentVoxelWrap % VOLUME_X : VOLUME_X - ((-currentVoxelWrap) % VOLUME_X);
    int top = (currentVoxelBase + VOLUME_X) % VOLUME_X;
    int numDown = currentVoxelWrap - deltaVoxelWrap;

    int bottom = top - numDown;

    if(bottom < 0)
    {
    	bottom = VOLUME_X + bottom;
    }

    clearVolumeInX<<<grid, block>>>(volume, bottom, numDown);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

template<typename T>
__global__ void
clearVolumeInY (PtrStep<T> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < VOLUME_X && y < VOLUME_Y)
  {
      int bottom = currentVoxelWrap > 0 ? currentVoxelWrap % VOLUME_Y : VOLUME_Y - ((-currentVoxelWrap) % VOLUME_Y);
      int numUp = -(currentVoxelWrap - deltaVoxelWrap);

      T * base = volume.ptr(0);
      T * pos;

      while(numUp >= 0)
      {
          pos = &base[x + (bottom++ % VOLUME_Y) * VOLUME_X + y * VOLUME_X * VOLUME_Y];
          clear_voxel(*pos);
          numUp--;
      }
  }
}

void
clearVolumeY (PtrStep<short> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
    dim3 block (32, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (VOLUME_X, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    clearVolumeInY<<<grid, block>>>(volume, currentVoxelWrap, deltaVoxelWrap);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

void
clearVolumeYc (PtrStep<uchar4> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
    dim3 block (32, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (VOLUME_X, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    clearVolumeInY<<<grid, block>>>(volume, currentVoxelWrap, deltaVoxelWrap);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

template<typename T>
__global__ void
clearVolumeInYBack (PtrStep<T> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < VOLUME_X && y < VOLUME_Y)
  {
      int currentVoxelBase = currentVoxelWrap > 0 ? currentVoxelWrap % VOLUME_Y : VOLUME_Y - ((-currentVoxelWrap) % VOLUME_Y);
      int top = (currentVoxelBase + VOLUME_Y) % VOLUME_Y;
      int numDown = currentVoxelWrap - deltaVoxelWrap;

      T * base = volume.ptr(0);
      T * pos;

      while(numDown >= 0)
      {
          pos = &base[x + (top-- % VOLUME_Y) * VOLUME_X + y * VOLUME_X * VOLUME_Y];
          clear_voxel(*pos);

          if(top < 0)
              top = VOLUME_Y - 1;

          numDown--;
      }
  }
}

void
clearVolumeYBack (PtrStep<short> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
    dim3 block (32, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (VOLUME_X, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    clearVolumeInYBack<<<grid, block>>>(volume, currentVoxelWrap, deltaVoxelWrap);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

void
clearVolumeYBackc (PtrStep<uchar4> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
    dim3 block (32, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (VOLUME_X, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    clearVolumeInYBack<<<grid, block>>>(volume, currentVoxelWrap, deltaVoxelWrap);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

template<typename T>
__global__ void
clearVolumeInZ (PtrStep<T> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < VOLUME_X && y < VOLUME_Y)
  {
      int bottom = currentVoxelWrap > 0 ? currentVoxelWrap % VOLUME_Z : VOLUME_Z - ((-currentVoxelWrap) % VOLUME_Z);
      int numUp = -(currentVoxelWrap - deltaVoxelWrap);

      T * base = volume.ptr(0);
      T * pos;

      while(numUp >= 0)
      {
          pos = &base[x + y * VOLUME_X + (bottom++ % VOLUME_Z) * VOLUME_X * VOLUME_Y];
          clear_voxel(*pos);
          numUp--;
      }
  }
}

void
clearVolumeZ (PtrStep<short> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
    dim3 block (32, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (VOLUME_X, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    clearVolumeInZ<<<grid, block>>>(volume, currentVoxelWrap, deltaVoxelWrap);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

void
clearVolumeZc (PtrStep<uchar4> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
    dim3 block (32, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (VOLUME_X, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    clearVolumeInZ<<<grid, block>>>(volume, currentVoxelWrap, deltaVoxelWrap);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

template<typename T>
__global__ void
clearVolumeInZBack (PtrStep<T> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < VOLUME_X && y < VOLUME_Y)
  {
      int currentVoxelBase = currentVoxelWrap > 0 ? currentVoxelWrap % VOLUME_Z : VOLUME_Z - ((-currentVoxelWrap) % VOLUME_Z);
      int top = (currentVoxelBase + VOLUME_Z) % VOLUME_Z;
      int numDown = currentVoxelWrap - deltaVoxelWrap;

      T * base = volume.ptr(0);
      T * pos;

      while(numDown >= 0)
      {
          pos = &base[x + y * VOLUME_X + (top-- % VOLUME_Z) * VOLUME_X * VOLUME_Y];
          clear_voxel(*pos);

          if(top < 0)
              top = VOLUME_Z - 1;

          numDown--;
      }
  }
}

void
clearVolumeZBack (PtrStep<short> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
    dim3 block (32, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (VOLUME_X, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    clearVolumeInZBack<<<grid, block>>>(volume, currentVoxelWrap, deltaVoxelWrap);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

void
clearVolumeZBackc (PtrStep<uchar4> volume, const int currentVoxelWrap, const int deltaVoxelWrap)
{
    dim3 block (32, 16);
    dim3 grid (1, 1, 1);
    grid.x = divUp (VOLUME_X, block.x);
    grid.y = divUp (VOLUME_Y, block.y);

    clearVolumeInZBack<<<grid, block>>>(volume, currentVoxelWrap, deltaVoxelWrap);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

template<typename T>
__global__ void
initializeVolume (PtrStep<T> volume)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;  if (x < VOLUME_X && y < VOLUME_Y)
  {
      T *pos = volume.ptr(y) + x;
      int z_step = VOLUME_Y * volume.step / sizeof(*pos);

#pragma unroll
      for(int z = 0; z < VOLUME_Z; ++z, pos+=z_step)
      {
          clear_voxel(*pos);
      }
  }
}

void
initVolume (PtrStep<short> volume)
{
  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = divUp (VOLUME_X, block.x);      
  grid.y = divUp (VOLUME_Y, block.y);

  initializeVolume<<<grid, block>>>(volume);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

struct Tsdf
{
  enum
  {
    CTA_SIZE_X = 32, CTA_SIZE_Y = 8,
    MAX_WEIGHT = 1 << 7
  };
};

__global__ void
scaleDepth (const PtrStepSz<unsigned short> depth, PtrStep<float> scaled, const Intr intr, bool angleColor)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= depth.cols || y >= depth.rows)
    return;

  int Dp = depth.ptr (y)[x];

  float xl = (x - intr.cx) / intr.fx;
  float yl = (y - intr.cy) / intr.fy;
  float lambda = sqrtf (xl * xl + yl * yl + 1);

  if(angleColor)
  {
      int STEP = 1;
      int ky = 7;
      int kx = 7;
      int ty = min (y - ky / 2 + ky, depth.rows - 1);
      int tx = min (x - kx / 2 + kx, depth.cols - 1);
      int count = 0;

      for (int cy = max (y - ky / 2, 0); cy < ty; cy += STEP)
      {
        for (int cx = max (x - kx / 2, 0); cx < tx; cx += STEP)
        {
            if (abs(Dp-depth.ptr (cy)[cx]) > 200 || Dp == 0)
            {
                count++;
            }
        }
      }

      if(count > 5)
      {
          scaled.ptr (y)[x] = -Dp * lambda/1000.f; //meters
      }
      else
      {
          scaled.ptr (y)[x] = Dp * lambda/1000.f; //meters
      }
  }
  else
  {
      scaled.ptr (y)[x] = Dp * lambda/1000.f; //meters
  }
}

__global__ void
tsdf23 (const PtrStepSz<float> depthScaled, PtrStep<short> volume,
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size,
        const int3 voxelWrap, PtrStep<uchar4> color_volume, PtrStepSz<uchar3> colors,
        PtrStep<float> nmap_curr, int rows, bool angleColor)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= VOLUME_X || y >= VOLUME_Y)
    return;

  float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
  float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
  float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

  float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

  float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
  float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
  float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

  float z_scaled = 0;

  float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
  float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

  float tranc_dist_inv = 1.0f / tranc_dist;

  for (int z = 0; z < VOLUME_Z;
       ++z,
       v_g_z += cell_size.z,
       z_scaled += cell_size.z,
       v_x += Rcurr_inv_0_z_scaled,
       v_y += Rcurr_inv_1_z_scaled)
  {
    float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
    if (inv_z < 0)
        continue;

    // project to current cam
    int2 coo =
    {
      __float2int_rn (v_x * inv_z + intr.cx),
      __float2int_rn (v_y * inv_z + intr.cy)
    };

    if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
    {
      float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

      bool no_color = false;

      if(Dp_scaled < 0.0)
      {
          Dp_scaled = -Dp_scaled;
          no_color = true;
      }

      float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

      if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
      {
        float3 ncurr;
        ncurr.x = nmap_curr.ptr (coo.y)[coo.x];
        ncurr.y = nmap_curr.ptr (coo.y + rows)[coo.x];
        ncurr.z = nmap_curr.ptr (coo.y + 2 * rows)[coo.x];
        if (ncurr.z < 0) ncurr.z = -ncurr.z;

        float tsdf = fmin (1.0f, sdf * tranc_dist_inv);

        //read and unpack
        short * pos = &volume.ptr(0)[((x + voxelWrap.x) % VOLUME_X) + ((y + voxelWrap.y) % VOLUME_Y) * VOLUME_X + ((z + voxelWrap.z) % VOLUME_Z) * VOLUME_X * VOLUME_Y];
        float tsdf_prev = unpack_tsdf(*pos);

        uchar4 * ptrColor = &color_volume.ptr(0)[((x + voxelWrap.x) % VOLUME_X) + ((y + voxelWrap.y) % VOLUME_Y) * VOLUME_X + ((z + voxelWrap.z) % VOLUME_Z) * VOLUME_X * VOLUME_Y];
        float weight_prev = ptrColor->w;

        const float Wrk = 1; //Try weight this?

        pack_tsdf((tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk), *pos);
        ptrColor->w = min(weight_prev + Wrk, (float)Tsdf::MAX_WEIGHT);

        if ((!isnan(ncurr.x) && !no_color) || (ptrColor->x == 0 && ptrColor->y == 0 && ptrColor->z == 0))
        {
            const float Wrkc = (angleColor ? min(1.0f, ncurr.z / RGB_VIEW_ANGLE_WEIGHT) : 1.0f) * 2.0f;

            uchar3 rgb = colors.ptr (coo.y)[coo.x];

            float new_x = (ptrColor->x * weight_prev + Wrkc * rgb.x) / (weight_prev + Wrkc);
            float new_y = (ptrColor->y * weight_prev + Wrkc * rgb.y) / (weight_prev + Wrkc);
            float new_z = (ptrColor->z * weight_prev + Wrkc * rgb.z) / (weight_prev + Wrkc);

            ptrColor->x = min (255, max (0, __float2int_rn (new_x)));
            ptrColor->y = min (255, max (0, __float2int_rn (new_y)));
            ptrColor->z = min (255, max (0, __float2int_rn (new_z)));
        }
      }
    }
  }
}

void
integrateTsdfVolume (const PtrStepSz<unsigned short>& depth, const Intr& intr,
                                  const float3& volume_size, const Mat33& Rcurr_inv, const float3& tcurr, 
                                  float tranc_dist,
                                  PtrStep<short> volume, DeviceArray2D<float>& depthScaled,
                                  const int3 & voxelWrap, PtrStep<uchar4> color_volume, PtrStepSz<uchar3> colors,
                                  const DeviceArray2D<float>& nmap_curr,
                                  bool angleColor)
{
  depthScaled.create (depth.rows, depth.cols);

  dim3 block_scale (32, 8);
  dim3 grid_scale (divUp (depth.cols, block_scale.x), divUp (depth.rows, block_scale.y));

  //scales depth along ray and converts mm -> meters.
  scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr, angleColor);
  cudaSafeCall ( cudaGetLastError () );

  float3 cell_size;
  cell_size.x = volume_size.x / VOLUME_X;
  cell_size.y = volume_size.y / VOLUME_Y;
  cell_size.z = volume_size.z / VOLUME_Z;

  dim3 block (16, 16);
  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

  int rows = nmap_curr.rows () / 3;

  tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size, voxelWrap, color_volume, colors, nmap_curr, rows, angleColor);

  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}
