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

const float sigma_color = 30;     //in mm
const float sigma_space = 4.5;     // in pixels

__global__ void
bilateralKernel (const PtrStepSz<unsigned short> src,
                 PtrStep<unsigned short> dst,
                 float sigma_space2_inv_half, float sigma_color2_inv_half)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= src.cols || y >= src.rows)
    return;

  const int R = 6;       //static_cast<int>(sigma_space * 1.5);
  const int D = R * 2 + 1;

  int value = src.ptr (y)[x];

  int tx = min (x - D / 2 + D, src.cols - 1);
  int ty = min (y - D / 2 + D, src.rows - 1);

  float sum1 = 0;
  float sum2 = 0;

  for (int cy = max (y - D / 2, 0); cy < ty; ++cy)
  {
    for (int cx = max (x - D / 2, 0); cx < tx; ++cx)
    {
      int tmp = src.ptr (cy)[cx];

      float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      float color2 = (value - tmp) * (value - tmp);

      float weight = __expf (-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

      sum1 += tmp * weight;
      sum2 += weight;
    }
  }

  int res = __float2int_rn (sum1 / sum2);
  dst.ptr (y)[x] = max (0, min (res, numeric_limits<short>::max ()));
}

__global__ void
pyrDownGaussKernel (const PtrStepSz<unsigned short> src, PtrStepSz<unsigned short> dst, float sigma_color)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows)
    return;

  const int D = 5;

  int center = src.ptr (2 * y)[2 * x];

  int x_mi = max(0, 2*x - D/2) - 2*x;
  int y_mi = max(0, 2*y - D/2) - 2*y;

  int x_ma = min(src.cols, 2*x -D/2+D) - 2*x;
  int y_ma = min(src.rows, 2*y -D/2+D) - 2*y;

  float sum = 0;
  float wall = 0;

  float weights[] = {0.375f, 0.25f, 0.0625f} ;

  for(int yi = y_mi; yi < y_ma; ++yi)
      for(int xi = x_mi; xi < x_ma; ++xi)
      {
          int val = src.ptr (2*y + yi)[2*x + xi];

          if (abs (val - center) < 3 * sigma_color)
          {
            sum += val * weights[abs(xi)] * weights[abs(yi)];
            wall += weights[abs(xi)] * weights[abs(yi)];
          }
      }  dst.ptr (y)[x] = static_cast<int>(sum /wall);
}

__global__ void
pyrDownKernel (const PtrStepSz<unsigned short> src, PtrStepSz<unsigned short> dst, float sigma_color)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows)
    return;

  const int D = 5;

  int center = src.ptr (2 * y)[2 * x];

  int tx = min (2 * x - D / 2 + D, src.cols - 1);
  int ty = min (2 * y - D / 2 + D, src.rows - 1);
  int cy = max (0, 2 * y - D / 2);

  int sum = 0;
  int count = 0;

  for (; cy < ty; ++cy)
    for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
    {
      int val = src.ptr (cy)[cx];
      if (abs (val - center) < 3 * sigma_color)
      {
        sum += val;
        ++count;
      }
    }
  dst.ptr (y)[x] = sum / count;
}

__global__ void
pyrDownKernelIntensityGauss(const PtrStepSz<unsigned char> src, PtrStepSz<unsigned char> dst, float * gaussKernel)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows)
    return;

  const int D = 5;

  int center = src.ptr (2 * y)[2 * x];

  int tx = min (2 * x - D / 2 + D, src.cols - 1);
  int ty = min (2 * y - D / 2 + D, src.rows - 1);
  int cy = max (0, 2 * y - D / 2);

  float sum = 0;
  int count = 0;

  for (; cy < ty; ++cy)
    for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
    {
        sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
        count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
    }
  dst.ptr (y)[x] = (sum / (float)count);
}

__global__ void
pyrDownKernelGaussF(const PtrStepSz<float> src, PtrStepSz<float> dst, float * gaussKernel)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows)
    return;

  const int D = 5;

  float center = src.ptr (2 * y)[2 * x];

  int tx = min (2 * x - D / 2 + D, src.cols - 1);
  int ty = min (2 * y - D / 2 + D, src.rows - 1);
  int cy = max (0, 2 * y - D / 2);

  float sum = 0;
  int count = 0;

  for (; cy < ty; ++cy)
  {
      for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
      {
          if(!isnan(src.ptr (cy)[cx]))
          {
              sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
              count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
          }
      }
  }
  dst.ptr (y)[x] = (float)(sum / (float)count);
}

__global__ void
short2FloatKernel(const PtrStepSz<unsigned short> src, PtrStepSz<float> dst, int cutOff)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows)
    return;

  dst.ptr(y)[x] = src.ptr(y)[x] > cutOff  || src.ptr(y)[x] <= 0 ? numeric_limits<float>::quiet_NaN() : ((float)src.ptr(y)[x]) / 1000.0f;
}

__global__ void
bgr2IntensityKernel(const PtrStepSz<PixelRGB> src, PtrStepSz<unsigned char> dst)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows)
    return;

  int value = (float)src.ptr(y)[x].r * 0.114f + (float)src.ptr(y)[x].b * 0.299f + (float)src.ptr(y)[x].g * 0.587f;

  dst.ptr (y)[x] = value;
}

__global__ void
truncateDepthKernel(PtrStepSz<unsigned short> depth, unsigned short max_distance_mm)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < depth.cols && y < depth.rows)
        if(depth.ptr(y)[x] > max_distance_mm)
            depth.ptr(y)[x] = 0;
}

__constant__ float gsobel_x3x3[9];
__constant__ float gsobel_y3x3[9];

__global__ void applyKernel(const PtrStepSz<unsigned char> src, PtrStep<short> dx, PtrStep<short> dy)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x >= src.cols || y >= src.rows)
    return;

  float dxVal = 0;
  float dyVal = 0;

  int kernelIndex = 8;
  for(int j = max(y - 1, 0); j <= min(y + 1, src.rows - 1); j++)
  {
      for(int i = max(x - 1, 0); i <= min(x + 1, src.cols - 1); i++)
      {
          dxVal += (float)src.ptr(j)[i] * gsobel_x3x3[kernelIndex];
          dyVal += (float)src.ptr(j)[i] * gsobel_y3x3[kernelIndex];
          --kernelIndex;
      }
  }

  dx.ptr(y)[x] = dxVal;
  dy.ptr(y)[x] = dyVal;
}

void computeDerivativeImages(DeviceArray2D<unsigned char>& src, DeviceArray2D<short>& dx, DeviceArray2D<short>& dy)
{
    static bool once = false;

    if(!once)
    {
        float gsx3x3[9] = {0.52201,  0.00000, -0.52201,
                           0.79451, -0.00000, -0.79451,
                           0.52201,  0.00000, -0.52201};

        float gsy3x3[9] = {0.52201, 0.79451, 0.52201,
                           0.00000, 0.00000, 0.00000,
                           -0.52201, -0.79451, -0.52201};

        cudaMemcpyToSymbol(gsobel_x3x3, gsx3x3, sizeof(float) * 9);
        cudaMemcpyToSymbol(gsobel_y3x3, gsy3x3, sizeof(float) * 9);

        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());

        once = true;
    }

    dim3 block(32, 8);
    dim3 grid(divUp (src.cols (), block.x), divUp (src.rows (), block.y));

    applyKernel<<<grid, block>>>(src, dx, dy);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void
bilateralFilter (const DeviceArray2D<unsigned short>& src, DeviceArray2D<unsigned short>& dst)
{
  dim3 block (32, 8);
  dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

  cudaFuncSetCacheConfig (bilateralKernel, cudaFuncCachePreferL1);
  bilateralKernel<<<grid, block>>>(src, dst, 0.5f / (sigma_space * sigma_space), 0.5f / (sigma_color * sigma_color));

  cudaSafeCall ( cudaGetLastError () );
};

void
pyrDown (const DeviceArray2D<unsigned short>& src, DeviceArray2D<unsigned short>& dst)
{
  dst.create (src.rows () / 2, src.cols () / 2);

  dim3 block (32, 8);
  dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

  pyrDownGaussKernel<<<grid, block>>>(src, dst, sigma_color);
  cudaSafeCall ( cudaGetLastError () );
};

void pyrDownGaussF(const DeviceArray2D<float>& src, DeviceArray2D<float> & dst)
{
  dst.create (src.rows () / 2, src.cols () / 2);

  dim3 block (32, 8);
  dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

  const float gaussKernel[25] = {1, 4, 6, 4, 1,
  								 4, 16, 24, 16, 4,
  								 6, 24, 36, 24, 6,
  								 4, 16, 24, 16, 4,
  								 1, 4, 6, 4, 1};

  float * gauss_cuda;

  cudaMalloc((void**) &gauss_cuda, sizeof(float) * 25);
  cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);

  pyrDownKernelGaussF<<<grid, block>>>(src, dst, gauss_cuda);
  cudaSafeCall ( cudaGetLastError () );

  cudaFree(gauss_cuda);
};

void pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src, DeviceArray2D<unsigned char> & dst)
{
  dst.create (src.rows () / 2, src.cols () / 2);

  dim3 block (32, 8);
  dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

  const float gaussKernel[25] = {1, 4, 6, 4, 1,
  								 4, 16, 24, 16, 4,
  								 6, 24, 36, 24, 6,
  								 4, 16, 24, 16, 4,
  								 1, 4, 6, 4, 1};

  float * gauss_cuda;

  cudaMalloc((void**) &gauss_cuda, sizeof(float) * 25);
  cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);

  pyrDownKernelIntensityGauss<<<grid, block>>>(src, dst, gauss_cuda);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
  cudaFree(gauss_cuda);
};

void shortDepthToMetres(const DeviceArray2D<unsigned short>& src, DeviceArray2D<float> & dst, int cutOff)
{
  dim3 block (32, 8);
  dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

  short2FloatKernel<<<grid, block>>>(src, dst, cutOff);
  cudaSafeCall ( cudaGetLastError () );
};

void imageBGRToIntensity(const DeviceArray2D<PixelRGB> & src, DeviceArray2D<unsigned char> & dst)
{
  dim3 block (32, 8);
  dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

  bgr2IntensityKernel<<<grid, block>>>(src, dst);
  cudaSafeCall ( cudaGetLastError () );
};
