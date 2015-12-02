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

__global__ void
computeVmapKernel (const PtrStepSz<unsigned short> depth, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx, float cy)
{
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  int v = threadIdx.y + blockIdx.y * blockDim.y;

  if (u < depth.cols && v < depth.rows)
  {
    float z = depth.ptr (v)[u] / 1000.f; // load and convert: mm -> meters

    if (z != 0)
    {
      float vx = z * (u - cx) * fx_inv;
      float vy = z * (v - cy) * fy_inv;
      float vz = z;

      vmap.ptr (v                 )[u] = vx;
      vmap.ptr (v + depth.rows    )[u] = vy;
      vmap.ptr (v + depth.rows * 2)[u] = vz;
    }
    else
      vmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();

  }
}

__global__ void
computeNmapKernel (int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
{
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  int v = threadIdx.y + blockIdx.y * blockDim.y;

  if (u >= cols || v >= rows)
    return;

  if (u == cols - 1 || v == rows - 1)
  {
    nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();
    return;
  }

  float3 v00, v01, v10;
  v00.x = vmap.ptr (v  )[u];
  v01.x = vmap.ptr (v  )[u + 1];
  v10.x = vmap.ptr (v + 1)[u];

  if (!isnan (v00.x) && !isnan (v01.x) && !isnan (v10.x))
  {
    v00.y = vmap.ptr (v + rows)[u];
    v01.y = vmap.ptr (v + rows)[u + 1];
    v10.y = vmap.ptr (v + 1 + rows)[u];

    v00.z = vmap.ptr (v + 2 * rows)[u];
    v01.z = vmap.ptr (v + 2 * rows)[u + 1];
    v10.z = vmap.ptr (v + 1 + 2 * rows)[u];

    float3 r = normalized (cross (v01 - v00, v10 - v00));

    nmap.ptr (v       )[u] = r.x;
    nmap.ptr (v + rows)[u] = r.y;
    nmap.ptr (v + 2 * rows)[u] = r.z;
  }
  else
    nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();
}

void
createVMap (const Intr& intr, const DeviceArray2D<unsigned short>& depth, DeviceArray2D<float>& vmap)
{
  vmap.create (depth.rows () * 3, depth.cols ());

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (depth.cols (), block.x);
  grid.y = divUp (depth.rows (), block.y);

  float fx = intr.fx, cx = intr.cx;
  float fy = intr.fy, cy = intr.cy;

  computeVmapKernel<<<grid, block>>>(depth, vmap, 1.f / fx, 1.f / fy, cx, cy);
  cudaSafeCall (cudaGetLastError ());
}

void
createNMap (const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap)
{
  nmap.create (vmap.rows (), vmap.cols ());

  int rows = vmap.rows () / 3;
  int cols = vmap.cols ();

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  computeNmapKernel<<<grid, block>>>(rows, cols, vmap, nmap);
  cudaSafeCall (cudaGetLastError ());
}

__global__ void
tranformMapsKernel (int rows, int cols, const PtrStep<float> vmap_src, const PtrStep<float> nmap_src,
                    const Mat33 Rmat, const float3 tvec, PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  const float qnan = numeric_limits<float>::quiet_NaN ();

  if (x < cols && y < rows)
  {
    //vetexes
    float3 vsrc, vdst = make_float3 (qnan, qnan, qnan);
    vsrc.x = vmap_src.ptr (y)[x];

    if (!isnan (vsrc.x))
    {
      vsrc.y = vmap_src.ptr (y + rows)[x];
      vsrc.z = vmap_src.ptr (y + 2 * rows)[x];

      vdst = Rmat * vsrc + tvec;

      vmap_dst.ptr (y + rows)[x] = vdst.y;
      vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
    }

    vmap_dst.ptr (y)[x] = vdst.x;

    //normals
    float3 nsrc, ndst = make_float3 (qnan, qnan, qnan);
    nsrc.x = nmap_src.ptr (y)[x];

    if (!isnan (nsrc.x))
    {
      nsrc.y = nmap_src.ptr (y + rows)[x];
      nsrc.z = nmap_src.ptr (y + 2 * rows)[x];

      ndst = Rmat * nsrc;

      nmap_dst.ptr (y + rows)[x] = ndst.y;
      nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
    }

    nmap_dst.ptr (y)[x] = ndst.x;
  }
}

void
tranformMaps (const DeviceArray2D<float>& vmap_src, const DeviceArray2D<float>& nmap_src,
                           const Mat33& Rmat, const float3& tvec,
                           DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst)
{
  int cols = vmap_src.cols ();
  int rows = vmap_src.rows () / 3;

  vmap_dst.create (rows * 3, cols);
  nmap_dst.create (rows * 3, cols);

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  tranformMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst);
  cudaSafeCall (cudaGetLastError ());

  cudaSafeCall (cudaDeviceSynchronize ());
}

template<bool normalize>
__global__ void
resizeMapKernel (int drows, int dcols, int srows, const PtrStep<float> input, PtrStep<float> output)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= dcols || y >= drows)
    return;

  const float qnan = numeric_limits<float>::quiet_NaN ();

  int xs = x * 2;
  int ys = y * 2;

  float x00 = input.ptr (ys + 0)[xs + 0];
  float x01 = input.ptr (ys + 0)[xs + 1];
  float x10 = input.ptr (ys + 1)[xs + 0];
  float x11 = input.ptr (ys + 1)[xs + 1];

  if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
  {
    output.ptr (y)[x] = qnan;
    return;
  }
  else
  {
    float3 n;

    n.x = (x00 + x01 + x10 + x11) / 4;

    float y00 = input.ptr (ys + srows + 0)[xs + 0];
    float y01 = input.ptr (ys + srows + 0)[xs + 1];
    float y10 = input.ptr (ys + srows + 1)[xs + 0];
    float y11 = input.ptr (ys + srows + 1)[xs + 1];

    n.y = (y00 + y01 + y10 + y11) / 4;

    float z00 = input.ptr (ys + 2 * srows + 0)[xs + 0];
    float z01 = input.ptr (ys + 2 * srows + 0)[xs + 1];
    float z10 = input.ptr (ys + 2 * srows + 1)[xs + 0];
    float z11 = input.ptr (ys + 2 * srows + 1)[xs + 1];

    n.z = (z00 + z01 + z10 + z11) / 4;

    if (normalize)
      n = normalized (n);

    output.ptr (y        )[x] = n.x;
    output.ptr (y + drows)[x] = n.y;
    output.ptr (y + 2 * drows)[x] = n.z;
  }
}

template<bool normalize>
void
resizeMap (const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
  int in_cols = input.cols ();
  int in_rows = input.rows () / 3;

  int out_cols = in_cols / 2;
  int out_rows = in_rows / 2;

  output.create (out_rows * 3, out_cols);

  dim3 block (32, 8);
  dim3 grid (divUp (out_cols, block.x), divUp (out_rows, block.y));
  resizeMapKernel<normalize><< < grid, block>>>(out_rows, out_cols, in_rows, input, output);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

void
resizeVMap (const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
  resizeMap<false>(input, output);
}

void
resizeNMap (const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
  resizeMap<true>(input, output);
}

__global__ void
projectPointsKernel(const PtrStepSz<float> depth,
                  PtrStepSz<float3> cloud,
                  const double invFx,
                  const double invFy,
                  const double cx,
                  const double cy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= depth.cols || y >= depth.rows)
      return;

    float z = depth.ptr(y)[x];

    cloud.ptr(y)[x].x = (float)((x - cx) * z * invFx);
    cloud.ptr(y)[x].y = (float)((y - cy) * z * invFy);
    cloud.ptr(y)[x].z = z;
}

void projectToPointCloud(const DeviceArray2D<float> & depth,
                          const DeviceArray2D<float3> & cloud,
                          IntrDoublePrecision & intrinsics,
                          const int & level)
{
    dim3 block (32, 8);
    dim3 grid (divUp (depth.cols (), block.x), divUp (depth.rows (), block.y));

    IntrDoublePrecision intrinsicsLevel = intrinsics(level);

    projectPointsKernel<<<grid, block>>>(depth, cloud, 1.0f / intrinsicsLevel.fx, 1.0f / intrinsicsLevel.fy, intrinsicsLevel.cx, intrinsicsLevel.cy);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

