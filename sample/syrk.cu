/*****************************************************************************/
/*                                                                           */
/* Copyright (c) 2021 Seoul National University.                             */
/* All rights reserved.                                                      */
/*                                                                           */
/* Redistribution and use in source and binary forms, with or without        */
/* modification, are permitted provided that the following conditions        */
/* are met:                                                                  */
/*   1. Redistributions of source code must retain the above copyright       */
/*      notice, this list of conditions and the following disclaimer.        */
/*   2. Redistributions in binary form must reproduce the above copyright    */
/*      notice, this list of conditions and the following disclaimer in the  */
/*      documentation and/or other materials provided with the distribution. */
/*   3. Neither the name of Seoul National University nor the names of its   */
/*      contributors may be used to endorse or promote products derived      */
/*      from this software without specific prior written permission.        */
/*                                                                           */
/* THIS SOFTWARE IS PROVIDED BY SEOUL NATIONAL UNIVERSITY "AS IS" AND ANY    */
/* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED */
/* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE    */
/* DISCLAIMED. IN NO EVENT SHALL SEOUL NATIONAL UNIVERSITY BE LIABLE FOR ANY */
/* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL        */
/* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS   */
/* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     */
/* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,       */
/* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN  */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           */
/* POSSIBILITY OF SUCH DAMAGE.                                               */
/*                                                                           */
/* Contact information:                                                      */
/*   THUNDER Research Group                                                  */
/*   Department of Computer Science and Engineering                          */
/*   Seoul National University, Seoul 08826, Korea                           */
/*   http://aces.snu.ac.kr                                                   */
/*                                                                           */
/* Contributors:                                                             */
/*   Jaehoon Jung, Daeyoung Park, Gangwon Jo, Jungho Park, and Jaejin Lee    */
/*                                                                           */
/*****************************************************************************/

/**
 * syrk.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

// Problem size
#define N 1024*8
#define M 1024*8

// Thread block dimensions
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

// Declared constant values for alpha and beta (same as values in PolyBench 2.0)
#define alpha 12435
#define beta 4546

// Can switch DATA_TYPE between float and double
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE* A, DATA_TYPE* C) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      A[i*M + j] = ((DATA_TYPE) i*j) / N;
    }

    for (j = 0; j < N; j++) {
      C[i*N + j] = ((DATA_TYPE) i*j + 2) / N;
    }
  }
}

void syrk(DATA_TYPE* A, DATA_TYPE* C) {
  struct timeval t_start, t_end;
  double t_time = 0;

  //  C := alpha*A*A' + beta*C
  int i, j, k;

  gettimeofday(&t_start, NULL);
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i*N + j] *= beta;
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < M; k++) {
        C[i*N + j] += alpha * A[i*M + k] * A[j*M + k];
      }
    }
  }
  gettimeofday(&t_end, NULL);

  t_time += (t_end.tv_sec - t_start.tv_sec) 
    + (t_end.tv_usec - t_start.tv_usec) / 1000000.0;
  printf("---> Elapsed Time (CPU): %.6lfs\n", t_time);
}

float absVal(float a) {
	if (a < 0)
		return (a * -1);
  else
		return a;
}

float percentDiff(double val1, double val2) {
  if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01)) {
    return 0.0f;
  }
  else {
    return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + 0.00000001f)));
  }
} 

void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu) {
  int i,j,fail;
  fail = 0;

  // Compare C with D
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      if (percentDiff(C[i*N + j], C_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  printf("Error count: %d\n", fail);
}

void GPU_argv_init() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
  printf("Setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
  cudaSetDevice(GPU_DEVICE);
  return;
}

__global__ void syrk_kernel(DATA_TYPE ALPHA, DATA_TYPE BETA, DATA_TYPE *a, DATA_TYPE *c) {
  //  C := alpha*A*A' + beta*C
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < N) && (j < N)) {
    c[i * N + j] *= beta;
    int k;    
    for(k=0; k< M; k++) {
      c[i * N + j] += alpha * a[i * M + k] * a[j * M + k];
    }
  }
}


void syrkCuda(DATA_TYPE* A, DATA_TYPE* C, DATA_TYPE* C_outputFromGpu) {
  struct timeval t_start, t_end;
  double t_time = 0;

  DATA_TYPE* A_gpu;
  DATA_TYPE* C_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * M);
  cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * N * N);

  gettimeofday(&t_start, NULL);
  cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)(ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X))),
      (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_Y)));
  syrk_kernel<<<grid,block>>>(alpha, beta, A_gpu,C_gpu);
  cudaDeviceSynchronize();

  cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);
  gettimeofday(&t_end, NULL);

  cudaFree(A_gpu);
  cudaFree(C_gpu);

  t_time += (t_end.tv_sec - t_start.tv_sec) 
    + (t_end.tv_usec - t_start.tv_usec) / 1000000.0;
  printf("---> Elapsed Time (GPU): %.6lfs\n", t_time);
}


int main(int argc, char** argv) {
  DATA_TYPE* A;
  DATA_TYPE* C;
  DATA_TYPE* C_outputFromGpu;

  A = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
  C = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
  C_outputFromGpu = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));

  init_arrays(A, C);

  GPU_argv_init();  
  printf("Running syrk benchmark on GPU...\n");
  syrkCuda(A, C, C_outputFromGpu);

  if (argc > 1) {
    printf("Running syrk benchmark on CPU...\n");
    syrk(A, C);
    compareResults(C, C_outputFromGpu);
  }

  free(A);
  free(C);
  free(C_outputFromGpu);

  return 0;
}

