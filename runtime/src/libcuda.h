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

#ifndef __RHAC_LIBCUDA_H__
#define __RHAC_LIBCUDA_H__

#include "utils.h"

#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_SYMBOL(x)                             \
  do {                                              \
    if (x == NULL) {                                \
      printf("Failed to load symbol " #x "\n");     \
      assert(0);                                    \
      return;                                       \
    }                                               \
  } while (0)

class LibCUDA {
  public:

    void OpenCUDART();
    void CloseCUDART();
    void OpenCUDADRV();
    void CloseCUDADRV();

    void (*__cudaInitModule)(void **fatCubinHandle);
    void (*__cudaRegisterFunction)(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
    void (*__cudaRegisterVar)(void **fatCubinHandle, char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, size_t size,
        int constant, int global);
    void (*__cudaRegisterTexture)(void **fatCubinHandle,
        const struct textureReference *hostVar, const void **deviceAddress,
        const char *deviceName, int dim, int norm, int ext);
    void** (*__cudaRegisterFatBinary)(void *fatCubin);
    void (*__cudaRegisterFatBinaryEnd)(void **fatCubinHandle);
    void (*__cudaUnregisterFatBinary)(void **fatCubinHandle);
    cudaError_t (*__cudaPopCallConfiguration)(dim3 *gridDim,
        dim3 *blockDim,  size_t *sharedMem, void *stream);
    unsigned (*__cudaPushCallConfiguration)(dim3 gridDim,
        dim3 blockDim,  size_t sharedMem, void *stream);

    cudaError_t (*cudaGetDeviceCount)(int *count);
    cudaError_t (*cudaSetDevice)(int);
    cudaError_t (*cudaStreamSynchronize)(cudaStream_t);
    cudaError_t (*cudaDeviceSynchronize)(void);
    cudaError_t (*cudaMallocManaged)(void**, size_t, unsigned);
    cudaError_t (*cudaMallocHost)(void **, size_t);
    cudaError_t (*cudaFreeHost)(void *);
    cudaError_t (*cudaFree)(void *);
    cudaError_t (*cudaHostAlloc)(void**, size_t, unsigned int);
    cudaError_t (*cudaGetDeviceProperties)(struct cudaDeviceProp*, int);
    cudaError_t (*cudaDeviceReset)(void);
    cudaError_t (*cudaMemset)(void*, int, size_t);
    cudaError_t (*cudaMemsetAsync)(void*, int, size_t, cudaStream_t);
    cudaError_t (*cudaDeviceGetAttribute)(int*, cudaDeviceAttr, int);
    cudaError_t (*cudaDeviceSetCacheConfig)(cudaFuncCache);
    cudaError_t (*cudaDeviceGetLimit)(size_t*, cudaLimit);
    cudaError_t (*cudaStreamCreate)(cudaStream_t*);
    cudaError_t (*cudaStreamDestroy)(cudaStream_t);
    cudaError_t (*cudaEventCreate)(cudaEvent_t*);
    cudaError_t (*cudaEventDestroy)(cudaEvent_t);
    cudaError_t (*cudaMemcpyToSymbol)(const void *, const void *, size_t, size_t, cudaMemcpyKind);
    cudaError_t (*cudaGetSymbolAddress)(void **, const char *);
    cudaChannelFormatDesc (*cudaCreateChannelDesc)(int, int, int, int, cudaChannelFormatKind);
    cudaError_t (*cudaMemGetInfo)(size_t*, size_t*);

    cudaError_t (*cudaLaunchKernel)( const void* func,
        dim3 gridDim, dim3 blockDim,
        void** args, size_t sharedMem,
        cudaStream_t stream );

    cudaError_t (*cudaMemcpy)(void*, const void*, size_t, enum cudaMemcpyKind);
    cudaError_t (*cudaMemcpy2DToArray)(cudaArray_t, size_t, size_t,
        const void*, size_t, size_t, size_t, cudaMemcpyKind);

    cudaError_t (*cudaMallocArray)(cudaArray_t*, const cudaChannelFormatDesc*,
        size_t, size_t, unsigned int);
    cudaError_t (*cudaFreeArray)(cudaArray_t);

    cudaError_t (*cudaGetChannelDesc)(cudaChannelFormatDesc*, cudaArray_const_t);
    cudaError_t (*cudaMemAdvise)(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device);
    cudaError_t (*cudaEventRecord)(cudaEvent_t event, cudaStream_t stream);
    cudaError_t (*cudaEventElapsedTime)(float *ms, cudaEvent_t start, cudaEvent_t end);

    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuDeviceGetAttribute)(int*, CUdevice_attribute, CUdevice);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
    CUresult (*cuCtxSetCurrent)(CUcontext);

    CUresult (*cuModuleLoad)(CUmodule*, const char*);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*cuModuleGetGlobal)(CUdeviceptr*, size_t*, CUmodule, const char*);
    CUresult (*cuModuleGetTexRef)(CUtexref*, CUmodule, const char*);
    CUresult (*cuModuleGetSurfRef)(CUsurfref*, CUmodule, const char*);

    CUresult (*cuFuncSetCacheConfig)(CUfunction, CUfunc_cache);

    CUresult (*cuTexRefSetAddress)(size_t*, CUtexref, CUdeviceptr, size_t);
    CUresult (*cuTexRefSetAddressMode)(CUtexref, int, CUaddress_mode);
    CUresult (*cuTexRefSetArray)(CUtexref, CUarray, unsigned int);
    CUresult (*cuTexRefSetFilterMode)(CUtexref, CUfilter_mode);
    CUresult (*cuTexRefSetFlags)(CUtexref, unsigned int);
    CUresult (*cuTexRefSetFormat)(CUtexref, CUarray_format, int);

    CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void*, size_t);
    CUresult (*cuMemcpyDtoH_v2)(void*, CUdeviceptr, size_t);

    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
        CUstream, void**, void**);

  private:

    void *cudart_handle_;
    void *cudadrv_handle_;

  // for singleton
  public:
    static LibCUDA* GetLibCUDA();

  private:
    LibCUDA();
    ~LibCUDA();

    static LibCUDA* singleton_;
    static mutex_t mutex_;
};

#endif // __RHAC_LIBCUDA_H__
