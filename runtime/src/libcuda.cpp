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

#include "libcuda.h"

#include <dlfcn.h>
#include <assert.h>

LibCUDA* LibCUDA::singleton_ = NULL;
mutex_t LibCUDA::mutex_;

LibCUDA* LibCUDA::GetLibCUDA() {
  mutex_.lock();
  if (singleton_ == NULL)
    singleton_ = new LibCUDA();
  mutex_.unlock();

  return singleton_;
}

LibCUDA::LibCUDA() {
  OpenCUDART();
  OpenCUDADRV();
}

LibCUDA::~LibCUDA() {
  CloseCUDART();
  CloseCUDADRV();
}

void LibCUDA::OpenCUDART() {
  const char* s = getenv("SNURHACCUDA");
  if (s == NULL) {
    fprintf(stderr, "Failed to read environment variable \"SNURHACCUDA\"\n");
    assert(0);
  }

  char buf[1024];
  sprintf(buf, "%s%s", s, RHAC_LIB_CUDART);

  cudart_handle_ = dlopen(buf, RTLD_NOW);
  if (!cudart_handle_) {
    fprintf(stderr, "Failed to load CUDA runtime library from \"%s\"\n", buf);
    assert(0);
  }

  __cudaInitModule = (void(*)(void**))dlsym(cudart_handle_, "__cudaInitModule");
  CHECK_SYMBOL(__cudaInitModule);

  __cudaRegisterFunction = (void(*)(void**, const char*, char*, const char*,
        int, uint3*, uint3*, dim3*, dim3*, int*))
    dlsym(cudart_handle_, "__cudaRegisterFunction");
  CHECK_SYMBOL(__cudaRegisterFunction);

  __cudaRegisterVar = (void(*)(void**, char*, char*, const char*, int, size_t,
        int, int))
    dlsym(cudart_handle_, "__cudaRegisterVar");
  CHECK_SYMBOL(__cudaRegisterVar);

  __cudaRegisterTexture = (void(*)(void**, const struct textureReference*,
        const void**, const char*, int, int, int))
    dlsym(cudart_handle_, "__cudaRegisterTexture");
  CHECK_SYMBOL(__cudaRegisterTexture);

  __cudaRegisterFatBinary = (void**(*)(void*))
    dlsym(cudart_handle_, "__cudaRegisterFatBinary");
  CHECK_SYMBOL(__cudaRegisterFatBinary);

  __cudaRegisterFatBinaryEnd = (void(*)(void**))
    dlsym(cudart_handle_, "__cudaRegisterFatBinaryEnd");
  CHECK_SYMBOL(__cudaRegisterFatBinaryEnd);

  __cudaUnregisterFatBinary = (void(*)(void**))
    dlsym(cudart_handle_, "__cudaUnregisterFatBinary");
  CHECK_SYMBOL(__cudaUnregisterFatBinary);

  __cudaPopCallConfiguration = (cudaError_t(*)(dim3*, dim3*, size_t*, void*))
    dlsym(cudart_handle_, "__cudaPopCallConfiguration");
  CHECK_SYMBOL(__cudaPopCallConfiguration);

  __cudaPushCallConfiguration = (unsigned(*)(dim3,
        dim3, size_t, void*))
    dlsym(cudart_handle_, "__cudaPushCallConfiguration");
  CHECK_SYMBOL(__cudaPushCallConfiguration);

  // TODO - other interfaces  
  cudaGetDeviceCount = (cudaError_t (*)(int *))
    dlsym(cudart_handle_, "cudaGetDeviceCount");
  CHECK_SYMBOL(cudaGetDeviceCount);

  cudaSetDevice = (cudaError_t (*)(int))
    dlsym(cudart_handle_, "cudaSetDevice");
  CHECK_SYMBOL(cudaSetDevice);

  cudaStreamSynchronize = (cudaError_t (*)(cudaStream_t))
    dlsym(cudart_handle_, "cudaStreamSynchronize");
  CHECK_SYMBOL(cudaStreamSynchronize);

  cudaDeviceSynchronize = (cudaError_t (*)(void))
    dlsym(cudart_handle_, "cudaDeviceSynchronize");
  CHECK_SYMBOL(cudaDeviceSynchronize);

  cudaMallocManaged = (cudaError_t(*)(void**, size_t, unsigned int))
    dlsym(cudart_handle_, "cudaMallocManaged");
  CHECK_SYMBOL(cudaMallocManaged);

  cudaMallocHost = (cudaError_t(*)(void**, size_t))
    dlsym(cudart_handle_, "cudaMallocHost");
  CHECK_SYMBOL(cudaMallocHost);

  cudaHostAlloc = (cudaError_t(*)(void**, size_t, unsigned int))
    dlsym(cudart_handle_, "cudaHostAlloc");
  CHECK_SYMBOL(cudaHostAlloc);

  cudaFreeHost = (cudaError_t(*)(void*))
    dlsym(cudart_handle_, "cudaFreeHost");
  CHECK_SYMBOL(cudaFreeHost);

  cudaFree = (cudaError_t(*)(void*))
    dlsym(cudart_handle_, "cudaFree");
  CHECK_SYMBOL(cudaFree);

  cudaGetDeviceProperties = (cudaError_t(*)(struct cudaDeviceProp*, int))
    dlsym(cudart_handle_, "cudaGetDeviceProperties");
  CHECK_SYMBOL(cudaGetDeviceProperties);

  cudaDeviceReset = (cudaError_t(*)(void))
    dlsym(cudart_handle_, "cudaDeviceReset");
  CHECK_SYMBOL(cudaDeviceReset);

  cudaMemset = (cudaError_t(*)(void*, int, size_t))
    dlsym(cudart_handle_, "cudaMemset");
  CHECK_SYMBOL(cudaMemset);

  cudaMemsetAsync = (cudaError_t(*)(void*, int, size_t, cudaStream_t))
    dlsym(cudart_handle_, "cudaMemsetAsync");
  CHECK_SYMBOL(cudaMemsetAsync);

  cudaDeviceGetAttribute = (cudaError_t(*)(int*, cudaDeviceAttr, int))
    dlsym(cudart_handle_, "cudaDeviceGetAttribute");
  CHECK_SYMBOL(cudaDeviceGetAttribute);

  cudaDeviceSetCacheConfig = (cudaError_t(*)(cudaFuncCache))
    dlsym(cudart_handle_, "cudaDeviceSetCacheConfig");
  CHECK_SYMBOL(cudaDeviceSetCacheConfig);

  cudaDeviceGetLimit = (cudaError_t (*)(size_t*, cudaLimit))
    dlsym(cudart_handle_, "cudaDeviceGetLimit");
  CHECK_SYMBOL(cudaDeviceGetLimit);

  cudaStreamCreate = (cudaError_t (*)(cudaStream_t*))
    dlsym(cudart_handle_, "cudaStreamCreate");
  CHECK_SYMBOL(cudaStreamCreate);

  cudaStreamDestroy = (cudaError_t (*)(cudaStream_t))
    dlsym(cudart_handle_, "cudaStreamDestroy");
  CHECK_SYMBOL(cudaStreamDestroy);

  cudaEventCreate = (cudaError_t (*)(cudaEvent_t *))
    dlsym(cudart_handle_, "cudaEventCreate");
  CHECK_SYMBOL(cudaEventCreate);

  cudaEventDestroy = (cudaError_t (*)(cudaEvent_t))
    dlsym(cudart_handle_, "cudaEventDestroy");
  CHECK_SYMBOL(cudaEventDestroy);

  cudaMemcpyToSymbol = (cudaError_t (*)(const void *symbol, const void *src,
        size_t count, size_t offset, cudaMemcpyKind kind))
    dlsym(cudart_handle_, "cudaMemcpyToSymbol");
  CHECK_SYMBOL(cudaMemcpyToSymbol);

  cudaGetSymbolAddress = (cudaError_t (*)(void **, const char*))
    dlsym(cudart_handle_, "cudaGetSymbolAddress");
  CHECK_SYMBOL(cudaGetSymbolAddress);

  cudaCreateChannelDesc = (cudaChannelFormatDesc (*) ( int  x, int  y, 
        int  z, int  w, cudaChannelFormatKind f ))
    dlsym(cudart_handle_, "cudaCreateChannelDesc");
  CHECK_SYMBOL(cudaCreateChannelDesc);

  cudaMemGetInfo = (cudaError_t(*)(size_t*, size_t*))
    dlsym(cudart_handle_, "cudaMemGetInfo");
  CHECK_SYMBOL(cudaMemGetInfo);

  cudaLaunchKernel = (cudaError_t(*)( const void* func,
        dim3 gridDim, dim3 blockDim,
        void** args, size_t sharedMem,
        cudaStream_t stream))
    dlsym(cudart_handle_, "cudaLaunchKernel");
  CHECK_SYMBOL(cudaLaunchKernel);

  cudaMemcpy = (cudaError_t(*)(void*, const void*, size_t, enum cudaMemcpyKind))
    dlsym(cudart_handle_, "cudaMemcpy");
  CHECK_SYMBOL(cudaMemcpy);

  cudaMemcpy2DToArray = (cudaError_t(*)(cudaArray_t, size_t, size_t,
        const void*, size_t, size_t, size_t, cudaMemcpyKind))
    dlsym(cudart_handle_, "cudaMemcpy2DToArray");
  CHECK_SYMBOL(cudaMemcpy2DToArray);

  cudaMallocArray = (cudaError_t(*)(cudaArray_t*, const cudaChannelFormatDesc*,
        size_t, size_t, unsigned int))
    dlsym(cudart_handle_, "cudaMallocArray");
  CHECK_SYMBOL(cudaMallocArray);

  cudaFreeArray = (cudaError_t(*)(cudaArray_t))
    dlsym(cudart_handle_, "cudaFreeArray");
  CHECK_SYMBOL(cudaFreeArray);

  cudaGetChannelDesc = (cudaError_t(*)(cudaChannelFormatDesc*, cudaArray_const_t))
    dlsym(cudart_handle_, "cudaGetChannelDesc");
  CHECK_SYMBOL(cudaGetChannelDesc);

  cudaMemAdvise = (cudaError_t (*)(const void*, size_t, cudaMemoryAdvise, int))
    dlsym(cudart_handle_, "cudaMemAdvise");
  CHECK_SYMBOL(cudaMemAdvise);

  cudaEventRecord = (cudaError_t (*)(cudaEvent_t event, cudaStream_t stream))
    dlsym(cudart_handle_, "cudaEventRecord");
  CHECK_SYMBOL(cudaEventRecord);

  cudaEventElapsedTime = (cudaError_t (*)(float *ms, cudaEvent_t start, cudaEvent_t end))
    dlsym(cudart_handle_, "cudaEventElapsedTime");
  CHECK_SYMBOL(cudaEventElapsedTime);

}

void LibCUDA::CloseCUDART() {
  dlclose(cudart_handle_);
}

void LibCUDA::OpenCUDADRV() {
  cudadrv_handle_ = dlopen(RHAC_LIB_CUDADRV, RTLD_NOW);
  if (!cudadrv_handle_) {
    assert(0);
  }

  cuDeviceGet = (CUresult (*)(CUdevice*, int))
    dlsym(cudadrv_handle_, "cuDeviceGet");
  CHECK_SYMBOL(cuDeviceGet);

  cuDeviceGetAttribute = (CUresult (*)(int*, CUdevice_attribute, CUdevice))
    dlsym(cudadrv_handle_, "cuDeviceGetAttribute");
  CHECK_SYMBOL(cuDeviceGetAttribute);

  cuDevicePrimaryCtxRetain = (CUresult (*)(CUcontext*, CUdevice))
    dlsym(cudadrv_handle_, "cuDevicePrimaryCtxRetain");
  CHECK_SYMBOL(cuDevicePrimaryCtxRetain);

  cuDevicePrimaryCtxRelease = (CUresult (*)(CUdevice))
    dlsym(cudadrv_handle_, "cuDevicePrimaryCtxRelease");
  CHECK_SYMBOL(cuDevicePrimaryCtxRelease);

  cuCtxSetCurrent = (CUresult (*)(CUcontext))
    dlsym(cudadrv_handle_, "cuCtxSetCurrent");
  CHECK_SYMBOL(cuCtxSetCurrent);

  cuModuleLoad = (CUresult (*)(CUmodule*, const char*))
    dlsym(cudadrv_handle_, "cuModuleLoad");
  CHECK_SYMBOL(cuModuleLoad);

  cuModuleGetFunction = (CUresult (*)(CUfunction*, CUmodule, const char*))
    dlsym(cudadrv_handle_, "cuModuleGetFunction");
  CHECK_SYMBOL(cuModuleGetFunction);

  cuModuleGetGlobal = (CUresult (*)(CUdeviceptr*, size_t*, CUmodule, const char*))
    dlsym(cudadrv_handle_, "cuModuleGetGlobal_v2");
  CHECK_SYMBOL(cuModuleGetGlobal);

  cuModuleGetTexRef = (CUresult (*)(CUtexref*, CUmodule, const char*))
    dlsym(cudadrv_handle_, "cuModuleGetTexRef");
  CHECK_SYMBOL(cuModuleGetTexRef);

  cuModuleGetSurfRef = (CUresult (*)(CUsurfref*, CUmodule, const char*))
    dlsym(cudadrv_handle_, "cuModuleGetSurfRef");
  CHECK_SYMBOL(cuModuleGetSurfRef);

  cuFuncSetCacheConfig = (CUresult (*)(CUfunction, CUfunc_cache))
    dlsym(cudadrv_handle_, "cuFuncSetCacheConfig");
  CHECK_SYMBOL(cuFuncSetCacheConfig);

  cuTexRefSetAddress = (CUresult (*)(size_t*, CUtexref, CUdeviceptr, size_t))
    dlsym(cudadrv_handle_, "cuTexRefSetAddress_v2");
  CHECK_SYMBOL(cuTexRefSetAddress);

  cuTexRefSetAddressMode = (CUresult (*)(CUtexref, int, CUaddress_mode))
    dlsym(cudadrv_handle_, "cuTexRefSetAddressMode");
  CHECK_SYMBOL(cuTexRefSetAddressMode);

  cuTexRefSetArray = (CUresult (*)(CUtexref, CUarray, unsigned int))
    dlsym(cudadrv_handle_, "cuTexRefSetArray");
  CHECK_SYMBOL(cuTexRefSetArray);

  cuTexRefSetFilterMode = (CUresult (*)(CUtexref, CUfilter_mode))
    dlsym(cudadrv_handle_, "cuTexRefSetFilterMode");
  CHECK_SYMBOL(cuTexRefSetFilterMode);

  cuTexRefSetFlags = (CUresult (*)(CUtexref, unsigned int))
    dlsym(cudadrv_handle_, "cuTexRefSetFlags");
  CHECK_SYMBOL(cuTexRefSetFlags);

  cuTexRefSetFormat = (CUresult (*)(CUtexref, CUarray_format, int))
    dlsym(cudadrv_handle_, "cuTexRefSetFormat");
  CHECK_SYMBOL(cuTexRefSetFormat);

  cuMemcpyHtoD_v2 = (CUresult (*)(CUdeviceptr, const void*, size_t))
    dlsym(cudadrv_handle_, "cuMemcpyHtoD_v2");
  CHECK_SYMBOL(cuMemcpyHtoD_v2);

  cuMemcpyDtoH_v2 = (CUresult (*)(void*, CUdeviceptr, size_t))
    dlsym(cudadrv_handle_, "cuMemcpyDtoH_v2");
  CHECK_SYMBOL(cuMemcpyDtoH_v2);

  cuLaunchKernel = (CUresult (*)(CUfunction, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
        CUstream, void**, void**))
    dlsym(cudadrv_handle_, "cuLaunchKernel");
  CHECK_SYMBOL(cuLaunchKernel);
}

void LibCUDA::CloseCUDADRV() {
  dlclose(cudadrv_handle_);
}
