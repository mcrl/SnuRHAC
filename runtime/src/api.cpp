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

#include "api.h"
#include "libcuda.h"
#include "libmapa.h"
#include "rhac.h"
#include "platform.h"
#include "rhac_command.h"
#include "rhac_driver.h"
#include "rhac_event.h"
#include "executor.h"
#include "fatbin_handler.h"
#include "rhac_barrier.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <map>
#include <math.h>
#include <unistd.h>

#define TODO() \
  fprintf(stderr, "*** %s function is not implemented yet ***", __FUNCTION__);\
  assert(0);\
  return cudaSuccess;

extern "C"
void __cudaInitModule(void **fatCubinHandle) 
{
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();
  libcuda->__cudaInitModule(fatCubinHandle);
}

extern "C"
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
    char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
    uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) 
{
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();

  fatbin_handler->RegisterFunction(fatCubinHandle, hostFun, deviceFun);

  RHAC_LOG(" %s : FuncName : %s", __func__, deviceName);
  libcuda->__cudaRegisterFunction(fatCubinHandle, 
                                  hostFun,
                                  deviceFun,
                                  deviceName,
                                  thread_limit,
                                  tid, bid,
                                  bDim, gDim, wSize);
}

extern "C"
void  __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
    char *deviceAddress, const char *deviceName, int ext, size_t size,
    int constant, int global) 
{
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();

  fatbin_handler->RegisterVar(fatCubinHandle, hostVar, deviceAddress);

  RHAC_LOG(" %s : deviceName : %s", __func__, deviceName);
  libcuda->__cudaRegisterVar(fatCubinHandle, 
                             hostVar,
                             deviceAddress, 
                             deviceName, 
                             ext, size, 
                             constant, global);
}

extern "C"
void __cudaRegisterTexture(void **fatCubinHandle,
    const struct textureReference *hostVar, const void **deviceAddress,
    const char *deviceName, int dim, int norm, int ext) 
{
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();

  fatbin_handler->RegisterTexture(fatCubinHandle, hostVar, deviceAddress);

  libcuda->__cudaRegisterTexture(fatCubinHandle,
                                 hostVar, 
                                 deviceAddress, 
                                 deviceName, 
                                 dim, norm, ext);
}

extern "C"
void**  __cudaRegisterFatBinary(void *fatCubin) 
{
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();

  void** cuda_handle = libcuda->__cudaRegisterFatBinary(fatCubin);
  fatbin_handler->RegisterFatbinary(cuda_handle);

  return cuda_handle;
}

extern "C"
void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();
  libcuda->__cudaRegisterFatBinaryEnd(fatCubinHandle);
}

extern "C"
void __cudaUnregisterFatBinary(void **fatCubinHandle) 
{
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();
  libcuda->__cudaUnregisterFatBinary(fatCubinHandle);
}

extern "C"
cudaError_t __cudaPopCallConfiguration(dim3 *gridDim,
    dim3 *blockDim,  size_t *sharedMem, void *stream) 
{
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();
  return libcuda->__cudaPopCallConfiguration(gridDim, 
                                             blockDim,
                                             sharedMem, 
                                             stream);
}

extern "C"
unsigned __cudaPushCallConfiguration(dim3 gridDim,
    dim3 blockDim, size_t sharedMem, void *stream) 
{
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();
  return libcuda->__cudaPushCallConfiguration(gridDim, 
                                              blockDim,
                                              sharedMem, 
                                              stream);

}

//=============================================================================
// CUDA Runtime API
// 5.1. Device Management
//=============================================================================
cudaError_t cudaDeviceSynchronize() 
{
  rhac_platform.FinishAllRequestQueue();
  return cudaSuccess;
}

cudaError_t cudaThreadSynchronize() 
{
  return cudaDeviceSynchronize();
}

cudaError_t cudaDeviceReset() 
{
  int n, d;
  int num_nodes = rhac_platform.GetClusterNumNodes();
  int num_devices;

  RHAC_LOG("%s", __func__);

  for (n = 0; n < num_nodes; n++) {
    num_devices = rhac_platform.GetNumDevicesIn(n);

    for (d = 0; d < num_devices; d++) {
      RHACCommand *cmd;
      cmd = new RHACCommand();
      cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                          DReset,
                          n, d);
      rhac_platform.EnqueueCommand(cmd);
    }
  }

  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

cudaError_t cudaSetDevice(int deviceId) 
{
  return cudaSuccess;
}

cudaError_t cudaGetDevice(int* device) 
{
  (*device) = 0;
  return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int* count) 
{
  (*count) = 1;
  return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) 
{
  // FIXME 
  // TODO();
  assert(device == 0);

  cudaError_t cuda_err;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  cuda_err = libcuda->cudaGetDeviceProperties(prop, device);
  return cuda_err;
}

cudaError_t cudaDeviceGetAttribute(int* pi, cudaDeviceAttr attr, int deviceId) 
{
  // FIXME
  // TODO();
  assert(deviceId == 0);

  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  return libcuda->cudaDeviceGetAttribute(pi, attr, deviceId);
}

cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) 
{
  rhac_platform.SetCudaCacheConfig(cacheConfig);

  for (int n = 0; n < rhac_platform.GetClusterNumNodes(); n++) {
    for (int d = 0; d < rhac_platform.GetNumDevicesIn(n); d++) {
      RHACCommandDSetCacheConfig *cmd;
      cmd = new RHACCommandDSetCacheConfig();
      cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                          DSetCacheConfig,
                          n, d);
      cmd->SetCacheConfig(cacheConfig);

      rhac_platform.EnqueueCommand((RHACCommand *)cmd);
    }
  }

  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache *cacheConfig) 
{
  *cacheConfig = rhac_platform.GetCudaCacheConfig();
  return cudaSuccess;
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, cudaLimit limit) 
{
  // FIXME
  // TODO();
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  return libcuda->cudaDeviceGetLimit(pValue, limit);
}

cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache config) 
{
  int fatbin_index;
  const char *func_name;
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();
  std::pair<int, char*> function_indicator =
    fatbin_handler->LookupFunction((const char*)func);

  fatbin_index = function_indicator.first;
  func_name = function_indicator.second;

  if (fatbin_index < 1) {
    fprintf(stderr, "Failed to lookup function\n");
    return cudaErrorInvalidDeviceFunction;
  }

  for (int n = 0; n < rhac_platform.GetClusterNumNodes(); n++) {
    for (int d = 0; d < rhac_platform.GetNumDevicesIn(n); d++) {
      RHACCommandDFuncSetCacheConfig *cmd;
      cmd = new RHACCommandDFuncSetCacheConfig();
      cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                          DFuncSetCacheConfig,
                          n, d);
      cmd->SetFatbinIndex(fatbin_index);
      cmd->SetFuncName(func_name);
      cmd->SetCacheConfig(config);

      rhac_platform.EnqueueCommand((RHACCommand *)cmd);
    }
  }

  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

cudaError_t cudaThreadExit() 
{
  // Deprecated and replaced to cudaDeviceReset
  return cudaDeviceReset();
}

//=============================================================================
// CUDA Runtime API
// 5.3. Error Handling
//=============================================================================
cudaError_t cudaPeekAtLastError() 
{
  return cudaGetLastError();
}

const char* cudaGetErrorName(cudaError_t cudaError) {
  switch (cudaError) {
  case cudaSuccess                           : return "cudaSuccess";
  case cudaErrorMissingConfiguration         : return "cudaErrorMissingConfiguration";
  case cudaErrorMemoryAllocation             : return "cudaErrorMemoryAllocation";
  case cudaErrorInitializationError          : return "cudaErrorInitializationError";
  case cudaErrorLaunchFailure                : return "cudaErrorLaunchFailure";
  case cudaErrorPriorLaunchFailure           : return "cudaErrorPriorLaunchFailure";
  case cudaErrorLaunchTimeout                : return "cudaErrorLaunchTimeout";
  case cudaErrorLaunchOutOfResources         : return "cudaErrorLaunchOutOfResources";
  case cudaErrorInvalidDeviceFunction        : return "cudaErrorInvalidDeviceFunction";
  case cudaErrorInvalidConfiguration         : return "cudaErrorInvalidConfiguration";
  case cudaErrorInvalidDevice                : return "cudaErrorInvalidDevice";
  case cudaErrorInvalidValue                 : return "cudaErrorInvalidValue";
  case cudaErrorInvalidPitchValue            : return "cudaErrorInvalidPitchValue";
  case cudaErrorInvalidSymbol                : return "cudaErrorInvalidSymbol";
  case cudaErrorMapBufferObjectFailed        : return "cudaErrorMapBufferObjectFailed";
  case cudaErrorUnmapBufferObjectFailed      : return "cudaErrorUnmapBufferObjectFailed";
  case cudaErrorInvalidHostPointer           : return "cudaErrorInvalidHostPointer";
  case cudaErrorInvalidDevicePointer         : return "cudaErrorInvalidDevicePointer";
  case cudaErrorInvalidTexture               : return "cudaErrorInvalidTexture";
  case cudaErrorInvalidTextureBinding        : return "cudaErrorInvalidTextureBinding";
  case cudaErrorInvalidChannelDescriptor     : return "cudaErrorInvalidChannelDescriptor";
  case cudaErrorInvalidMemcpyDirection       : return "cudaErrorInvalidMemcpyDirection";
  case cudaErrorAddressOfConstant            : return "cudaErrorAddressOfConstant";
  case cudaErrorTextureFetchFailed           : return "cudaErrorTextureFetchFailed";
  case cudaErrorTextureNotBound              : return "cudaErrorTextureNotBound";
  case cudaErrorSynchronizationError         : return "cudaErrorSynchronizationError";
  case cudaErrorInvalidFilterSetting         : return "cudaErrorInvalidFilterSetting";
  case cudaErrorInvalidNormSetting           : return "cudaErrorInvalidNormSetting";
  case cudaErrorMixedDeviceExecution         : return "cudaErrorMixedDeviceExecution";
  case cudaErrorCudartUnloading              : return "cudaErrorCudartUnloading";
  case cudaErrorUnknown                      : return "cudaErrorUnknown";
  case cudaErrorNotYetImplemented            : return "cudaErrorNotYetImplemented";
  case cudaErrorMemoryValueTooLarge          : return "cudaErrorMemoryValueTooLarge";
  case cudaErrorInvalidResourceHandle        : return "cudaErrorInvalidResourceHandle";
  case cudaErrorNotReady                     : return "cudaErrorNotReady";
  case cudaErrorInsufficientDriver           : return "cudaErrorInsufficientDriver";
  case cudaErrorSetOnActiveProcess           : return "cudaErrorSetOnActiveProcess";
  case cudaErrorInvalidSurface               : return "cudaErrorInvalidSurface";
  case cudaErrorNoDevice                     : return "cudaErrorNoDevice";
  case cudaErrorECCUncorrectable             : return "cudaErrorECCUncorrectable";
  case cudaErrorSharedObjectSymbolNotFound   : return "cudaErrorSharedObjectSymbolNotFound";
  case cudaErrorSharedObjectInitFailed       : return "cudaErrorSharedObjectInitFailed";
  case cudaErrorUnsupportedLimit             : return "cudaErrorUnsupportedLimit";
  case cudaErrorDuplicateVariableName        : return "cudaErrorDuplicateVariableName";
  case cudaErrorDuplicateTextureName         : return "cudaErrorDuplicateTextureName";
  case cudaErrorDuplicateSurfaceName         : return "cudaErrorDuplicateSurfaceName";
  case cudaErrorDevicesUnavailable           : return "cudaErrorDevicesUnavailable";
  case cudaErrorInvalidKernelImage           : return "cudaErrorInvalidKernelImage";
  case cudaErrorNoKernelImageForDevice       : return "cudaErrorNoKernelImageForDevice";
  case cudaErrorIncompatibleDriverContext    : return "cudaErrorIncompatibleDriverContext";
  case cudaErrorPeerAccessAlreadyEnabled     : return "cudaErrorPeerAccessAlreadyEnabled";
  case cudaErrorPeerAccessNotEnabled         : return "cudaErrorPeerAccessNotEnabled";
  case cudaErrorDeviceAlreadyInUse           : return "cudaErrorDeviceAlreadyInUse";
  case cudaErrorProfilerDisabled             : return "cudaErrorProfilerDisabled";
  case cudaErrorProfilerNotInitialized       : return "cudaErrorProfilerNotInitialized";
  case cudaErrorProfilerAlreadyStarted       : return "cudaErrorProfilerAlreadyStarted";
  case cudaErrorProfilerAlreadyStopped       : return "cudaErrorProfilerAlreadyStopped";
  case cudaErrorAssert                       : return "cudaErrorAssert";
  case cudaErrorTooManyPeers                 : return "cudaErrorTooManyPeers";
  case cudaErrorHostMemoryAlreadyRegistered  : return "cudaErrorHostMemoryAlreadyRegistered";
  case cudaErrorHostMemoryNotRegistered      : return "cudaErrorHostMemoryNotRegistered";
  case cudaErrorOperatingSystem              : return "cudaErrorOperatingSystem";
  case cudaErrorPeerAccessUnsupported        : return "cudaErrorPeerAccessUnsupported";
  case cudaErrorLaunchMaxDepthExceeded       : return "cudaErrorLaunchMaxDepthExceeded";
  case cudaErrorLaunchFileScopedTex          : return "cudaErrorLaunchFileScopedTex";
  case cudaErrorLaunchFileScopedSurf         : return "cudaErrorLaunchFileScopedSurf";
  case cudaErrorSyncDepthExceeded            : return "cudaErrorSyncDepthExceeded";
  case cudaErrorLaunchPendingCountExceeded   : return "cudaErrorLaunchPendingCountExceeded";
  case cudaErrorNotPermitted                 : return "cudaErrorNotPermitted";
  case cudaErrorNotSupported                 : return "cudaErrorNotSupported";
  case cudaErrorHardwareStackError           : return "cudaErrorHardwareStackError";
  case cudaErrorIllegalInstruction           : return "cudaErrorIllegalInstruction";
  case cudaErrorMisalignedAddress            : return "cudaErrorMisalignedAddress";
  case cudaErrorInvalidAddressSpace          : return "cudaErrorInvalidAddressSpace";
  case cudaErrorInvalidPc                    : return "cudaErrorInvalidPc";
  case cudaErrorIllegalAddress               : return "cudaErrorIllegalAddress";
  case cudaErrorInvalidPtx                   : return "cudaErrorInvalidPtx";
  case cudaErrorInvalidGraphicsContext       : return "cudaErrorInvalidGraphicsContext";
  case cudaErrorNvlinkUncorrectable          : return "cudaErrorNvlinkUncorrectable";
  case cudaErrorStartupFailure               : return "cudaErrorStartupFailure";
  case cudaErrorApiFailureBase               : return "cudaErrorApiFailureBase";
  default                                    : return "cudaInvalidError";
  }
}

const char* cudaGetErrorString(cudaError_t cudaError) {
  // TODO
  switch (cudaError) {
  case cudaSuccess                           : return "cudaSuccess";
  case cudaErrorMissingConfiguration         : return "cudaErrorMissingConfiguration";
  case cudaErrorMemoryAllocation             : return "cudaErrorMemoryAllocation";
  case cudaErrorInitializationError          : return "cudaErrorInitializationError";
  case cudaErrorLaunchFailure                : return "cudaErrorLaunchFailure";
  case cudaErrorPriorLaunchFailure           : return "cudaErrorPriorLaunchFailure";
  case cudaErrorLaunchTimeout                : return "cudaErrorLaunchTimeout";
  case cudaErrorLaunchOutOfResources         : return "cudaErrorLaunchOutOfResources";
  case cudaErrorInvalidDeviceFunction        : return "cudaErrorInvalidDeviceFunction";
  case cudaErrorInvalidConfiguration         : return "cudaErrorInvalidConfiguration";
  case cudaErrorInvalidDevice                : return "cudaErrorInvalidDevice";
  case cudaErrorInvalidValue                 : return "cudaErrorInvalidValue";
  case cudaErrorInvalidPitchValue            : return "cudaErrorInvalidPitchValue";
  case cudaErrorInvalidSymbol                : return "cudaErrorInvalidSymbol";
  case cudaErrorMapBufferObjectFailed        : return "cudaErrorMapBufferObjectFailed";
  case cudaErrorUnmapBufferObjectFailed      : return "cudaErrorUnmapBufferObjectFailed";
  case cudaErrorInvalidHostPointer           : return "cudaErrorInvalidHostPointer";
  case cudaErrorInvalidDevicePointer         : return "cudaErrorInvalidDevicePointer";
  case cudaErrorInvalidTexture               : return "cudaErrorInvalidTexture";
  case cudaErrorInvalidTextureBinding        : return "cudaErrorInvalidTextureBinding";
  case cudaErrorInvalidChannelDescriptor     : return "cudaErrorInvalidChannelDescriptor";
  case cudaErrorInvalidMemcpyDirection       : return "cudaErrorInvalidMemcpyDirection";
  case cudaErrorAddressOfConstant            : return "cudaErrorAddressOfConstant";
  case cudaErrorTextureFetchFailed           : return "cudaErrorTextureFetchFailed";
  case cudaErrorTextureNotBound              : return "cudaErrorTextureNotBound";
  case cudaErrorSynchronizationError         : return "cudaErrorSynchronizationError";
  case cudaErrorInvalidFilterSetting         : return "cudaErrorInvalidFilterSetting";
  case cudaErrorInvalidNormSetting           : return "cudaErrorInvalidNormSetting";
  case cudaErrorMixedDeviceExecution         : return "cudaErrorMixedDeviceExecution";
  case cudaErrorCudartUnloading              : return "cudaErrorCudartUnloading";
  case cudaErrorUnknown                      : return "cudaErrorUnknown";
  case cudaErrorNotYetImplemented            : return "cudaErrorNotYetImplemented";
  case cudaErrorMemoryValueTooLarge          : return "cudaErrorMemoryValueTooLarge";
  case cudaErrorInvalidResourceHandle        : return "cudaErrorInvalidResourceHandle";
  case cudaErrorNotReady                     : return "cudaErrorNotReady";
  case cudaErrorInsufficientDriver           : return "cudaErrorInsufficientDriver";
  case cudaErrorSetOnActiveProcess           : return "cudaErrorSetOnActiveProcess";
  case cudaErrorInvalidSurface               : return "cudaErrorInvalidSurface";
  case cudaErrorNoDevice                     : return "cudaErrorNoDevice";
  case cudaErrorECCUncorrectable             : return "cudaErrorECCUncorrectable";
  case cudaErrorSharedObjectSymbolNotFound   : return "cudaErrorSharedObjectSymbolNotFound";
  case cudaErrorSharedObjectInitFailed       : return "cudaErrorSharedObjectInitFailed";
  case cudaErrorUnsupportedLimit             : return "cudaErrorUnsupportedLimit";
  case cudaErrorDuplicateVariableName        : return "cudaErrorDuplicateVariableName";
  case cudaErrorDuplicateTextureName         : return "cudaErrorDuplicateTextureName";
  case cudaErrorDuplicateSurfaceName         : return "cudaErrorDuplicateSurfaceName";
  case cudaErrorDevicesUnavailable           : return "cudaErrorDevicesUnavailable";
  case cudaErrorInvalidKernelImage           : return "cudaErrorInvalidKernelImage";
  case cudaErrorNoKernelImageForDevice       : return "cudaErrorNoKernelImageForDevice";
  case cudaErrorIncompatibleDriverContext    : return "cudaErrorIncompatibleDriverContext";
  case cudaErrorPeerAccessAlreadyEnabled     : return "cudaErrorPeerAccessAlreadyEnabled";
  case cudaErrorPeerAccessNotEnabled         : return "cudaErrorPeerAccessNotEnabled";
  case cudaErrorDeviceAlreadyInUse           : return "cudaErrorDeviceAlreadyInUse";
  case cudaErrorProfilerDisabled             : return "cudaErrorProfilerDisabled";
  case cudaErrorProfilerNotInitialized       : return "cudaErrorProfilerNotInitialized";
  case cudaErrorProfilerAlreadyStarted       : return "cudaErrorProfilerAlreadyStarted";
  case cudaErrorProfilerAlreadyStopped       : return "cudaErrorProfilerAlreadyStopped";
  case cudaErrorAssert                       : return "cudaErrorAssert";
  case cudaErrorTooManyPeers                 : return "cudaErrorTooManyPeers";
  case cudaErrorHostMemoryAlreadyRegistered  : return "cudaErrorHostMemoryAlreadyRegistered";
  case cudaErrorHostMemoryNotRegistered      : return "cudaErrorHostMemoryNotRegistered";
  case cudaErrorOperatingSystem              : return "cudaErrorOperatingSystem";
  case cudaErrorPeerAccessUnsupported        : return "cudaErrorPeerAccessUnsupported";
  case cudaErrorLaunchMaxDepthExceeded       : return "cudaErrorLaunchMaxDepthExceeded";
  case cudaErrorLaunchFileScopedTex          : return "cudaErrorLaunchFileScopedTex";
  case cudaErrorLaunchFileScopedSurf         : return "cudaErrorLaunchFileScopedSurf";
  case cudaErrorSyncDepthExceeded            : return "cudaErrorSyncDepthExceeded";
  case cudaErrorLaunchPendingCountExceeded   : return "cudaErrorLaunchPendingCountExceeded";
  case cudaErrorNotPermitted                 : return "cudaErrorNotPermitted";
  case cudaErrorNotSupported                 : return "cudaErrorNotSupported";
  case cudaErrorHardwareStackError           : return "cudaErrorHardwareStackError";
  case cudaErrorIllegalInstruction           : return "cudaErrorIllegalInstruction";
  case cudaErrorMisalignedAddress            : return "cudaErrorMisalignedAddress";
  case cudaErrorInvalidAddressSpace          : return "cudaErrorInvalidAddressSpace";
  case cudaErrorInvalidPc                    : return "cudaErrorInvalidPc";
  case cudaErrorIllegalAddress               : return "cudaErrorIllegalAddress";
  case cudaErrorInvalidPtx                   : return "cudaErrorInvalidPtx";
  case cudaErrorInvalidGraphicsContext       : return "cudaErrorInvalidGraphicsContext";
  case cudaErrorNvlinkUncorrectable          : return "cudaErrorNvlinkUncorrectable";
  case cudaErrorStartupFailure               : return "cudaErrorStartupFailure";
  case cudaErrorApiFailureBase               : return "cudaErrorApiFailureBase";
  default                                    : return "cudaInvalidError";
  }
}

cudaError_t cudaGetLastError() {
  //FIXME
  //TODO();
  return cudaSuccess;
}

//=============================================================================
// CUDA Runtime API
// 5.4. Stream Management
//=============================================================================
cudaError_t cudaStreamCreate(cudaStream_t *stream) 
{
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  cudaError_t err = libcuda->cudaStreamCreate(stream);
  return err;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) 
{
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  return libcuda->cudaStreamDestroy(stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) 
{
  // FIXME
  // TODO();
  rhac_platform.FinishAllRequestQueue();
  return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
    unsigned int flags) {
  // we just bypass this API because RHAC internally uses only one stream
  return cudaSuccess;
}

//=============================================================================
// CUDA Runtime API
// 5.5. Event Management
//=============================================================================
cudaError_t cudaEventCreate(cudaEvent_t* event) 
{
  // 1. node executor : create rhac event, cuda event
  // 2. node executor : add (cuda event, rhac_event) on rhac_platform.map
  // 3. host thread : wait until 1 and 2 finish
  // 4. host thread : return 
  // Notice - only host reads the map(including rhac_event's data), 
  // and only node executor writes the map(including rhac_event's data)
  // when the node executor writes the map(including rhac_event's data),
  // host waits the write on rhac_platform.FinishAllRequestQueue in this fucntion.
  // So there is no concurrency issue.
  // But, maybe the write result will not be visible to host thread immediately
  RHACCommandNEventCreate *e_cmd;
  e_cmd = new RHACCommandNEventCreate();

  e_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                        NEventCreate,
                        0, -1);
  e_cmd->SetEventPtr(event);
  rhac_platform.EnqueueCommand((RHACCommand *)e_cmd);

  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) 
{
  // Enqueue Global Barrier
  rhac_platform.RhacBarrierEnqueue();

  RHACEvent *rhac_event;
  rhac_event = rhac_platform.GetEvent(event);

  // Register event id and enqueue Command
  RHACCommand *cmd;
  RHACCommandNEventRecord *e_cmd;
  int nNodes, nDevs;
  nNodes = rhac_platform.GetClusterNumNodes();


  for (int n = 0; n < nNodes; n++) {
    e_cmd = new RHACCommandNEventRecord();
    e_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                        NEventRecord,
                        n, -1);
    if (n == 0) {
      e_cmd->SetEvent(event);
      e_cmd->SetStream(stream);
    }

    rhac_event->RegisterEventIDs(e_cmd->GetCommandID(), n);
    rhac_platform.EnqueueCommand((RHACCommand*)e_cmd);

    nDevs = rhac_platform.GetNumDevicesIn(n);
    for (int d = 0; d < nDevs; d++) {
      cmd = new RHACCommand();
      cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                          DEvent,
                          n, d);
      rhac_event->RegisterEventIDs(cmd->GetCommandID(), n, d);
      rhac_platform.EnqueueCommand((RHACCommand *)cmd);
    }
  }

  return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) 
{
  RHACEvent *rhac_event;
  rhac_event = rhac_platform.GetEvent(event);
  rhac_platform.EraseEvent(event);
  delete rhac_event;

  return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) 
{
  RHACEvent *rhac_event;
  rhac_event = rhac_platform.GetEvent(event);
  rhac_event->WaitEvent();

  return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) 
{
  cudaError_t cuda_err;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  cuda_err = libcuda->cudaEventElapsedTime(ms, start, end);
  return cuda_err;
}

cudaError_t cudaEventQuery(cudaEvent_t event) 
{
  cudaError_t ret;
  RHACEvent *rhac_event;
  rhac_event = rhac_platform.GetEvent(event);
  if (rhac_event->QueryEvent()) {
    /* all works are completed */
    ret = cudaSuccess;
  }
  else {
    /* all works are completed */
    ret = cudaErrorNotReady;
  }

  return ret;
}

//=============================================================================
// CUDA Runtime API
// 5.7. Execution Control
//=============================================================================

// Get Single Image WorkItem Distribution
enum BlockBoundDim {
  BLOCK_BOUND_X,
  BLOCK_BOUND_Y,
  BLOCK_BOUND_Z
};

enum BlockBoundDim GetBlockBound(unsigned bound[], dim3 gridDim, int node, int dev)
{
  int cluster_num_devs = rhac_platform.GetClusterNumDevices();
  int cluster_dev_idx = rhac_platform.GetClusterDeviceIndex(node, dev);

  unsigned blocks, res;
  unsigned my_lower, my_upper;

  if (gridDim.z >= (unsigned)cluster_num_devs) {
    blocks = gridDim.z / cluster_num_devs;
    res = gridDim.z % cluster_num_devs;
    if ((unsigned) cluster_dev_idx < res) {
      my_lower = cluster_dev_idx * blocks + cluster_dev_idx;
      my_upper = my_lower + blocks;
    }
    else {
      my_lower = cluster_dev_idx * blocks + res;
      my_upper = my_lower + blocks - 1;
    }

    bound[0] = 0;
    bound[1] = gridDim.x - 1;
    bound[2] = 0;
    bound[3] = gridDim.y - 1;
    bound[4] = my_lower;
    bound[5] = my_upper;
    return BLOCK_BOUND_Z;
  }
  else if (gridDim.y >= (unsigned)cluster_num_devs) {
    blocks = gridDim.y / cluster_num_devs;
    res = gridDim.y % cluster_num_devs;
    if ((unsigned) cluster_dev_idx < res) {
      my_lower = cluster_dev_idx * blocks + cluster_dev_idx;
      my_upper = my_lower + blocks;
    }
    else {
      my_lower = cluster_dev_idx * blocks + res;
      my_upper = my_lower + blocks - 1;
    }

    bound[0] = 0;
    bound[1] = gridDim.x - 1;
    bound[2] = my_lower;
    bound[3] = my_upper;
    bound[4] = 0;
    bound[5] = gridDim.z - 1;
    return BLOCK_BOUND_Y;
  }
  else {
    blocks = gridDim.x / cluster_num_devs;
    res = gridDim.x % cluster_num_devs;
    if ((unsigned) cluster_dev_idx < res) {
      my_lower = cluster_dev_idx * blocks + cluster_dev_idx;
      my_upper = my_lower + blocks;
    }
    else {
      my_lower = cluster_dev_idx * blocks + res;
      my_upper = my_lower + blocks - 1;
    }

    bound[0] = my_lower;
    bound[1] = my_upper;
    bound[2] = 0;
    bound[3] = gridDim.y - 1;
    bound[4] = 0;
    bound[5] = gridDim.z - 1;
    return BLOCK_BOUND_X;
  }
}

#define NUM_ARGS_FOR_AN_EXPRESSION  15
#define CALCULATE_BLOCK_RANGE(coeff, bound_lower, bound_upper) {\
  if (coeff != 0) {                                             \
    value_lower = (coeff) * (bound_lower);                      \
    value_upper = (coeff) * (bound_upper);                      \
    if (value_lower < value_upper) {                            \
      value_min = value_lower;                                  \
      value_max = value_upper;                                  \
    }                                                           \
    else {                                                      \
      value_min = value_lower;                                  \
      value_max = value_upper;                                  \
    }                                                           \
    block_min = value_min;                                      \
    block_max = value_max;                                      \
  }                                                             \
}
#define CALCULATE_OFFSET_RANGE(coeff, bound_lower, bound_upper) {\
  if (coeff != 0) {                                              \
    value_lower = (coeff) * (bound_lower);                       \
    value_upper = (coeff) * (bound_upper);                       \
    if (value_lower < value_upper) {                             \
      value_min = value_lower;                                   \
      value_max = value_upper;                                   \
    }                                                            \
    else {                                                       \
      value_min = value_lower;                                   \
      value_max = value_upper;                                   \
    }                                                            \
    offset_min += value_min;                                     \
    offset_max += value_max;                                     \
    offset_max += (fetch_size - 1);                              \
  }                                                              \
}
bool CheckOverlappingRange(LibMAPA* libmapa, int kernel_id,
    unsigned int num_expressions, enum BlockBoundDim bound_dim,
    unsigned *block_bound, dim3 gridDim, dim3 blockDim, size_t* MAPA_args,
    const char* func_name) {
  for (unsigned int i = 0; i < num_expressions; ++i) {
    if (libmapa->MAPA_is_readonly_buffer(kernel_id, i) ||
        libmapa->MAPA_is_one_thread_expression(kernel_id, i))
      continue;

    int64_t value_lower, value_upper, value_min, value_max;
    int64_t block_min, block_max, offset_min, offset_max;
    size_t fetch_size = MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 14];
    offset_min = 0;
    offset_max = 0;

    switch (bound_dim) {
    case BLOCK_BOUND_X:
      CALCULATE_BLOCK_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 3],
          block_bound[0], block_bound[1]);
      CALCULATE_OFFSET_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 4],
          0, gridDim.y-1);
      CALCULATE_OFFSET_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 5],
          0, gridDim.z-1);
      break;
    case BLOCK_BOUND_Y:
      CALCULATE_OFFSET_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 3],
          0, gridDim.x-1);
      CALCULATE_BLOCK_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 4],
          block_bound[2], block_bound[3]);
      CALCULATE_OFFSET_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 5],
          0, gridDim.z-1);
      break;
    case BLOCK_BOUND_Z:
      CALCULATE_OFFSET_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 3],
          0, gridDim.x-1);
      CALCULATE_OFFSET_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 4],
          0, gridDim.y-1);
      CALCULATE_BLOCK_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 5],
          block_bound[4], block_bound[5]);
      break;
    }

    CALCULATE_OFFSET_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 6],
        0, blockDim.x-1);
    CALCULATE_OFFSET_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 7],
        0, blockDim.y-1);
    CALCULATE_OFFSET_RANGE(MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 8],
        0, blockDim.z-1);

    int64_t i0_bound, i0_step;
    i0_bound = MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 9];
    i0_step = MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 10];
    CALCULATE_OFFSET_RANGE(i0_step,
        0, (size_t)ceil((float)i0_bound/i0_step) - 1);

    int64_t i1_bound, i1_step;
    i1_bound = MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 11];
    i1_step = MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 12];
    CALCULATE_OFFSET_RANGE(i1_step,
        0, (size_t)ceil((float)i1_bound/i1_step) - 1);

    uint64_t offset_width = offset_max - offset_min + 1;
    uint64_t block_width = block_max - block_min;
    float overlapping_ratio =
      (float)offset_width / (offset_width + block_width);

//    printf("Overlapping ratio of expression %d is %.2f% (%lubytes)\n",
//        i, overlapping_ratio*100, offset_width);

    if (overlapping_ratio > ONE_GPU_OVERLAPPING_THRESHOLD)
      return true;
  }
  return false;
}

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
    void** args, size_t sharedMem, cudaStream_t stream) 
{
  FunctionInfo *func_info;
  const char *func_name;
  int fatbin_index;
  int n, d;
  int nNodes;
  int nDevs;
  bool oneGPU_mode = false;
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();
  std::pair<int, char*> function_indicator =
    fatbin_handler->LookupFunction((const char*)func);

  fatbin_index = function_indicator.first;
  func_name = function_indicator.second;

  if (fatbin_index < 1) {
    fprintf(stderr, "Failed to lookup function\n");
    return cudaErrorInvalidDeviceFunction;
  }

  RHAC_LOG("Launching CUDA kernel \"%s\" in fatbin %d...",
      func_name, fatbin_index);

  // get kernel info
  func_info = rhac_platform.GetKernelInfo(fatbin_index, func_name);
  if (func_info == NULL) {
    fprintf(stderr, "Failed to find function info of \"%s\" from fatbin %d\n",
        func_name, fatbin_index);
    return cudaErrorInvalidKernelImage;
  }

  // get payload_size;
  size_t payload_size = 0;
  payload_size += sizeof(int);      // fatbin index
  payload_size += sizeof(char)*128; // func_name
  payload_size += sizeof(dim3);     // gridDim
  payload_size += sizeof(dim3);     // blockDim
  payload_size += sizeof(size_t);   // sharedMem

  for (unsigned int I = 0, E = func_info->arg_sizes.size(); I != E; ++I) {
    payload_size += sizeof(char)*(func_info->arg_sizes[I]);
  }
  payload_size += 6*sizeof(unsigned);

#if defined(RHAC_PREFETCH)
  size_t* MAPA_args = NULL;
  size_t MAPA_args_size = 0;
  ClusterSVM* cluster_svm = ClusterSVM::GetClusterSVM();
  LibMAPA* libmapa = LibMAPA::GetLibMAPA();
  int kernel_id = libmapa->MAPA_get_kernel_id(func_name);
  unsigned int num_expressions =
    (kernel_id == -1) ? 0 : libmapa->MAPA_get_num_expressions(kernel_id);
  MAPA_args_size =
    sizeof(size_t) * (1 + NUM_ARGS_FOR_AN_EXPRESSION * num_expressions);
  MAPA_args = (size_t*)malloc(MAPA_args_size);

  MAPA_args[0] = num_expressions;
  for (unsigned int i = 0; i < num_expressions; ++i) {
    uint64_t buffer_base;
    size_t buffer_length;

    size_t kernel_arg_index = libmapa->MAPA_get_kernel_arg_index(kernel_id, i);
    uint64_t buffer_ptr = *((uint64_t*)args[kernel_arg_index]);
    cluster_svm->GetBaseAndLength((void*)buffer_ptr, &buffer_base, &buffer_length);
    size_t buffer_bound = buffer_base + buffer_length - buffer_ptr;

    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 1] = kernel_arg_index;
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 2] = buffer_bound;
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 3] = libmapa->MAPA_get_gx_coeff(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 4] = libmapa->MAPA_get_gy_coeff(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 5] = libmapa->MAPA_get_gz_coeff(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 6] = libmapa->MAPA_get_lx_coeff(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 7] = libmapa->MAPA_get_ly_coeff(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 8] = libmapa->MAPA_get_lz_coeff(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i + 9] = libmapa->MAPA_get_i0_bound(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i +10] = libmapa->MAPA_get_i0_step(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i +11] = libmapa->MAPA_get_i1_bound(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i +12] = libmapa->MAPA_get_i1_step(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i +13] = libmapa->MAPA_get_const(
        kernel_id, i, args, gridDim, blockDim);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i +14] = libmapa->MAPA_get_fetch_size(
        kernel_id, i);
    MAPA_args[NUM_ARGS_FOR_AN_EXPRESSION * i +15] =
      libmapa->MAPA_is_one_thread_expression(kernel_id, i);
  }
  payload_size += MAPA_args_size;

  // read-duplicate flag processing
  unsigned int num_readonly = (kernel_id == -1) ? 0 :
    libmapa->MAPA_get_num_readonly_buffers(kernel_id);
  std::vector<uint32_t> readonly_bufs(num_readonly);

  if (kernel_id == -1) {
#ifdef READDUP_FLAG_CACHING
    // we do not know anything about this kernel
    // remove all read-duplicate flags in the cache list
    for (std::vector<uint64_t>::iterator I = rhac_platform.readdup_begin(),
        E = rhac_platform.readdup_end(); I != E; ++I) {
      uint64_t vaddr = *I;
      uint64_t base = (uint64_t)cluster_svm->GetBase((void*)vaddr);
      size_t buf_length = cluster_svm->GetLength((void*)base);

      RHACCommandNSetDupFlag *m_cmd;
      for (int n = 0; n < rhac_platform.GetClusterNumNodes(); n++) {
        m_cmd = new RHACCommandNSetDupFlag();
        m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                              NSetDupFlag,
                              n, -1);
        m_cmd->SetBase(base);
        m_cmd->SetLength(buf_length);
        m_cmd->SetFlag(0);
        rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);
      }
    }
    rhac_platform.ClearReadDupList();
#endif
  }
  else {
    libmapa->MAPA_get_readonly_buffers(kernel_id, &readonly_bufs[0]);

#ifdef READDUP_FLAG_CACHING
    unsigned int num_non_readonly =
      libmapa->MAPA_get_num_non_readonly_buffers(kernel_id);
    std::vector<uint32_t> non_readonly_bufs(num_non_readonly);
    libmapa->MAPA_get_non_readonly_buffers(kernel_id, &non_readonly_bufs[0]);

    // process non read-only buffers when read-dup flag is set in previous kernels
    for (std::vector<uint32_t>::iterator I = non_readonly_bufs.begin(),
        E = non_readonly_bufs.end(); I != E; ++I) {
      uint64_t vaddr = *((uint64_t*)args[*I]);
      uint64_t base = (uint64_t)cluster_svm->GetBase((void*)vaddr);

      if (rhac_platform.IsInReadDupList(base)) {
        rhac_platform.RemoveFromReadDupList(base);

        size_t buf_length = cluster_svm->GetLength((void*)base);

        RHACCommandNSetDupFlag *m_cmd;
        for (int n = 0; n < rhac_platform.GetClusterNumNodes(); n++) {
          m_cmd = new RHACCommandNSetDupFlag();
          m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                                NSetDupFlag,
                                n, -1);
          m_cmd->SetBase(base);
          m_cmd->SetLength(buf_length);
          m_cmd->SetFlag(0);
          rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);
        }
      }
    }
#endif

    // process read-only buffers when read-dup flag is not set in previous kernels
    for (std::vector<uint32_t>::iterator I = readonly_bufs.begin(),
        E = readonly_bufs.end(); I != E; ++I) {
      uint64_t vaddr = *((uint64_t*)args[*I]);
      uint64_t base = (uint64_t)cluster_svm->GetBase((void*)vaddr);

#ifdef READDUP_FLAG_CACHING
      if (!rhac_platform.IsInReadDupList(base)) {
        rhac_platform.AddToReadDupList(base);
#endif
        size_t buf_length = cluster_svm->GetLength((void*)base);

        RHACCommandNSetDupFlag *m_cmd;
        for (int n = 0; n < rhac_platform.GetClusterNumNodes(); n++) {
          m_cmd = new RHACCommandNSetDupFlag();
          m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                                NSetDupFlag,
                                n, -1);
          m_cmd->SetBase(base);
          m_cmd->SetLength(buf_length);
          m_cmd->SetFlag(1);
          rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);
        }
#ifdef READDUP_FLAG_CACHING
      }
#endif
    }
  }

  // execute kernel when flag changing is finished
  rhac_platform.RhacBarrierEnqueue();
#endif

  nNodes = rhac_platform.GetClusterNumNodes();
  if (func_info->has_global_atomics == 1) {
    // when a function has global atomics we run this kernel using only one GPU
    nNodes = 1;
    oneGPU_mode = true;
    RHAC_LOG("Launching kernel \"%s\" using only one GPU", func_name);
  }

  // generate and enqueue commands
  for (n = 0; n < nNodes; n++) {
    nDevs = rhac_platform.GetNumDevicesIn(n);

    if (oneGPU_mode == true)
      nDevs = 1;

    for (d = 0; d < nDevs; d++) {
      unsigned block_bound[6];
      RHACCommandDKernelPartialExecution *cmd;
      cmd = new RHACCommandDKernelPartialExecution();

      cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                          DKernelPartialExecution,
                          n, d); 
      cmd->AllocPayload(payload_size);

      // Write payload
      cmd->SetFatbinIndex(fatbin_index);
      cmd->SetFuncName(func_name);
      cmd->SetGridDim(gridDim);
      cmd->SetBlockDim(blockDim);
      cmd->SetSharedMem(sharedMem);
      RHAC_LOG("send sharedMem : %zu", sharedMem);

      for (unsigned int I = 0, E = func_info->arg_sizes.size(); I != E; ++I) {
        int arg_size = func_info->arg_sizes[I];
        cmd->PushArg((char*)args[I], (size_t)arg_size);
      }

      enum BlockBoundDim bound_dim = GetBlockBound(block_bound, gridDim, n, d);
      RHAC_LOG("block bound : %u ~ %u (%u ~ %u), %u ~ %u (%u ~ %u), %u ~ %u (%u ~ %u)",
          block_bound[0], block_bound[1], 0, gridDim.x-1,
          block_bound[2], block_bound[3], 0, gridDim.y-1,
          block_bound[4], block_bound[5], 0, gridDim.z-1);

      // check overlapping access range of expressions
      // if percentage of overlapping range exceeds the threshold
      // run this kernel using only one GPU
      if (oneGPU_mode == false &&
          n == 0 && d ==0) {
        oneGPU_mode = CheckOverlappingRange(libmapa, kernel_id, num_expressions,
            bound_dim, block_bound, gridDim, blockDim, MAPA_args,
            func_name);
        if (oneGPU_mode == true) {
          // reset block bound
          switch (bound_dim) {
          case BLOCK_BOUND_X:
            block_bound[0] = 0; block_bound[1] = gridDim.x - 1;
            break;
          case BLOCK_BOUND_Y:
            block_bound[2] = 0; block_bound[3] = gridDim.y - 1;
            break;
          case BLOCK_BOUND_Z:
            block_bound[4] = 0; block_bound[5] = gridDim.z - 1;
            break;
          }
          RHAC_LOG("Launching kernel \"%s\" using only one GPU", func_name);
        }
      }

      for (unsigned int I = 0; I < 6; ++I) {
        cmd->PushArg((char *)&block_bound[I], sizeof(unsigned));
      }

#if defined(RHAC_PREFETCH)
      if (MAPA_args_size != 0)
        cmd->PushArg((char*)MAPA_args, MAPA_args_size);
#endif

      rhac_platform.EnqueueCommand((RHACCommand *)cmd);

      if (oneGPU_mode == true) {
        nNodes = 1;
        nDevs = 1;
        n = 1;
        d = 1;
      }
    }
  }

  rhac_platform.RhacBarrierEnqueue();

  // generate and enqueue commands for NSVMSync
  // Send NSVM Sync
  for (n = 0; n < nNodes; n++) {
    RHACCommand *cmd;
    cmd = new RHACCommand();
    cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                        NSVMSync,
                        n, -1);
    RHAC_LOG("NSVMSync to Rank %d with id %lu", n, cmd->GetCommandID());

    rhac_platform.EnqueueCommand(cmd);
  }

  rhac_platform.RhacBarrierEnqueue();

#if defined(RHAC_PREFETCH) && !defined(READDUP_FLAG_CACHING)
  {
    for (std::vector<uint32_t>::iterator I = readonly_bufs.begin(),
        E = readonly_bufs.end(); I != E; ++I) {
      uint64_t vaddr = *((uint64_t*)args[*I]);
      uint64_t base = (uint64_t)cluster_svm->GetBase((void*)vaddr);

      size_t buf_length = cluster_svm->GetLength((void*)base);

      RHACCommandNSetDupFlag *m_cmd;
      for (int n = 0; n < rhac_platform.GetClusterNumNodes(); n++) {
        m_cmd = new RHACCommandNSetDupFlag();
        m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                              NSetDupFlag,
                              n, -1);
        m_cmd->SetBase(base);
        m_cmd->SetLength(buf_length);
        m_cmd->SetFlag(0);
        rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);
      }
    }
  }
#endif

#if defined(RHAC_PREFETCH)
  if (MAPA_args_size != 0)
    free(MAPA_args);
#endif

  return cudaSuccess;
}

//=============================================================================
// CUDA Runtime API
// 5.9. Memory Management
//=============================================================================
cudaError_t cudaFree(void* devPtr) 
{
  assert(rhac_platform.IsHost());
  ClusterSVM* cluster_svm = ClusterSVM::GetClusterSVM();
  
  cluster_svm->FreeClusterSVM(devPtr);

  RHAC_LOG("RHAC HOST free SVM Ptr : %p", devPtr);

  return cudaSuccess;
}

cudaError_t cudaMalloc(void** devPtr, size_t size) 
{
  assert(rhac_platform.IsHost());

  ClusterSVM* cluster_svm = ClusterSVM::GetClusterSVM();
  *devPtr = cluster_svm->MallocClusterSVM(size);

  // FIXME
  RHAC_LOG("RHAC HOST new SVM(Device) Ptr : %p", *devPtr);

  RHACCommandNSplitVARange *m_cmd;
  for (int n = 0; n < rhac_platform.GetClusterNumNodes(); n++) {
    m_cmd = new RHACCommandNSplitVARange();
    m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                          NSplitVARange,
                          n, -1);
    m_cmd->SetBase((uint64_t)*devPtr);
    m_cmd->SetLength(size);
    rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);
  }

  return cudaSuccess;
}

cudaError_t cudaMallocHost(void** ptr, size_t size) 
{
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  return libcuda->cudaMallocHost(ptr, size);
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) 
{
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  return libcuda->cudaHostAlloc(pHost, size, flags);
}

cudaError_t cudaFreeHost(void* ptr) 
{
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  return libcuda->cudaFreeHost(ptr);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
    cudaMemcpyKind kind) 
{
  RHAC_LOG("Hook %s function", __func__);

  //FIXME
  rhac_platform.FinishAllRequestQueue();

  RHACCommandNSVMMemcpy *m_cmd;
  enum CommandKind cmd_kind;
  m_cmd = new RHACCommandNSVMMemcpy();

  switch (kind) {
    case cudaMemcpyHostToDevice:
      cmd_kind = NSVMMemcpyHostToDevice;
      break;
    case cudaMemcpyDeviceToHost:
      cmd_kind = NSVMMemcpyDeviceToHost;
      break;
    case cudaMemcpyHostToHost:
      cmd_kind = NSVMMemcpyHostToHost;
      break;
    case cudaMemcpyDeviceToDevice:
      cmd_kind = NSVMMemcpyDeviceToDevice;
      break;
    default:
      assert(0);
      break;
  }

  m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(), cmd_kind, 0, -1);
  m_cmd->SetDestination((uint64_t)dst);
  m_cmd->SetSource((uint64_t)src);
  m_cmd->SetSize(count);

  rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);

  //rhac_platform.FinishRequestQueue(0);
  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
    cudaMemcpyKind kind, cudaStream_t stream) 
{
  // FIXME - stream;

  RHACCommandNSVMMemcpy *m_cmd;
  enum CommandKind cmd_kind;
  m_cmd = new RHACCommandNSVMMemcpy();

  switch (kind) {
    case cudaMemcpyHostToDevice:
      cmd_kind = NSVMMemcpyAsyncHostToDevice;
      break;
    case cudaMemcpyDeviceToHost:
      cmd_kind = NSVMMemcpyAsyncDeviceToHost;
      break;
    case cudaMemcpyHostToHost:
    case cudaMemcpyDeviceToDevice:
    default:
      assert(0);
      break;
  }

  m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(), cmd_kind, 0, -1);
  m_cmd->SetDestination((uint64_t)dst);
  m_cmd->SetSource((uint64_t)src);
  m_cmd->SetSize(count);

  rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);

  return cudaSuccess;
}

cudaError_t cudaMemset(void* dst, int value, size_t sizeBytes) 
{
  RHACCommandNSVMMemset *m_cmd;
  m_cmd = new RHACCommandNSVMMemset();

  m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(), NSVMMemset, 0, -1);
  m_cmd->SetDestination((uint64_t)dst);
  m_cmd->SetValue(value);
  m_cmd->SetSize(sizeBytes);

  rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);

  rhac_platform.FinishRequestQueue(0);

  return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void* dst, int  value, size_t sizeBytes,
    cudaStream_t stream) 
{
  //FIXME - stream
  RHACCommandNSVMMemset *m_cmd;
  m_cmd = new RHACCommandNSVMMemset();

  m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(), NSVMMemsetAsync, 0, -1);
  m_cmd->SetDestination((uint64_t)dst);
  m_cmd->SetValue(value);
  m_cmd->SetSize(sizeBytes);

  rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);

  //rhac_platform.FinishRequestQueue(0);

  return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src,
    size_t count, size_t offset, cudaMemcpyKind kind) 
{
  int fatbin_index;
  int n, d;
  const char *var_name;
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();
  std::pair<int, char*> var_indicator =
    fatbin_handler->LookupVar((char*)symbol);
  std::vector<rhac_command_id_t> wait_list;

  fatbin_index = var_indicator.first;
  var_name = var_indicator.second;

  RHAC_LOG("send fatbin idx : %d, symbol : %s, src : %p, count : %zu, offset : %zu",
      fatbin_index, var_name, src, count, offset);

  // send command for host devices
  RHACCommandADMemcpyToSymbol *m_cmd;
  m_cmd = new RHACCommandADMemcpyToSymbol();
  m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                        ADMemcpyToSymbol,
                        0, rhac_platform.GetNumDevicesIn(0)); // device # = ref cnt
  m_cmd->SetFatbinIndex(fatbin_index);
  m_cmd->SetSymbolName(var_name);
  m_cmd->SetMemcpyKind(kind);
  m_cmd->SetOffset(offset);
  m_cmd->SetSourceAndCount((const char *)src, count);
  m_cmd->SetReferenceCount(rhac_platform.GetNumDevicesIn(0)); // FIXME

  for (d = 0; d < rhac_platform.GetNumDevicesIn(0); d++) {
    rhac_platform.EnqueueCommand((RHACCommand *)m_cmd, 0, d);
  }

  wait_list.empty();
  wait_list.push_back(m_cmd->GetCommandID());


  for (n = 1; n < rhac_platform.GetClusterNumNodes(); n++) {
    m_cmd = new RHACCommandADMemcpyToSymbol();
    m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                          ADMemcpyToSymbol,
                          n, 0);

    m_cmd->SetFatbinIndex(fatbin_index);
    m_cmd->SetSymbolName(var_name);
    m_cmd->SetMemcpyKind(kind);
    m_cmd->SetOffset(offset);
    m_cmd->SetSourceAndCount((const char*)src, count);

    rhac_platform.EnqueueCommand((RHACCommand *)m_cmd, n, 0);

    wait_list.push_back(m_cmd->GetCommandID());
  }

  // wait until the job is finished
  for (n = 0; n < rhac_platform.GetClusterNumNodes(); n++) {
    for (d = 0; d < rhac_platform.GetNumDevicesIn(n); d++) {
      rhac_platform.WaitResponse(wait_list[n], n, d);
    }
  }

  return cudaSuccess;
}

cudaError_t cudaMallocArray(cudaArray_t* array,
    const cudaChannelFormatDesc* desc, size_t width, size_t height,
    unsigned int flags) {
  unsigned int array_index = rhac_platform.GetNextArrayIndex();
  *array = (cudaArray_t)(uint64_t)array_index;

  for (int n = 0; n < rhac_platform.GetClusterNumNodes(); ++n) {
    for (int d = 0; d < rhac_platform.GetNumDevicesIn(n); ++d) {
      RHACCommandDMallocArray *cmd;
      cmd = new RHACCommandDMallocArray();
      cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
          DMallocArray, n, d);
      cmd->SetDescFormat(desc->f);
      cmd->SetDescW(desc->w);
      cmd->SetDescX(desc->x);
      cmd->SetDescY(desc->y);
      cmd->SetDescZ(desc->z);
      cmd->SetWidth(width);
      cmd->SetHeight(height);
      cmd->SetFlags(flags);

      rhac_platform.EnqueueCommand((RHACCommand *)cmd);
    }
  }

  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

cudaError_t cudaFreeArray(cudaArray_t array) {
  unsigned int array_index = (unsigned int)((uint64_t)array);

  for (int n = 0; n < rhac_platform.GetClusterNumNodes(); ++n) {
    for (int d = 0; d < rhac_platform.GetNumDevicesIn(n); ++d) {
      RHACCommandDFreeArray *cmd;
      cmd = new RHACCommandDFreeArray();
      cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
          DFreeArray, n, d);
      cmd->SetArrayIndex(array_index);

      rhac_platform.EnqueueCommand((RHACCommand *)cmd);
    }
  }

  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total) 
{
  // FIXME
  // TODO();
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  return libcuda->cudaMemGetInfo(free, total);
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset,
    size_t hOffset, const void* src, size_t spitch, size_t width,
    size_t height, cudaMemcpyKind kind) {
  unsigned int array_index = (unsigned int)((uint64_t)dst);
  int n, d;
  std::vector<rhac_command_id_t> wait_list;

  assert(spitch * height <= 4L*1024L*GB);

  // send command for host devices
  RHACCommandADMemcpy2DToArray *m_cmd;
  m_cmd = new RHACCommandADMemcpy2DToArray();
  m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                        ADMemcpy2DToArray,
                        0, rhac_platform.GetNumDevicesIn(0)); // device # = ref cnt
  m_cmd->SetArrayIndex(array_index);
  m_cmd->SetWOffset(wOffset);
  m_cmd->SetHOffset(hOffset);
  m_cmd->SetSourceAndCount(src, spitch*height);
  m_cmd->SetSPitch(spitch);
  m_cmd->SetWidth(width);
  m_cmd->SetMemcpyKind(kind);
  m_cmd->SetReferenceCount(rhac_platform.GetNumDevicesIn(0));

  for (d = 0; d < rhac_platform.GetNumDevicesIn(0); d++) {
    rhac_platform.EnqueueCommand((RHACCommand *)m_cmd, 0, d);
  }

  wait_list.empty();
  wait_list.push_back(m_cmd->GetCommandID());

  for (n = 1; n < rhac_platform.GetClusterNumNodes(); n++) {
    m_cmd = new RHACCommandADMemcpy2DToArray();
    m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                          ADMemcpy2DToArray,
                          n, 0);
    m_cmd->SetArrayIndex(array_index);
    m_cmd->SetWOffset(wOffset);
    m_cmd->SetHOffset(hOffset);
    m_cmd->SetSourceAndCount(src, spitch*height);
    m_cmd->SetSPitch(spitch);
    m_cmd->SetWidth(width);
    m_cmd->SetMemcpyKind(kind);
    
    rhac_platform.EnqueueCommand((RHACCommand *)m_cmd, n, 0);

    wait_list.push_back(m_cmd->GetCommandID());
  }

  // wait until the job is finished
  for (n = 0; n < rhac_platform.GetClusterNumNodes(); n++) {
    for (d = 0; d < rhac_platform.GetNumDevicesIn(n); d++) {
      rhac_platform.WaitResponse(wait_list[n], n, d);
    }
  }

  return cudaSuccess;
}

cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) {
  rhac_platform.FinishAllRequestQueue();

  if (p->srcArray != 0 || p->dstArray != 0) {
    fprintf(stderr, "CUDA array is passed to cudaMemcpy3D call\n");
    return cudaErrorInvalidValue;
  }

  printf("\n[NYI] cudaMemcpy3D\n\n\n");

//  rhac_platform.EnqueueCommand((RHACCommand*)m_cmd);
  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) {
  printf("\n[NYI] cudaMemcpy3DAsync\n\n\n");

//  rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);

  return cudaSuccess;
}

cudaError_t cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device)
{
  assert(advice == cudaMemAdviseSetReadMostly ||
      advice == cudaMemAdviseUnsetReadMostly);

  RHAC_LOG("Send Rank %d NMemAdvise devPtr : %p, count %zu, advise : %d, device : %d",
      rhac_platform.GetRank(),
      devPtr, count, advice, device);


  RHACCommandNMemAdvise *m_cmd;

  for (int n = 0; n < rhac_platform.GetClusterNumNodes(); n++) {
    m_cmd = new RHACCommandNMemAdvise();
    m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
                          NMemAdvise,
                          n, -1);
    m_cmd->SetDevPtr((uint64_t)devPtr);
    m_cmd->SetCount(count);
    m_cmd->SetAdvice(advice);
    m_cmd->SetDevice(device);


    rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);
  }

  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

//=============================================================================
// CUDA Runtime API
// 5.24. Texture Reference Management
//=============================================================================
cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
    cudaChannelFormatKind f) 
{
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  return libcuda->cudaCreateChannelDesc(x, y, z, w, f);
}

cudaError_t cudaBindTexture(size_t* offset, const textureReference* texref,
    const void* devPtr, const cudaChannelFormatDesc* desc, size_t size) {
  int fatbin_index;
  const char* ref_name;
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();
  std::pair<int, char*> ref_indicator =
    fatbin_handler->LookupTexture(texref);

  fatbin_index = ref_indicator.first;
  ref_name = ref_indicator.second;

  // call for every devices
  for (int n = 0; n < rhac_platform.GetClusterNumNodes(); ++n) {
    for (int d = 0; d < rhac_platform.GetNumDevicesIn(n); ++d) {
      RHACCommandDBindTexture *m_cmd;
      m_cmd = new RHACCommandDBindTexture();
      m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
          DBindTexture, n, d);
      m_cmd->SetFatbinIndex(fatbin_index);
      m_cmd->SetRefName(ref_name);
      m_cmd->SetRefFilterMode(texref->filterMode);
      m_cmd->SetRefNormalized(texref->normalized);
      m_cmd->SetDevPtr((void*)devPtr);

      if (desc) {
        m_cmd->SetDescX(desc->x);
        m_cmd->SetDescY(desc->y);
        m_cmd->SetDescZ(desc->z);
        m_cmd->SetDescW(desc->w);
        m_cmd->SetDescFormat(desc->f);
      }
      else {
        m_cmd->SetDescX(texref->channelDesc.x);
        m_cmd->SetDescY(texref->channelDesc.y);
        m_cmd->SetDescZ(texref->channelDesc.z);
        m_cmd->SetDescW(texref->channelDesc.w);
        m_cmd->SetDescFormat(texref->channelDesc.f);
      }
      m_cmd->SetSize(size);

      rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);
    }
  }
  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc,
    cudaArray_const_t array) {
  assert(rhac_platform.GetNumDevicesIn(0));

  cudaArray_t cuda_array;
  unsigned int array_index = (unsigned int)((uint64_t)array);

  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  DeviceExecutor *executor0 = rhac_platform.device_executors_[0];

  cuda_array = executor0->GetCUDAArray(array_index);
  return libcuda->cudaGetChannelDesc(desc, cuda_array);
}

cudaError_t cudaBindTextureToArray(const textureReference* texref,
    cudaArray_const_t array, const cudaChannelFormatDesc* desc) {
  int fatbin_index;
  const char* ref_name;
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();
  std::pair<int, char*> ref_indicator =
    fatbin_handler->LookupTexture(texref);

  fatbin_index = ref_indicator.first;
  ref_name = ref_indicator.second;

  for (int n = 0; n < rhac_platform.GetClusterNumNodes(); ++n) {
    for (int d = 0; d < rhac_platform.GetNumDevicesIn(n); ++d) {
      RHACCommandDBindTextureToArray *m_cmd;
      m_cmd = new RHACCommandDBindTextureToArray();
      m_cmd->SetDefaultInfo(rhac_platform.GenerateCommandID(),
          DBindTextureToArray, n, d);
      m_cmd->SetFatbinIndex(fatbin_index);
      m_cmd->SetRefName(ref_name);
      m_cmd->SetRefFilterMode(texref->filterMode);
      m_cmd->SetRefNormalized(texref->normalized);
      m_cmd->SetArrayIndex((unsigned int)((uint64_t)array));

      if (desc) {
        m_cmd->SetDescX(desc->x);
        m_cmd->SetDescY(desc->y);
        m_cmd->SetDescZ(desc->z);
        m_cmd->SetDescW(desc->w);
        m_cmd->SetDescFormat(desc->f);
      }
      else {
        m_cmd->SetDescX(texref->channelDesc.x);
        m_cmd->SetDescY(texref->channelDesc.y);
        m_cmd->SetDescZ(texref->channelDesc.z);
        m_cmd->SetDescW(texref->channelDesc.w);
        m_cmd->SetDescFormat(texref->channelDesc.f);
      }

      rhac_platform.EnqueueCommand((RHACCommand *)m_cmd);
    }
  }
  rhac_platform.FinishAllRequestQueue();

  return cudaSuccess;
}

cudaError_t cudaUnbindTexture(const textureReference* texref) {
  return cudaSuccess;
}
