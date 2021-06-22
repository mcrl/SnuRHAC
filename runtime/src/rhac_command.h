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

#ifndef __RHAC_COMMAND_H__
#define __RHAC_COMMAND_H__

#include "rhac.h"

#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define COMMAND_HEADER_SIZE 64

#define OFFSET_COMMAND_ID                 0
#define OFFSET_COMMAND_KIND               8
#define OFFSET_COMMAND_NODE               12
#define OFFSET_COMMAND_DEVICE             16
#define OFFSET_COMMAND_PAYLOAD_SIZE       20
#define OFFSET_COMMAND_HEADER_END         24
#define OFFSET_COMMAND_REFERENCE_COUNTER  OFFSET_COMMAND_DEVICE

#define DEVICE_COMMAND_START          100
#define ALL_COMMAND_START             200 // for all executors (node + device) - GlobalBarrier, GlobalBarrierEnd
#define ALL_DEVICE_COMMAND_START      300 // for all deivce executors - MemcpyToSymbol, MemcpyToArray

enum CommandKind {
  NExit = 0,
  NSVMMemcpyHostToDevice, // FIXME
  NSVMMemcpyDeviceToHost, // FIXME
  NSVMMemcpyHostToHost,
  NSVMMemcpyDeviceToDevice,
  NSVMMemcpyAsyncHostToDevice, // FIXME
  NSVMMemcpyAsyncDeviceToHost, // FIXME
  NSVMMemset,
  NSVMMemsetAsync,
  NSVMReserve,
  NSVMSync,
  NBarrier,
  NEventCreate,
  NEventRecord,
  NMemAdvise,
  NSplitVARange,
  NSetDupFlag,
  DExit = DEVICE_COMMAND_START,
  DSetDevice,
  DSetCacheConfig,
  DFuncSetCacheConfig,
  DMallocArray,
  DFreeArray,
  DBindTexture,
  DBindTextureToArray,
  DReset,
  DKernelPartialExecution,
  DEvent,
  AGlobalBarrier = ALL_COMMAND_START,
  AGlobalBarrierEnd,
  ADMemcpyToSymbol = ALL_DEVICE_COMMAND_START,
  ADMemcpy2DToArray,
  // FIXME - prefix (internally used command between transmitter and receiver)
  GlobalBarrierEnd,  // FIXME
};

class RHACCommand {
  public:
    RHACCommand();
    ~RHACCommand();

    void SetDefaultInfo(rhac_command_id_t command_id,
                        enum CommandKind command_kind,
                        int node, int device);
    bool WriteHeader(const char *src, const size_t size, const size_t offset);
    void ReadHeader(char *dst, const size_t size, const size_t offset);
    bool WritePayload(const char *src, const size_t size, const size_t offset);
    void ReadPayload(char *dst, const size_t size, const size_t offset);
    void *ReadPayloadPtr(const size_t size, const size_t offset);

    rhac_command_id_t GetCommandID();
    enum CommandKind GetCommandKind();
    int GetTargetNode();
    int GetTargetDevice();
    char* GetHeaderPtr();
    uint32_t GetPayloadSize();
    char* GetPayloadPtr();

    void SetPayloadSize(uint32_t size);
    void SetPayloadPtr(const char *src, const uint32_t size);

    void AllocPayload(size_t size);
    bool IsNodeCommand();
    bool IsDeviceCommand();
    bool IsAllCommand();
    bool IsAllDeviceCommand();
    bool HasPayload();

    void PrintInfo(const char *prefix);

  protected:
    // <Header data format>         | <Offset>
    // uint64_t cmd_id;             | 0
    // enum cmd_kind                | 8
    // int node                     | 12
    // int device (cluster index)   | 16
    // uint32_t payload size        | 20 
    char header_[COMMAND_HEADER_SIZE];
    char *payload_;
    size_t payload_varg_head_offset_;
};

class RHACCommandVarg : public RHACCommand {
  public:
    RHACCommandVarg();
    ~RHACCommandVarg();
    
    void PushArg(const char *src, const size_t size);
    void* PopArgPtr(const size_t size);
    template <typename T> T PopArg() {
      T ret;
      ReadPayload((char*)&ret, sizeof(T), payload_varg_head_offset_);
      payload_varg_head_offset_ += sizeof(T);
      return ret;
    }
    // Caution All children of RHACCommandVarg must have ResetVargHead !!
};

class RHACCommandNSVMReserve : public RHACCommand {
  public:
    RHACCommandNSVMReserve();
    ~RHACCommandNSVMReserve();

    void SetReserveSize(size_t size);
    size_t GetReserveSize();
};

class RHACCommandNSVMMemcpy : public RHACCommand {
  public:
    RHACCommandNSVMMemcpy();
    ~RHACCommandNSVMMemcpy();

    void SetDestination(uint64_t dst);
    uint64_t GetDestination();
    void SetSource(uint64_t src);
    uint64_t GetSource();
    void SetSize(size_t size);
    size_t GetSize();
};

class RHACCommandNSVMMemset : public RHACCommand {
  public :
    RHACCommandNSVMMemset();
    ~RHACCommandNSVMMemset();

    void SetDestination(uint64_t dst);
    uint64_t GetDestination();
    void SetValue(int value);
    int GetValue();
    void SetSize(size_t size);
    size_t GetSize();
};

class RHACCommandNBarrier : public RHACCommandVarg {
  public:
    RHACCommandNBarrier();
    ~RHACCommandNBarrier();

    void SetNumWait(uint32_t num_wait);
    uint32_t GetNumWait();
    void PushWaitItem(int node, int dev, rhac_command_id_t id);
    void PopWaitItem(int *node, int *dev, rhac_command_id_t *id);
    // Caution All children of RHACCommandVarg must have ResetVargHead !!
    void ResetVargHead();
};

class RHACCommandNMemAdvise : public RHACCommand {
  public:
    RHACCommandNMemAdvise();
    ~RHACCommandNMemAdvise();

    void SetDevPtr(uint64_t devPtr);
    uint64_t GetDevPtr();
    void SetCount(size_t count);
    size_t GetCount();
    void SetAdvice(cudaMemoryAdvise advice);
    cudaMemoryAdvise GetAdvice();
    void SetDevice(int device);
    int GetDevice();
};

class RHACCommandNSplitVARange : public RHACCommand {
  public:
    RHACCommandNSplitVARange();
    ~RHACCommandNSplitVARange();

    void SetBase(uint64_t base);
    uint64_t GetBase();
    void SetLength(size_t length);
    size_t GetLength();
};

class RHACCommandNSetDupFlag : public RHACCommand {
  public:
    RHACCommandNSetDupFlag();
    ~RHACCommandNSetDupFlag();

    void SetBase(uint64_t base);
    uint64_t GetBase();
    void SetLength(size_t length);
    size_t GetLength();
    void SetFlag(bool turnon);
    bool GetFlag();
};

class RHACCommandNEventCreate : public RHACCommand {
  public:
    RHACCommandNEventCreate();
    ~RHACCommandNEventCreate();
    void SetEventPtr(cudaEvent_t *event);
    cudaEvent_t* GetEventPtr();
};

class RHACCommandNEventRecord : public RHACCommand {
  public:
    RHACCommandNEventRecord();
    ~RHACCommandNEventRecord();
    void SetEvent(cudaEvent_t event);
    cudaEvent_t GetEvent();
    void SetStream(cudaStream_t stream);
    cudaStream_t GetStream();
};

// =================================================================
// Device Commands

class RHACCommandDKernelPartialExecution : public RHACCommandVarg {

  public:
    RHACCommandDKernelPartialExecution();
    ~RHACCommandDKernelPartialExecution();

    void SetFatbinIndex(int fatbin_index);
    int GetFatbinIndex();
    void SetFuncName(const char *func_name);
    void GetFuncName(char *func_name);
    void SetGridDim(dim3 gridDim);
    dim3 GetGridDim();
    void SetBlockDim(dim3 blockDim);
    dim3 GetBlockDim();
    void SetSharedMem(size_t sharedMem);
    size_t GetSharedMem();
    // Caution All children of RHACCommandVarg must have ResetVargHead !!
    void ResetVargHead();
};

class RHACCommandDSetCacheConfig : public RHACCommand {
  public:
    RHACCommandDSetCacheConfig();
    ~RHACCommandDSetCacheConfig();

    void SetCacheConfig(cudaFuncCache cache_config);
    cudaFuncCache GetCacheConfig();
};

class RHACCommandDFuncSetCacheConfig : public RHACCommand {
  public:
    RHACCommandDFuncSetCacheConfig();
    ~RHACCommandDFuncSetCacheConfig();

    void SetFatbinIndex(int fatbin_index);
    int GetFatbinIndex();
    void SetFuncName(const char *func_name);
    void GetFuncName(char *func_name);
    void SetCacheConfig(cudaFuncCache cache_config);
    cudaFuncCache GetCacheConfig();
};

class RHACCommandDMallocArray : public RHACCommand {
  public:
    RHACCommandDMallocArray();
    ~RHACCommandDMallocArray();

    void SetDescFormat(cudaChannelFormatKind format);
    cudaChannelFormatKind GetDescFormat();
    void SetDescW(int w);
    int GetDescW();
    void SetDescX(int x);
    int GetDescX();
    void SetDescY(int y);
    int GetDescY();
    void SetDescZ(int z);
    int GetDescZ();
    void SetWidth(size_t width);
    size_t GetWidth();
    void SetHeight(size_t height);
    size_t GetHeight();
    void SetFlags(unsigned int flags);
    unsigned int GetFlags();
};

class RHACCommandDFreeArray : public RHACCommand {
  public:
    RHACCommandDFreeArray();
    ~RHACCommandDFreeArray();

    void SetArrayIndex(unsigned int array_index);
    unsigned int GetArrayIndex();
};

class RHACCommandDBindTexture : public RHACCommand {
  public:
    RHACCommandDBindTexture();
    ~RHACCommandDBindTexture();

    void SetFatbinIndex(int fatbin_index);
    int GetFatbinIndex();
    void SetRefName(const char* ref_name);
    void GetRefName(char* ref_name);
    void SetRefFilterMode(int mode);
    int GetRefFilterMode();
    void SetRefNormalized(int flag);
    int GetRefNormalized();
    void SetDevPtr(void *ptr);
    void* GetDevPtr();
    void SetDescX(int x);
    int GetDescX();
    void SetDescY(int y);
    int GetDescY();
    void SetDescZ(int z);
    int GetDescZ();
    void SetDescW(int w);
    int GetDescW();
    void SetDescFormat(cudaChannelFormatKind kind);
    cudaChannelFormatKind GetDescFormat();
    void SetSize(size_t size);
    size_t GetSize();
};

class RHACCommandDBindTextureToArray : public RHACCommand {
  public:
    RHACCommandDBindTextureToArray();
    ~RHACCommandDBindTextureToArray();

    void SetFatbinIndex(int fatbin_index);
    int GetFatbinIndex();
    void SetRefName(const char* ref_name);
    void GetRefName(char* ref_name);
    void SetRefFilterMode(int mode);
    int GetRefFilterMode();
    void SetRefNormalized(int flag);
    int GetRefNormalized();
    void SetArrayIndex(unsigned int array_index);
    unsigned int GetArrayIndex();
    void SetDescX(int x);
    int GetDescX();
    void SetDescY(int y);
    int GetDescY();
    void SetDescZ(int z);
    int GetDescZ();
    void SetDescW(int w);
    int GetDescW();
    void SetDescFormat(cudaChannelFormatKind kind);
    cudaChannelFormatKind GetDescFormat();
};

// =================================================================
// Commands for all executors 
class RHACCommandAll : public RHACCommand {
  public:
    RHACCommandAll();
    ~RHACCommandAll();

    void SetReferenceCount(int cnt);
    int DecreaseReferenceCount();
    bool CheckReferenceCount();
    int GetReferenceCount();
};


// =================================================================
// Commands for all device executors
class RHACCommandAllDevice : public RHACCommandAll {
  public:
    RHACCommandAllDevice();
    ~RHACCommandAllDevice();
};

class RHACCommandADMemcpyToSymbol : public RHACCommandAllDevice {
  public:
    RHACCommandADMemcpyToSymbol();
    ~RHACCommandADMemcpyToSymbol();

    void SetFatbinIndex(int fatbin_index);
    int GetFatbinIndex();
    void SetSymbolName(const char *symbol);
    void GetSymbolName(char *symbol);
    void SetMemcpyKind(cudaMemcpyKind kind);
    cudaMemcpyKind GetMemcpyKind();
    void SetOffset(uint64_t offset);
    uint64_t GetOffset();
    void SetSourceAndCount(const char *src, uint32_t size);
    char *GetSource();
    uint32_t GetCount(); 
};

class RHACCommandADMemcpy2DToArray : public RHACCommandAllDevice {
  public:
    RHACCommandADMemcpy2DToArray();
    ~RHACCommandADMemcpy2DToArray();

    void SetArrayIndex(unsigned int array_index);
    unsigned int GetArrayIndex();
    void SetWOffset(size_t wOffset);
    size_t GetWOffset();
    void SetHOffset(size_t hOffset);
    size_t GetHOffset();
    void SetSourceAndCount(const void* src, uint32_t size);
    void* GetSource();
    void SetSPitch(size_t spitch);
    size_t GetSPitch();
    void SetWidth(size_t width);
    size_t GetWidth();
    size_t GetHeight();
    void SetMemcpyKind(cudaMemcpyKind kind);
    cudaMemcpyKind GetMemcpyKind();
};

#endif // __RHAC_COMMAND_H__
