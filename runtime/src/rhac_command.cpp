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

#include "rhac_command.h"
#include "rhac.h"
#include "rhac_response.h"
#include "utils.h"
#include "platform.h"

#include <cstring>
#include <assert.h>

RHACCommand::RHACCommand()
{
  uint32_t payload = 0;
  assert(WriteHeader((char *)&payload, sizeof(uint32_t), OFFSET_COMMAND_PAYLOAD_SIZE));
  payload_ = NULL;
}

RHACCommand::~RHACCommand()
{
  if (payload_ != NULL) {

    enum CommandKind command_kind = GetCommandKind();

    // payload is shared between all device executors
    if (command_kind == ADMemcpyToSymbol ||
        command_kind == ADMemcpy2DToArray) 
    {
      RHACCommandAll *a_cmd;
      a_cmd = reinterpret_cast<RHACCommandAll*>(this);
      assert(a_cmd != NULL);
      assert(a_cmd->CheckReferenceCount());

      // in host the payload is delete by the user
      if (!rhac_platform.IsHost()) 
        delete[] payload_;
    }
    else {
      delete[] payload_;
    }

  }
}

// CAUTION - device (device index in each node)
void RHACCommand::SetDefaultInfo(rhac_command_id_t command_id,
                                 enum CommandKind command_kind,
                                 int node, int device)
{
  assert(WriteHeader((char *)&command_id, sizeof(rhac_command_id_t), OFFSET_COMMAND_ID));
  assert(WriteHeader((char *)&command_kind, sizeof(enum CommandKind), OFFSET_COMMAND_KIND));
  assert(WriteHeader((char *)&node, sizeof(int), OFFSET_COMMAND_NODE));
  assert(WriteHeader((char *)&device, sizeof(int), OFFSET_COMMAND_DEVICE));
}

bool RHACCommand::WriteHeader(const char *src,
                              const size_t size,
                              const size_t offset)
{
  if (offset + size > COMMAND_HEADER_SIZE)
    return false;

  memcpy(header_ + offset, src, size);

  return true;
}

void RHACCommand::ReadHeader(char *dst,
                             const size_t size,
                             const size_t offset)
{
  assert(offset + size <= COMMAND_HEADER_SIZE);

  memcpy(dst, header_ + offset, size);
}

bool RHACCommand::WritePayload(const char *src,
                               const size_t size,
                               const size_t offset)
{
  uint32_t payload_size;
  payload_size = GetPayloadSize();
  assert(payload_ != NULL);
  assert(payload_size != 0);

  if (offset + size > payload_size)
    return false;

  memcpy(payload_ + offset, src, size);

  return true;
}

void RHACCommand::ReadPayload(char *dst,
                              const size_t size,
                              const size_t offset)
{
  uint32_t payload_size;
  payload_size = GetPayloadSize();
  assert(payload_ != NULL && payload_size != 0);
  assert(offset + size <= payload_size);

  memcpy(dst, payload_ + offset, size);
}

void* RHACCommand::ReadPayloadPtr(const size_t size,
                                  const size_t offset)
{
  uint32_t payload_size;
  payload_size = GetPayloadSize();
  assert(payload_ != NULL && payload_size != 0);
  assert(offset + size <= payload_size);

  return payload_ + offset;
}

// Default Getters 
rhac_command_id_t RHACCommand::GetCommandID()
{
  rhac_command_id_t ret;
  ReadHeader((char *)&ret, sizeof(rhac_command_id_t), OFFSET_COMMAND_ID);
  return ret;
}

enum CommandKind RHACCommand::GetCommandKind()
{
  enum CommandKind ret;
  ReadHeader((char *)&ret, sizeof(enum CommandKind), OFFSET_COMMAND_KIND);
  return ret;
}

int RHACCommand::GetTargetNode()
{
  int ret;
  ReadHeader((char *)&ret, sizeof(int), OFFSET_COMMAND_NODE);
  return ret;
}

int RHACCommand::GetTargetDevice()
{
  int ret;
  ReadHeader((char *)&ret, sizeof(int), OFFSET_COMMAND_DEVICE);
  return ret;
}

char* RHACCommand::GetHeaderPtr()
{
  return header_;
}

uint32_t RHACCommand::GetPayloadSize()
{
  uint32_t ret;
  ReadHeader((char *)&ret, sizeof(uint32_t), OFFSET_COMMAND_PAYLOAD_SIZE);
  return ret;
}

char* RHACCommand::GetPayloadPtr()
{
  return payload_;
}

// Default Setter
void RHACCommand::SetPayloadSize(uint32_t size)
{
  WriteHeader((char *)&size, sizeof(uint32_t), OFFSET_COMMAND_PAYLOAD_SIZE);
}

void RHACCommand::SetPayloadPtr(const char* src, const uint32_t size)
{
  assert(src != NULL);

  payload_ = const_cast<char *>(src);

  assert(size != 0);

  SetPayloadSize(size);
}

void RHACCommand::AllocPayload(size_t size)
{
  assert(payload_ == NULL);
  uint32_t temp_size;

  payload_ = new char[size];
  assert(payload_ != NULL);

  temp_size = (uint32_t)size;

  SetPayloadSize(temp_size);
}

// Other helper functions
bool RHACCommand::IsNodeCommand()
{
  enum CommandKind cmd_kind;

  cmd_kind = GetCommandKind();

  return (cmd_kind < DEVICE_COMMAND_START) ? true : false;
}

bool RHACCommand::IsDeviceCommand()
{
  enum CommandKind cmd_kind;

  cmd_kind = GetCommandKind();

  return (DEVICE_COMMAND_START <= cmd_kind && cmd_kind < ALL_COMMAND_START) ? true : false;
}

bool RHACCommand::IsAllCommand()
{
  enum CommandKind cmd_kind;

  cmd_kind = GetCommandKind();

  return (ALL_COMMAND_START <= cmd_kind && cmd_kind < ALL_DEVICE_COMMAND_START) ? true : false;
}

bool RHACCommand::IsAllDeviceCommand()
{
  enum CommandKind cmd_kind;

  cmd_kind = GetCommandKind();

  return (ALL_DEVICE_COMMAND_START <= cmd_kind) ? true : false;
}

bool RHACCommand::HasPayload()
{
  return (GetPayloadSize() > 0) ? true : false;
}

void RHACCommand::PrintInfo(const char *prefix)
{
  RHAC_LOG("(%s) Command Info - ID : %zu, Kind : %d, Node : %d, Device : %d",
            prefix,
            GetCommandID(),
            GetCommandKind(),
            GetTargetNode(),
            GetTargetDevice());
}
// =====================================================================
//  Varg Command 
RHACCommandVarg::RHACCommandVarg()
{
}
RHACCommandVarg::~RHACCommandVarg()
{
}

void RHACCommandVarg::PushArg(const char *src, const size_t size)
{
  bool t;
  t = WritePayload(src, size, payload_varg_head_offset_);
  assert(t);
  payload_varg_head_offset_ += size;
}

void* RHACCommandVarg::PopArgPtr(const size_t size)
{
  void *ret;
  ret = ReadPayloadPtr(size, payload_varg_head_offset_);
  payload_varg_head_offset_ += size;
  return ret;
}

// =====================================================================
//  NSVMReserve Command 
#define OFFSET_HEADER_RESERVE_SIZE OFFSET_COMMAND_HEADER_END
RHACCommandNSVMReserve::RHACCommandNSVMReserve() 
{
}

RHACCommandNSVMReserve::~RHACCommandNSVMReserve()
{
}

void RHACCommandNSVMReserve::SetReserveSize(size_t size)
{
  WriteHeader((char *)&size, sizeof(size_t), OFFSET_HEADER_RESERVE_SIZE);
}

size_t RHACCommandNSVMReserve::GetReserveSize()
{
  size_t ret;
  ReadHeader((char *)&ret, sizeof(size_t), OFFSET_HEADER_RESERVE_SIZE);
  return ret;
}

// =====================================================================
//  NSVMMemcpy Command 
#define OFFSET_HEADER_MEMCPY_DESTINATION OFFSET_COMMAND_HEADER_END
#define OFFSET_HEADER_MEMCPY_SOURCE (OFFSET_COMMAND_HEADER_END + 8)
#define OFFSET_HEADER_MEMCPY_SIZE (OFFSET_COMMAND_HEADER_END + 16)

RHACCommandNSVMMemcpy::RHACCommandNSVMMemcpy()
{
}

RHACCommandNSVMMemcpy::~RHACCommandNSVMMemcpy()
{
}

void RHACCommandNSVMMemcpy::SetDestination(uint64_t dst)
{
  WriteHeader((char *)&dst, sizeof(uint64_t), OFFSET_HEADER_MEMCPY_DESTINATION);
}

uint64_t RHACCommandNSVMMemcpy::GetDestination()
{
  uint64_t ret;
  ReadHeader((char *)&ret, sizeof(size_t), OFFSET_HEADER_MEMCPY_DESTINATION);
  return ret;
}

void RHACCommandNSVMMemcpy::SetSource(uint64_t src)
{
  WriteHeader((char *)&src, sizeof(uint64_t), OFFSET_HEADER_MEMCPY_SOURCE);
}

uint64_t RHACCommandNSVMMemcpy::GetSource()
{
  uint64_t ret;
  ReadHeader((char *)&ret, sizeof(uint64_t), OFFSET_HEADER_MEMCPY_SOURCE);
  return ret;
}

void RHACCommandNSVMMemcpy::SetSize(size_t size)
{
  WriteHeader((char *)&size, sizeof(size_t), OFFSET_HEADER_MEMCPY_SIZE);
}

size_t RHACCommandNSVMMemcpy::GetSize()
{
  size_t ret;
  ReadHeader((char *)&ret, sizeof(size_t), OFFSET_HEADER_MEMCPY_SIZE);
  return ret;
}
// =====================================================================
//  NSVMMemset Command 
#define OFFSET_HEADER_MEMSET_DESTINATION OFFSET_COMMAND_HEADER_END
#define OFFSET_HEADER_MEMSET_VALUE (OFFSET_COMMAND_HEADER_END + 8)
#define OFFSET_HEADER_MEMSET_SIZE (OFFSET_COMMAND_HEADER_END + 12)

RHACCommandNSVMMemset::RHACCommandNSVMMemset()
{
}

RHACCommandNSVMMemset::~RHACCommandNSVMMemset()
{
}

void RHACCommandNSVMMemset::SetDestination(uint64_t dst)
{
  WriteHeader((char *)&dst, sizeof(uint64_t), OFFSET_HEADER_MEMSET_DESTINATION);
}

uint64_t RHACCommandNSVMMemset::GetDestination()
{
  uint64_t ret;
  ReadHeader((char *)&ret, sizeof(uint64_t), OFFSET_HEADER_MEMSET_DESTINATION);
  return ret;
}

void RHACCommandNSVMMemset::SetValue(int value)
{
  WriteHeader((char *)&value, sizeof(int), OFFSET_HEADER_MEMSET_VALUE);
}

int RHACCommandNSVMMemset::GetValue()
{
  int ret;
  ReadHeader((char *)&ret, sizeof(int), OFFSET_HEADER_MEMSET_VALUE);
  return ret;
}

void RHACCommandNSVMMemset::SetSize(size_t size)
{
  WriteHeader((char *)&size, sizeof(size_t), OFFSET_HEADER_MEMSET_SIZE);
}

size_t RHACCommandNSVMMemset::GetSize()
{
  size_t ret;
  ReadHeader((char *)&ret, sizeof(size_t), OFFSET_HEADER_MEMSET_SIZE);
  return ret;
}

// =====================================================================
//  NBarrier Command 
#define OFFSET_PAYLOAD_NBARRIER_WAITNUM 0
#define OFFSET_PAYLOAD_NBARRIER_VARARG  4
RHACCommandNBarrier::RHACCommandNBarrier()
{
}

RHACCommandNBarrier::~RHACCommandNBarrier()
{
}

void RHACCommandNBarrier::SetNumWait(uint32_t num_wait)
{
  bool t;
  t = WritePayload((char *)&num_wait, sizeof(uint32_t), OFFSET_PAYLOAD_NBARRIER_WAITNUM);
  assert(t);
}

uint32_t RHACCommandNBarrier::GetNumWait()
{
  uint32_t ret;
  ReadPayload((char *)&ret, sizeof(uint32_t), OFFSET_PAYLOAD_NBARRIER_WAITNUM);
  return ret;
}

void RHACCommandNBarrier::PushWaitItem(int node, int dev, rhac_command_id_t id)
{
  PushArg((char *)&node, sizeof(int));
  PushArg((char *)&dev, sizeof(int));
  PushArg((char *)&id, sizeof(rhac_command_id_t));
}

void RHACCommandNBarrier::PopWaitItem(int *node, int *dev, rhac_command_id_t *id)
{
  int *node_ret;
  int *dev_ret;
  rhac_command_id_t *id_ret;

  node_ret = (int *)PopArgPtr(sizeof(int));
  dev_ret = (int *)PopArgPtr(sizeof(int));
  id_ret = (rhac_command_id_t *)PopArgPtr(sizeof(rhac_command_id_t));

  *node = *node_ret;
  *dev = *dev_ret;
  *id = *id_ret;
}

// Caution All children of RHACCommandVarg must have ResetVargHead !!
void RHACCommandNBarrier::ResetVargHead()
{
  payload_varg_head_offset_ = OFFSET_PAYLOAD_NBARRIER_VARARG;
}

// =====================================================================
//  RHACCommandNMemAdvise
#define OFFSET_HEADER_MEMADVICE_DEVPTR  OFFSET_COMMAND_HEADER_END
#define OFFSET_HEADER_MEMADVICE_COUNT   (OFFSET_COMMAND_HEADER_END + 8)
#define OFFSET_HEADER_MEMADVICE_ADVICE  (OFFSET_COMMAND_HEADER_END + 16)
#define OFFSET_HEADER_MEMADVICE_DEVICE  (OFFSET_COMMAND_HEADER_END + 20)

RHACCommandNMemAdvise::RHACCommandNMemAdvise()
{
}

RHACCommandNMemAdvise::~RHACCommandNMemAdvise()
{
}

void RHACCommandNMemAdvise::SetDevPtr(uint64_t devPtr)
{
  WriteHeader((char *)&devPtr, sizeof(uint64_t), OFFSET_HEADER_MEMADVICE_DEVPTR);
}

uint64_t RHACCommandNMemAdvise::GetDevPtr()
{
  uint64_t ret;
  ReadHeader((char *)&ret, sizeof(uint64_t), OFFSET_HEADER_MEMADVICE_DEVPTR);
  return ret;
}

void RHACCommandNMemAdvise::SetCount(size_t count)
{
  WriteHeader((char *)&count, sizeof(size_t), OFFSET_HEADER_MEMADVICE_COUNT);
}

size_t RHACCommandNMemAdvise::GetCount()
{
  size_t ret;
  ReadHeader((char *)&ret, sizeof(size_t), OFFSET_HEADER_MEMADVICE_COUNT);
  return ret;
}

void RHACCommandNMemAdvise::SetAdvice(cudaMemoryAdvise advice)
{
  WriteHeader((char *)&advice, sizeof(cudaMemoryAdvise), OFFSET_HEADER_MEMADVICE_ADVICE);
}

cudaMemoryAdvise RHACCommandNMemAdvise::GetAdvice()
{
  cudaMemoryAdvise ret;
  ReadHeader((char *)&ret, sizeof(cudaMemoryAdvise), OFFSET_HEADER_MEMADVICE_ADVICE);
  return ret;
}

void RHACCommandNMemAdvise::SetDevice(int device)
{
  WriteHeader((char *)&device, sizeof(int), OFFSET_HEADER_MEMADVICE_DEVICE);
}

int RHACCommandNMemAdvise::GetDevice()
{
  int ret;
  ReadHeader((char *)&ret, sizeof(int), OFFSET_HEADER_MEMADVICE_DEVICE);
  return ret;
}

// =====================================================================
//  RHACCommandNSplitVARange
#define OFFSET_HEADER_SPLITVARANGE_BASE   OFFSET_COMMAND_HEADER_END
#define OFFSET_HEADER_SPLITVARANGE_LENGTH (OFFSET_COMMAND_HEADER_END + 8)

RHACCommandNSplitVARange::RHACCommandNSplitVARange() {
}

RHACCommandNSplitVARange::~RHACCommandNSplitVARange() {
}

void RHACCommandNSplitVARange::SetBase(uint64_t base) {
  WriteHeader((char *)&base, sizeof(uint64_t), OFFSET_HEADER_SPLITVARANGE_BASE);
}

uint64_t RHACCommandNSplitVARange::GetBase() {
  uint64_t ret;
  ReadHeader((char *)&ret, sizeof(uint64_t), OFFSET_HEADER_SPLITVARANGE_BASE);
  return ret;
}

void RHACCommandNSplitVARange::SetLength(size_t length) {
  WriteHeader((char *)&length, sizeof(size_t), OFFSET_HEADER_SPLITVARANGE_LENGTH);
}

size_t RHACCommandNSplitVARange::GetLength() {
  size_t ret;
  ReadHeader((char *)&ret, sizeof(size_t), OFFSET_HEADER_SPLITVARANGE_LENGTH);
  return ret;
}

// =====================================================================
//  RHACCommandNSetDupFlag
#define OFFSET_HEADER_SETDUPFLAG_BASE   OFFSET_COMMAND_HEADER_END
#define OFFSET_HEADER_SETDUPFLAG_LENGTH (OFFSET_COMMAND_HEADER_END + 8)
#define OFFSET_HEADER_SETDUPFLAG_FLAG   (OFFSET_COMMAND_HEADER_END + 16)

RHACCommandNSetDupFlag::RHACCommandNSetDupFlag() {
}

RHACCommandNSetDupFlag::~RHACCommandNSetDupFlag() {
}

void RHACCommandNSetDupFlag::SetBase(uint64_t base) {
  WriteHeader((char *)&base, sizeof(uint64_t), OFFSET_HEADER_SETDUPFLAG_BASE);
}

uint64_t RHACCommandNSetDupFlag::GetBase() {
  uint64_t ret;
  ReadHeader((char *)&ret, sizeof(uint64_t), OFFSET_HEADER_SETDUPFLAG_BASE);
  return ret;
}

void RHACCommandNSetDupFlag::SetLength(size_t length) {
  WriteHeader((char *)&length, sizeof(size_t), OFFSET_HEADER_SETDUPFLAG_LENGTH);
}

size_t RHACCommandNSetDupFlag::GetLength() {
  size_t ret;
  ReadHeader((char *)&ret, sizeof(size_t), OFFSET_HEADER_SETDUPFLAG_LENGTH);
  return ret;
}

void RHACCommandNSetDupFlag::SetFlag(bool turnon) {
  WriteHeader((char *)&turnon, sizeof(bool), OFFSET_HEADER_SETDUPFLAG_FLAG);
}

bool RHACCommandNSetDupFlag::GetFlag() {
  bool ret;
  ReadHeader((char *)&ret, sizeof(bool), OFFSET_HEADER_SETDUPFLAG_FLAG);
  return ret;
}

// =====================================================================
//  RHACCommandNEventCreate
#define OFFSET_HEADER_EVENTCREATE_EVENTPTR OFFSET_COMMAND_HEADER_END

RHACCommandNEventCreate::RHACCommandNEventCreate()
{
}

RHACCommandNEventCreate::~RHACCommandNEventCreate()
{
}

void RHACCommandNEventCreate::SetEventPtr(cudaEvent_t *event)
{
  WriteHeader((char *)&event, sizeof(cudaEvent_t *), OFFSET_HEADER_EVENTCREATE_EVENTPTR);
}

cudaEvent_t* RHACCommandNEventCreate::GetEventPtr()
{
  assert(rhac_platform.IsHost());
  cudaEvent_t *ret;
  ReadHeader((char *)&ret, sizeof(cudaEvent_t *), OFFSET_HEADER_EVENTCREATE_EVENTPTR);
  return ret;
}

// =====================================================================
//  RHACCommandNEventRecord
#define OFFSET_HEADER_EVENTRECORD_EVENT OFFSET_COMMAND_HEADER_END
#define OFFSET_HEADER_EVENTRECORD_STREAM (OFFSET_COMMAND_HEADER_END + 8)

RHACCommandNEventRecord::RHACCommandNEventRecord()
{
}

RHACCommandNEventRecord::~RHACCommandNEventRecord()
{
}

void RHACCommandNEventRecord::SetEvent(cudaEvent_t event)
{
  // cudaEvent_t is pointer internally
  WriteHeader((char *)&event, sizeof(cudaEvent_t), OFFSET_HEADER_EVENTRECORD_EVENT);
}

cudaEvent_t RHACCommandNEventRecord::GetEvent()
{
  assert(rhac_platform.IsHost());
  cudaEvent_t ret;
  ReadHeader((char *)&ret, sizeof(cudaEvent_t), OFFSET_HEADER_EVENTRECORD_EVENT);
  return ret;
}

void RHACCommandNEventRecord::SetStream(cudaStream_t stream)
{
  // cudaStream_t is pointer internally 
  WriteHeader((char *)&stream, sizeof(cudaStream_t), OFFSET_HEADER_EVENTRECORD_STREAM);
}

cudaStream_t RHACCommandNEventRecord::GetStream()
{
  assert(rhac_platform.IsHost());
  cudaStream_t ret;
  ReadHeader((char *)&ret, sizeof(cudaStream_t), OFFSET_HEADER_EVENTRECORD_STREAM);
  return ret;
}

// =====================================================================
//  RHACCommandDKernelPartialExecution Command

#define OFFSET_PAYLOAD_KERNEL_FATBIN_INDEX  0
#define OFFSET_PAYLOAD_KERNEL_FUNCNAME      4
#define OFFSET_PAYLOAD_KERNEL_GRIDDIM       132
#define OFFSET_PAYLOAD_KERNEL_BLOCKDIM      144
#define OFFSET_PAYLOAD_KERNEL_SHAREDMEM     156
#define OFFSET_PAYLOAD_KERNEL_VARARG        164

RHACCommandDKernelPartialExecution::RHACCommandDKernelPartialExecution() 
{
  payload_varg_head_offset_ = OFFSET_PAYLOAD_KERNEL_VARARG;
}

RHACCommandDKernelPartialExecution::~RHACCommandDKernelPartialExecution() 
{
}

void RHACCommandDKernelPartialExecution::SetFatbinIndex(int fatbin_index)
{
  bool ret;
  ret = WritePayload((char*)&fatbin_index, sizeof(int),
      OFFSET_PAYLOAD_KERNEL_FATBIN_INDEX);
  assert(ret);
}

int RHACCommandDKernelPartialExecution::GetFatbinIndex()
{
  int fatbin_index;
  ReadPayload((char*)&fatbin_index, sizeof(int),
      OFFSET_PAYLOAD_KERNEL_FATBIN_INDEX);
  return fatbin_index;
}

void RHACCommandDKernelPartialExecution::SetFuncName(const char *func_name) 
{
  bool ret;
  ret = WritePayload(func_name, 128, OFFSET_PAYLOAD_KERNEL_FUNCNAME);
  assert(ret);
}

void RHACCommandDKernelPartialExecution::GetFuncName(char *func_name)
{
  ReadPayload(func_name, 128, OFFSET_PAYLOAD_KERNEL_FUNCNAME);
}

void RHACCommandDKernelPartialExecution::SetGridDim(dim3 gridDim)
{
  bool t;
  t = WritePayload((char *)&gridDim, sizeof(dim3), OFFSET_PAYLOAD_KERNEL_GRIDDIM);
  assert(t);
}

dim3 RHACCommandDKernelPartialExecution::GetGridDim()
{
  dim3 ret;
  ReadPayload((char *)&ret, sizeof(dim3), OFFSET_PAYLOAD_KERNEL_GRIDDIM);
  return ret;
}

void RHACCommandDKernelPartialExecution::SetBlockDim(dim3 blockDim)
{
  bool t;
  t = WritePayload((char *)&blockDim, sizeof(dim3), OFFSET_PAYLOAD_KERNEL_BLOCKDIM);
  assert(t);
}

dim3 RHACCommandDKernelPartialExecution::GetBlockDim()
{
  dim3 ret;
  ReadPayload((char *)&ret, sizeof(dim3), OFFSET_PAYLOAD_KERNEL_BLOCKDIM);
  return ret;
}

void RHACCommandDKernelPartialExecution::SetSharedMem(size_t sharedMem)
{
  bool t;
  t = WritePayload((char *)&sharedMem, sizeof(size_t), OFFSET_PAYLOAD_KERNEL_SHAREDMEM);
  assert(t);
}

size_t RHACCommandDKernelPartialExecution::GetSharedMem()
{
  size_t ret;
  ReadPayload((char *)&ret, sizeof(size_t), OFFSET_PAYLOAD_KERNEL_SHAREDMEM);
  return ret;
}

// Caution All children of RHACCommandVarg must have ResetVargHead !!
void RHACCommandDKernelPartialExecution::ResetVargHead()
{
  payload_varg_head_offset_ = OFFSET_PAYLOAD_KERNEL_VARARG;
}

// =====================================================================
//  DSetCacheConfig Command 
//  cudaFuncCache : enum (int)
#define OFFSET_HEADER_CACHE_CONFIG OFFSET_COMMAND_HEADER_END
RHACCommandDSetCacheConfig::RHACCommandDSetCacheConfig()
{
}

RHACCommandDSetCacheConfig::~RHACCommandDSetCacheConfig()
{
}

void RHACCommandDSetCacheConfig::SetCacheConfig(cudaFuncCache cache_config)
{
  WriteHeader((char *)&cache_config, sizeof(cudaFuncCache), OFFSET_HEADER_CACHE_CONFIG);
}

cudaFuncCache RHACCommandDSetCacheConfig::GetCacheConfig()
{
  cudaFuncCache ret;
  ReadHeader((char *)&ret, sizeof(cudaFuncCache), OFFSET_HEADER_CACHE_CONFIG);
  return ret;
}

// =====================================================================
//  DFuncSetCacheConfig Command
#define OFFSET_PAYLOAD_FUNCSETCACHECONFIG_FATBIN_INDEX  0
#define OFFSET_PAYLOAD_FUNCSETCACHECONFIG_FUNC_NAME     4
#define OFFSET_PAYLOAD_FUNCSETCACHECONFIG_CONFIG        132
#define OFFSET_PAYLOAD_FUNCSETCACHECONFIG_END           136

RHACCommandDFuncSetCacheConfig::RHACCommandDFuncSetCacheConfig() {
  AllocPayload(OFFSET_PAYLOAD_FUNCSETCACHECONFIG_END);
}

RHACCommandDFuncSetCacheConfig::~RHACCommandDFuncSetCacheConfig() {
}

void RHACCommandDFuncSetCacheConfig::SetFatbinIndex(int fatbin_index) {
  bool ret;
  ret = WritePayload((char*)&fatbin_index, sizeof(int),
      OFFSET_PAYLOAD_FUNCSETCACHECONFIG_FATBIN_INDEX);
  assert(ret);
}

int RHACCommandDFuncSetCacheConfig::GetFatbinIndex() {
  int fatbin_index;
  ReadPayload((char*)&fatbin_index, sizeof(int),
      OFFSET_PAYLOAD_FUNCSETCACHECONFIG_FATBIN_INDEX);
  return fatbin_index;
}

void RHACCommandDFuncSetCacheConfig::SetFuncName(const char *func_name) {
  bool ret;
  ret = WritePayload(func_name, 128, OFFSET_PAYLOAD_FUNCSETCACHECONFIG_FUNC_NAME);
  assert(ret);
}

void RHACCommandDFuncSetCacheConfig::GetFuncName(char *func_name) {
  ReadPayload(func_name, 128, OFFSET_PAYLOAD_FUNCSETCACHECONFIG_FUNC_NAME);
}

void RHACCommandDFuncSetCacheConfig::SetCacheConfig(cudaFuncCache cache_config) {
  bool ret;
  ret = WritePayload((char*)&cache_config, sizeof(cudaFuncCache),
      OFFSET_PAYLOAD_FUNCSETCACHECONFIG_CONFIG);
  assert(ret);
}

cudaFuncCache RHACCommandDFuncSetCacheConfig::GetCacheConfig() {
  cudaFuncCache ret;
  ReadPayload((char *)&ret, sizeof(cudaFuncCache),
      OFFSET_PAYLOAD_FUNCSETCACHECONFIG_CONFIG);
  return ret;
}

// =====================================================================
//  DMallocArray Command
#define OFFSET_HEADER_MALLOCARRAY_DESC_FORMAT OFFSET_COMMAND_HEADER_END
#define OFFSET_HEADER_MALLOCARRAY_DESC_W      (OFFSET_COMMAND_HEADER_END + 4)
#define OFFSET_HEADER_MALLOCARRAY_DESC_X      (OFFSET_COMMAND_HEADER_END + 8)
#define OFFSET_HEADER_MALLOCARRAY_DESC_Y      (OFFSET_COMMAND_HEADER_END + 12)
#define OFFSET_HEADER_MALLOCARRAY_DESC_Z      (OFFSET_COMMAND_HEADER_END + 16)
#define OFFSET_HEADER_MALLOCARRAY_WIDTH       (OFFSET_COMMAND_HEADER_END + 20)
#define OFFSET_HEADER_MALLOCARRAY_HEIGHT      (OFFSET_COMMAND_HEADER_END + 28)
#define OFFSET_HEADER_MALLOCARRAY_FLAGS       (OFFSET_COMMAND_HEADER_END + 36)
RHACCommandDMallocArray::RHACCommandDMallocArray() {
}

RHACCommandDMallocArray::~RHACCommandDMallocArray() {
}

void RHACCommandDMallocArray::SetDescFormat(cudaChannelFormatKind format) {
  WriteHeader((char*)&format, sizeof(cudaChannelFormatKind),
      OFFSET_HEADER_MALLOCARRAY_DESC_FORMAT);
}

cudaChannelFormatKind RHACCommandDMallocArray::GetDescFormat() {
  cudaChannelFormatKind format;
  ReadHeader((char*)&format, sizeof(cudaChannelFormatKind),
      OFFSET_HEADER_MALLOCARRAY_DESC_FORMAT);
  return format;
}

void RHACCommandDMallocArray::SetDescW(int w) {
  WriteHeader((char*)&w, sizeof(int), OFFSET_HEADER_MALLOCARRAY_DESC_W);
}

int RHACCommandDMallocArray::GetDescW() {
  int w;
  ReadHeader((char*)&w, sizeof(int), OFFSET_HEADER_MALLOCARRAY_DESC_W);
  return w;
}

void RHACCommandDMallocArray::SetDescX(int x) {
  WriteHeader((char*)&x, sizeof(int), OFFSET_HEADER_MALLOCARRAY_DESC_X);
}

int RHACCommandDMallocArray::GetDescX() {
  int x;
  ReadHeader((char*)&x, sizeof(int), OFFSET_HEADER_MALLOCARRAY_DESC_X);
  return x;
}

void RHACCommandDMallocArray::SetDescY(int y) {
  WriteHeader((char*)&y, sizeof(int), OFFSET_HEADER_MALLOCARRAY_DESC_Y);
}

int RHACCommandDMallocArray::GetDescY() {
  int y;
  ReadHeader((char*)&y, sizeof(int), OFFSET_HEADER_MALLOCARRAY_DESC_Y);
  return y;
}

void RHACCommandDMallocArray::SetDescZ(int z) {
  WriteHeader((char*)&z, sizeof(int), OFFSET_HEADER_MALLOCARRAY_DESC_Z);
}

int RHACCommandDMallocArray::GetDescZ() {
  int z;
  ReadHeader((char*)&z, sizeof(int), OFFSET_HEADER_MALLOCARRAY_DESC_Z);
  return z;
}

void RHACCommandDMallocArray::SetWidth(size_t width) {
  WriteHeader((char*)&width, sizeof(size_t), OFFSET_HEADER_MALLOCARRAY_WIDTH);
}

size_t RHACCommandDMallocArray::GetWidth() {
  size_t width;
  ReadHeader((char*)&width, sizeof(size_t), OFFSET_HEADER_MALLOCARRAY_WIDTH);
  return width;
}

void RHACCommandDMallocArray::SetHeight(size_t height) {
  WriteHeader((char*)&height, sizeof(size_t), OFFSET_HEADER_MALLOCARRAY_HEIGHT);
}

size_t RHACCommandDMallocArray::GetHeight() {
  size_t height;
  ReadHeader((char*)&height, sizeof(size_t), OFFSET_HEADER_MALLOCARRAY_HEIGHT);
  return height;
}

void RHACCommandDMallocArray::SetFlags(unsigned int flags) {
  WriteHeader((char*)&flags, sizeof(unsigned int), OFFSET_HEADER_MALLOCARRAY_FLAGS);
}

unsigned int RHACCommandDMallocArray::GetFlags() {
  unsigned int flags;
  ReadHeader((char*)&flags, sizeof(unsigned int), OFFSET_HEADER_MALLOCARRAY_FLAGS);
  return flags;
}

// =====================================================================
//  DFreeArray Command
#define OFFSET_HEADER_FREEARRAY_ARRAY_INDEX OFFSET_COMMAND_HEADER_END
RHACCommandDFreeArray::RHACCommandDFreeArray() {
}

RHACCommandDFreeArray::~RHACCommandDFreeArray() {
}

void RHACCommandDFreeArray::SetArrayIndex(unsigned int array_index) {
  WriteHeader((char*)&array_index, sizeof(unsigned int),
      OFFSET_HEADER_FREEARRAY_ARRAY_INDEX);
}

unsigned int RHACCommandDFreeArray::GetArrayIndex() {
  unsigned int array_index;
  ReadHeader((char*)&array_index, sizeof(unsigned int),
      OFFSET_HEADER_FREEARRAY_ARRAY_INDEX);
  return array_index;
}

// =====================================================================
//  DBindTexture
#define OFFSET_PAYLOAD_BINDTEXTURE_FATBIN_INDEX   0
#define OFFSET_PAYLOAD_BINDTEXTURE_REF_NAME       4
#define OFFSET_PAYLOAD_BINDTEXTURE_REF_FILTERMODE 132
#define OFFSET_PAYLOAD_BINDTEXTURE_REF_NORMALIZED 136
#define OFFSET_PAYLOAD_BINDTEXTURE_DEVPTR         140
#define OFFSET_PAYLOAD_BINDTEXTURE_DESC_X         148
#define OFFSET_PAYLOAD_BINDTEXTURE_DESC_Y         152
#define OFFSET_PAYLOAD_BINDTEXTURE_DESC_Z         156
#define OFFSET_PAYLOAD_BINDTEXTURE_DESC_W         160
#define OFFSET_PAYLOAD_BINDTEXTURE_DESC_FORMAT    164
#define OFFSET_PAYLOAD_BINDTEXTURE_SIZE           168
#define OFFSET_PAYLOAD_BINDTEXTURE_END            176

RHACCommandDBindTexture::RHACCommandDBindTexture()
{
  AllocPayload(OFFSET_PAYLOAD_BINDTEXTURE_END);
}

RHACCommandDBindTexture::~RHACCommandDBindTexture()
{
}

void RHACCommandDBindTexture::SetFatbinIndex(int fatbin_index) {
  bool ret;
  ret = WritePayload((char*)&fatbin_index, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_FATBIN_INDEX);
  assert(ret);
}

int RHACCommandDBindTexture::GetFatbinIndex() {
  int fatbin_index;
  ReadPayload((char*)&fatbin_index, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_FATBIN_INDEX);
  return fatbin_index;
}

void RHACCommandDBindTexture::SetRefName(const char *ref_name) {
  bool ret;
  ret = WritePayload(ref_name, 128, OFFSET_PAYLOAD_BINDTEXTURE_REF_NAME);
  assert(ret);
}

void RHACCommandDBindTexture::GetRefName(char *ref_name) {
  ReadPayload(ref_name, 128, OFFSET_PAYLOAD_BINDTEXTURE_REF_NAME);
}

void RHACCommandDBindTexture::SetRefFilterMode(int mode) {
  bool ret;
  ret = WritePayload((char*)&mode, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_REF_FILTERMODE);
  assert(ret);
}

int RHACCommandDBindTexture::GetRefFilterMode() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_REF_FILTERMODE);
  return ret;
}

void RHACCommandDBindTexture::SetRefNormalized(int flag) {
  bool ret;
  ret = WritePayload((char*)&flag, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_REF_NORMALIZED);
  assert(ret);
}

int RHACCommandDBindTexture::GetRefNormalized() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_REF_NORMALIZED);
  return ret;
}

void RHACCommandDBindTexture::SetDevPtr(void *ptr) {
  bool ret;
  ret = WritePayload((char*)&ptr, sizeof(void*),
      OFFSET_PAYLOAD_BINDTEXTURE_DEVPTR);
  assert(ret);
}

void* RHACCommandDBindTexture::GetDevPtr() {
  void* ret;
  ReadPayload((char *)&ret, sizeof(void*),
      OFFSET_PAYLOAD_BINDTEXTURE_DEVPTR);
  return ret;
}

void RHACCommandDBindTexture::SetDescX(int x) {
  bool ret;
  ret = WritePayload((char*)&x, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_DESC_X);
  assert(ret);
}

int RHACCommandDBindTexture::GetDescX() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_DESC_X);
  return ret;
}

void RHACCommandDBindTexture::SetDescY(int y) {
  bool ret;
  ret = WritePayload((char*)&y, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_DESC_Y);
  assert(ret);
}

int RHACCommandDBindTexture::GetDescY() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_DESC_Y);
  return ret;
}

void RHACCommandDBindTexture::SetDescZ(int z) {
  bool ret;
  ret = WritePayload((char*)&z, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_DESC_Z);
  assert(ret);
}

int RHACCommandDBindTexture::GetDescZ() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_DESC_Z);
  return ret;
}

void RHACCommandDBindTexture::SetDescW(int w) {
  bool ret;
  ret = WritePayload((char*)&w, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_DESC_W);
  assert(ret);
}

int RHACCommandDBindTexture::GetDescW() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURE_DESC_W);
  return ret;
}

void RHACCommandDBindTexture::SetDescFormat(cudaChannelFormatKind kind) {
  bool ret;
  ret = WritePayload((char*)&kind, sizeof(cudaChannelFormatKind),
      OFFSET_PAYLOAD_BINDTEXTURE_DESC_FORMAT);
  assert(ret);
}

cudaChannelFormatKind RHACCommandDBindTexture::GetDescFormat() {
  cudaChannelFormatKind ret;
  ReadPayload((char *)&ret, sizeof(cudaChannelFormatKind),
      OFFSET_PAYLOAD_BINDTEXTURE_DESC_FORMAT);
  return ret;
}

void RHACCommandDBindTexture::SetSize(size_t size) {
  bool ret;
  ret = WritePayload((char*)&size, sizeof(size_t),
      OFFSET_PAYLOAD_BINDTEXTURE_SIZE);
  assert(ret);
}

size_t RHACCommandDBindTexture::GetSize() {
  size_t ret;
  ReadPayload((char *)&ret, sizeof(size_t),
      OFFSET_PAYLOAD_BINDTEXTURE_SIZE);
  return ret;
}

// =====================================================================
//  DBindTextureToArray
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_FATBIN_INDEX   0
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_REF_NAME       4
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_REF_FILTERMODE 132
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_REF_NORMALIZED 136
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_ARRAY_INDEX    140
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_X         144
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_Y         148
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_Z         152
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_W         156
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_FORMAT    160
#define OFFSET_PAYLOAD_BINDTEXTURETOARRAY_END            164

RHACCommandDBindTextureToArray::RHACCommandDBindTextureToArray()
{
  AllocPayload(OFFSET_PAYLOAD_BINDTEXTURETOARRAY_END);
}

RHACCommandDBindTextureToArray::~RHACCommandDBindTextureToArray()
{
}

void RHACCommandDBindTextureToArray::SetFatbinIndex(int fatbin_index) {
  bool ret;
  ret = WritePayload((char*)&fatbin_index, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_FATBIN_INDEX);
  assert(ret);
}

int RHACCommandDBindTextureToArray::GetFatbinIndex() {
  int fatbin_index;
  ReadPayload((char*)&fatbin_index, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_FATBIN_INDEX);
  return fatbin_index;
}

void RHACCommandDBindTextureToArray::SetRefName(const char *ref_name) {
  bool ret;
  ret = WritePayload(ref_name, 128, OFFSET_PAYLOAD_BINDTEXTURETOARRAY_REF_NAME);
  assert(ret);
}

void RHACCommandDBindTextureToArray::GetRefName(char *ref_name) {
  ReadPayload(ref_name, 128, OFFSET_PAYLOAD_BINDTEXTURETOARRAY_REF_NAME);
}

void RHACCommandDBindTextureToArray::SetRefFilterMode(int mode) {
  bool ret;
  ret = WritePayload((char*)&mode, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_REF_FILTERMODE);
  assert(ret);
}

int RHACCommandDBindTextureToArray::GetRefFilterMode() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_REF_FILTERMODE);
  return ret;
}

void RHACCommandDBindTextureToArray::SetRefNormalized(int flag) {
  bool ret;
  ret = WritePayload((char*)&flag, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_REF_NORMALIZED);
  assert(ret);
}

int RHACCommandDBindTextureToArray::GetRefNormalized() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_REF_NORMALIZED);
  return ret;
}

void RHACCommandDBindTextureToArray::SetArrayIndex(unsigned int array_index) {
  bool ret;
  ret = WritePayload((char*)&array_index, sizeof(unsigned int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_ARRAY_INDEX);
  assert(ret);
}

unsigned int RHACCommandDBindTextureToArray::GetArrayIndex() {
  unsigned int ret;
  ReadPayload((char *)&ret, sizeof(unsigned int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_ARRAY_INDEX);
  return ret;
}

void RHACCommandDBindTextureToArray::SetDescX(int x) {
  bool ret;
  ret = WritePayload((char*)&x, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_X);
  assert(ret);
}

int RHACCommandDBindTextureToArray::GetDescX() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_X);
  return ret;
}

void RHACCommandDBindTextureToArray::SetDescY(int y) {
  bool ret;
  ret = WritePayload((char*)&y, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_Y);
  assert(ret);
}

int RHACCommandDBindTextureToArray::GetDescY() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_Y);
  return ret;
}

void RHACCommandDBindTextureToArray::SetDescZ(int z) {
  bool ret;
  ret = WritePayload((char*)&z, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_Z);
  assert(ret);
}

int RHACCommandDBindTextureToArray::GetDescZ() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_Z);
  return ret;
}

void RHACCommandDBindTextureToArray::SetDescW(int w) {
  bool ret;
  ret = WritePayload((char*)&w, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_W);
  assert(ret);
}

int RHACCommandDBindTextureToArray::GetDescW() {
  int ret;
  ReadPayload((char *)&ret, sizeof(int),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_W);
  return ret;
}

void RHACCommandDBindTextureToArray::SetDescFormat(cudaChannelFormatKind kind) {
  bool ret;
  ret = WritePayload((char*)&kind, sizeof(cudaChannelFormatKind),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_FORMAT);
  assert(ret);
}

cudaChannelFormatKind RHACCommandDBindTextureToArray::GetDescFormat() {
  cudaChannelFormatKind ret;
  ReadPayload((char *)&ret, sizeof(cudaChannelFormatKind),
      OFFSET_PAYLOAD_BINDTEXTURETOARRAY_DESC_FORMAT);
  return ret;
}

// =====================================================================
//  Command for All executors
RHACCommandAll::RHACCommandAll()
{
}

RHACCommandAll::~RHACCommandAll()
{
}

void RHACCommandAll::SetReferenceCount(int cnt)
{
  WriteHeader((char *)&cnt, sizeof(int), OFFSET_COMMAND_REFERENCE_COUNTER);
}

int RHACCommandAll::DecreaseReferenceCount()
{
  volatile int *ref_cnt_;
  int ret;
  ref_cnt_ = (volatile int *)(header_ + OFFSET_COMMAND_REFERENCE_COUNTER);

  ret = __sync_sub_and_fetch(ref_cnt_, 1);
  
  return ret;
}

bool RHACCommandAll::CheckReferenceCount()
{
  volatile int *ref_cnt_;
  ref_cnt_ = (volatile int *)(header_ + OFFSET_COMMAND_REFERENCE_COUNTER);

  return (*ref_cnt_ == 0) ? true : false;
}

int RHACCommandAll::GetReferenceCount()
{
  volatile int *ref_cnt_;
  ref_cnt_ = (volatile int *)(header_ + OFFSET_COMMAND_REFERENCE_COUNTER);

  return *ref_cnt_;
}

// =====================================================================
//  Command for All device executors
RHACCommandAllDevice::RHACCommandAllDevice()
{
}

RHACCommandAllDevice::~RHACCommandAllDevice()
{
}

// =====================================================================
//  ADMemcpyToSymbol
#define OFFSET_HEADER_MEMCPYTOSYMBOL_FATBIN_INDEX   OFFSET_COMMAND_HEADER_END
#define OFFSET_HEADER_MEMCPYTOSYMBOL_SYMBOL_NAME    (OFFSET_COMMAND_HEADER_END + 4)
#define OFFSET_HEADER_MEMCPYTOSYMBOL_KIND           (OFFSET_COMMAND_HEADER_END + 28)
#define OFFSET_HEADER_MEMCPYTOSYMBOL_OFFSET         (OFFSET_COMMAND_HEADER_END + 32)

#define OFFSET_PAYLOAD_MEMCPYTOSYMBOL_SOURCE        0

RHACCommandADMemcpyToSymbol::RHACCommandADMemcpyToSymbol()
{
}

RHACCommandADMemcpyToSymbol::~RHACCommandADMemcpyToSymbol()
{
}

void RHACCommandADMemcpyToSymbol::SetFatbinIndex(int fatbin_index) 
{
  WriteHeader((char *)&fatbin_index, sizeof(int), OFFSET_HEADER_MEMCPYTOSYMBOL_FATBIN_INDEX);
}

int RHACCommandADMemcpyToSymbol::GetFatbinIndex() 
{
  int fatbin_index;
  ReadHeader((char *)&fatbin_index, sizeof(int), OFFSET_HEADER_MEMCPYTOSYMBOL_FATBIN_INDEX);
  return fatbin_index;
}

void RHACCommandADMemcpyToSymbol::SetSymbolName(const char *symbol) 
{
  assert(strlen(symbol) < 27);

  WriteHeader(symbol, sizeof(char)*(strlen(symbol) + 1), OFFSET_HEADER_MEMCPYTOSYMBOL_SYMBOL_NAME);
}

void RHACCommandADMemcpyToSymbol::GetSymbolName(char *symbol) 
{
  // FIXME - strlen ??
  ReadHeader(symbol, sizeof(char)*30, OFFSET_HEADER_MEMCPYTOSYMBOL_SYMBOL_NAME);
}

void RHACCommandADMemcpyToSymbol::SetMemcpyKind(cudaMemcpyKind kind)
{
  assert(kind == cudaMemcpyHostToDevice);
  WriteHeader((char *)&kind, sizeof(cudaMemcpyKind), OFFSET_HEADER_MEMCPYTOSYMBOL_KIND);
}

cudaMemcpyKind RHACCommandADMemcpyToSymbol::GetMemcpyKind()
{
  cudaMemcpyKind ret;
  ReadHeader((char *)&ret, sizeof(cudaMemcpyKind), OFFSET_HEADER_MEMCPYTOSYMBOL_KIND);
  return ret;
}

void RHACCommandADMemcpyToSymbol::SetOffset(uint64_t offset)
{
  WriteHeader((char *)&offset, sizeof(uint64_t), OFFSET_HEADER_MEMCPYTOSYMBOL_OFFSET);
}

uint64_t RHACCommandADMemcpyToSymbol::GetOffset()
{
  uint64_t ret;
  ReadHeader((char *)&ret, sizeof(uint64_t), OFFSET_HEADER_MEMCPYTOSYMBOL_OFFSET);
  return ret;
}

void RHACCommandADMemcpyToSymbol::SetSourceAndCount(const char *src, uint32_t size)
{
  SetPayloadPtr(src, size);
}

char *RHACCommandADMemcpyToSymbol::GetSource()
{
  return payload_;
}

uint32_t RHACCommandADMemcpyToSymbol::GetCount()
{
  return GetPayloadSize();
}

// =====================================================================
//  ADMemcpy2DToArray
#define OFFSET_HEADER_MEMCPY2DTOARRAY_ARRAY_INDEX   OFFSET_COMMAND_HEADER_END
#define OFFSET_HEADER_MEMCPY2DTOARRAY_W_OFFSET      (OFFSET_COMMAND_HEADER_END + 4)
#define OFFSET_HEADER_MEMCPY2DTOARRAY_H_OFFSET      (OFFSET_COMMAND_HEADER_END + 12)
#define OFFSET_HEADER_MEMCPY2DTOARRAY_S_PITCH       (OFFSET_COMMAND_HEADER_END + 20)
#define OFFSET_HEADER_MEMCPY2DTOARRAY_WIDTH         (OFFSET_COMMAND_HEADER_END + 28)
#define OFFSET_HEADER_MEMCPY2DTOARRAY_KIND          (OFFSET_COMMAND_HEADER_END + 36)

RHACCommandADMemcpy2DToArray::RHACCommandADMemcpy2DToArray() {
}

RHACCommandADMemcpy2DToArray::~RHACCommandADMemcpy2DToArray() {
}

void RHACCommandADMemcpy2DToArray::SetArrayIndex(unsigned int array_index) {
  WriteHeader((char*)&array_index, sizeof(unsigned int),
      OFFSET_HEADER_MEMCPY2DTOARRAY_ARRAY_INDEX);
}

unsigned int RHACCommandADMemcpy2DToArray::GetArrayIndex() {
  unsigned int array_index;
  ReadHeader((char*)&array_index, sizeof(unsigned int),
      OFFSET_HEADER_MEMCPY2DTOARRAY_ARRAY_INDEX);
  return array_index;
}

void RHACCommandADMemcpy2DToArray::SetWOffset(size_t wOffset) {
  WriteHeader((char*)&wOffset, sizeof(size_t),
      OFFSET_HEADER_MEMCPY2DTOARRAY_W_OFFSET);
}

size_t RHACCommandADMemcpy2DToArray::GetWOffset() {
  size_t ret;
  ReadHeader((char*)&ret, sizeof(size_t),
      OFFSET_HEADER_MEMCPY2DTOARRAY_W_OFFSET);
  return ret;
}

void RHACCommandADMemcpy2DToArray::SetHOffset(size_t hOffset) {
  WriteHeader((char*)&hOffset, sizeof(size_t),
      OFFSET_HEADER_MEMCPY2DTOARRAY_H_OFFSET);
}

size_t RHACCommandADMemcpy2DToArray::GetHOffset() {
  size_t ret;
  ReadHeader((char*)&ret, sizeof(size_t),
      OFFSET_HEADER_MEMCPY2DTOARRAY_H_OFFSET);
  return ret;
}

void RHACCommandADMemcpy2DToArray::SetSourceAndCount(const void *src,
    uint32_t size) {
  SetPayloadPtr((const char*)src, size);
}

void* RHACCommandADMemcpy2DToArray::GetSource() {
  return payload_;
}

void RHACCommandADMemcpy2DToArray::SetSPitch(size_t spitch) {
  WriteHeader((char*)&spitch, sizeof(size_t),
      OFFSET_HEADER_MEMCPY2DTOARRAY_S_PITCH);
}

size_t RHACCommandADMemcpy2DToArray::GetSPitch() {
  size_t ret;
  ReadHeader((char*)&ret, sizeof(size_t),
      OFFSET_HEADER_MEMCPY2DTOARRAY_S_PITCH);
  return ret;
}

void RHACCommandADMemcpy2DToArray::SetWidth(size_t width) {
  WriteHeader((char*)&width, sizeof(size_t),
      OFFSET_HEADER_MEMCPY2DTOARRAY_WIDTH);
}

size_t RHACCommandADMemcpy2DToArray::GetWidth() {
  size_t ret;
  ReadHeader((char*)&ret, sizeof(size_t),
      OFFSET_HEADER_MEMCPY2DTOARRAY_WIDTH);
  return ret;
}

size_t RHACCommandADMemcpy2DToArray::GetHeight() {
  return GetPayloadSize() / GetSPitch();
}

void RHACCommandADMemcpy2DToArray::SetMemcpyKind(cudaMemcpyKind kind) {
  assert(kind == cudaMemcpyHostToDevice);
  WriteHeader((char *)&kind, sizeof(cudaMemcpyKind),
      OFFSET_HEADER_MEMCPY2DTOARRAY_KIND);
}

cudaMemcpyKind RHACCommandADMemcpy2DToArray::GetMemcpyKind() {
  cudaMemcpyKind ret;
  ReadHeader((char *)&ret, sizeof(cudaMemcpyKind),
      OFFSET_HEADER_MEMCPY2DTOARRAY_KIND);
  return ret;
}

