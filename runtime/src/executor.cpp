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

#include "executor.h"
#include "platform.h"
#include "utils.h"
#include "rhac_command.h"
#include "rhac_response.h"
#include "libcuda.h"
#include "rhac_driver.h"
#include "fatbin_handler.h"
#include "config.h"
#include "rhac_event.h"

#include <unistd.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>

// ====================================================================
// Executor abstract class
Executor::Executor(volatile rhac_command_id_t* response_queue,
                   RequestQueue* request_queue)
  : Thread(AFFINITY_TYPE_0),
  response_queue_(response_queue), request_queue_(request_queue) {
}

Executor::~Executor() {
}

void Executor::run_() {
  bool quit_signal = false;
  rhac_command_id_t cmd_id;
  RHACCommand *cmd;
  enum CommandKind cmd_kind;

  // FIXME
  PrintStart();

  while (!quit_signal) {
    if (request_queue_->Dequeue(&cmd)) {
      quit_signal = CheckQuit(cmd);
      Execute(cmd);

      // post response 
      cmd_id = cmd->GetCommandID();
      cmd_kind = cmd->GetCommandKind();
      assert(*response_queue_ <= cmd_id);
      *response_queue_ = cmd_id;

      if (cmd_kind == ADMemcpyToSymbol ||
          cmd_kind == ADMemcpy2DToArray ||
          cmd_kind == AGlobalBarrier) 
      {
        RHACCommandAll *a_cmd;
        int ref_cnt;
        a_cmd = reinterpret_cast<RHACCommandAll *>(cmd);
        ref_cnt = a_cmd->DecreaseReferenceCount();

        if (ref_cnt == 0)
          delete cmd;
      }
      else {
        delete cmd;
      }
    }
  }

  // FIXME
  PrintEnd();
}

bool Executor::CheckQuit(RHACCommand *cmd)
{
  enum CommandKind cmd_kind;
  cmd_kind = cmd->GetCommandKind();

  if (cmd_kind == NExit || cmd_kind == DExit)
    return true;

  return false;
}


// ====================================================================
// Node Executor class
NodeExecutor::NodeExecutor(volatile rhac_command_id_t* response_queue,
                           RequestQueue* request_queue)
  :Executor(response_queue, request_queue) {
#ifdef RHAC_MEMCPY_HELPER
  for (unsigned int i = 0; i < NUM_MEMCPY_THREAD; ++i) {
    memcpy_helper_[i] = new MemcpyHelper(i);
    memcpy_helper_[i]->Run();
  }
#endif
}

NodeExecutor::~NodeExecutor() {
#ifdef RHAC_MEMCPY_HELPER
  for (unsigned int i = 0; i < NUM_MEMCPY_THREAD; ++i) {
    memcpy_helper_[i]->Kill();
    memcpy_helper_[i]->Quit();
  }
#endif
}

void NodeExecutor::PrintStart()
{
  RHAC_LOG("Rank %d NodeExecutor Start", rhac_platform.GetRank());
}

void NodeExecutor::PrintEnd()
{
  RHAC_LOG("Rank %d NodeExecutor End ", rhac_platform.GetRank());
}

void NodeExecutor::Execute(RHACCommand *cmd)
{
  enum CommandKind cmd_kind;
  cmd_kind = cmd->GetCommandKind();

  assert(cmd->IsNodeCommand() || cmd->IsAllCommand());

  switch (cmd_kind) {
    case NExit:
      break;
    case NSVMMemcpyHostToDevice:
      ExecuteNSVMMemcpyHostToDevice(cmd);
      break;
    case NSVMMemcpyDeviceToHost:
      ExecuteNSVMMemcpyDeviceToHost(cmd);
      break;
    case NSVMMemcpyHostToHost:
      ExecuteNSVMMemcpyHostToHost(cmd);
      break;
    case NSVMMemcpyDeviceToDevice:
      ExecuteNSVMMemcpyDeviceToDevice(cmd);
      break;
    case NSVMMemcpyAsyncHostToDevice:
      ExecuteNSVMMemcpyAsyncHostToDevice(cmd);
      break;
    case NSVMMemcpyAsyncDeviceToHost:
      ExecuteNSVMMemcpyAsyncDeviceToHost(cmd);
      break;
    case NSVMMemset:
      ExecuteNSVMMemset(cmd);
      break;
    case NSVMMemsetAsync:
      ExecuteNSVMMemsetAsync(cmd);
      break;
    case NSVMReserve:
      ExecuteNSVMReserve(cmd);
      break;
    case NSVMSync:
      ExecuteNSVMSync(cmd);
      break;
    case NBarrier:
      ExecuteNBarrier(cmd);
      break;
    case NEventCreate:
      if (rhac_platform.IsHost())
        ExecuteNEventCreate(cmd);
      break;
    case NEventRecord:
      if (rhac_platform.IsHost())
        ExecuteNEventRecord(cmd);
      break;
    case NMemAdvise:
      ExecuteNMemAdvise(cmd);
      break;
    case NSplitVARange:
      ExecuteNSplitVARange(cmd);
      break;
    case NSetDupFlag:
      ExecuteNSetDupFlag(cmd);
      break;
    case AGlobalBarrier:
      ExecuteAGlobalBarrier(cmd);
      break;
    default:
      RHAC_LOG(" Unimplemented cmd_kind = %d", cmd_kind);
      YD_TODO();
      break;
  }
}

void NodeExecutor::SVMMemcpy(RHACCommandNSVMMemcpy *m_cmd)
{
  void *dst, *src;
  size_t size;

  dst = (void *)(m_cmd->GetDestination());
  src = (void *)(m_cmd->GetSource());
  size = m_cmd->GetSize();

  memcpy(dst, src, size);
}

void NodeExecutor::ExecuteNSVMMemcpyHostToDevice(RHACCommand *cmd) {
  assert(rhac_platform.IsHost());
  RHACCommandNSVMMemcpy *m_cmd = (RHACCommandNSVMMemcpy *)cmd;
  size_t copy_size = m_cmd->GetSize();

  rhac_platform.FinishAllRequestQueueExcept(0);

#if defined(RHAC_MEMCPY_HELPER)
  if (copy_size < MEMCPY_CHUNK_SIZE) {
#endif
#if defined(RHAC_PREFETCH)
    RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
    rhac_driver->PrefetchToCPU(m_cmd->GetDestination(), copy_size, -1, true);
#endif
    SVMMemcpy(m_cmd);
#if defined(RHAC_MEMCPY_HELPER)
  }
  else {
    // spread jobs to workers
    int active_threads = std::min((int)
        std::ceil((double)copy_size / MEMCPY_CHUNK_SIZE), NUM_MEMCPY_THREAD);
    size_t size_per_threads = copy_size / active_threads;

    size_t mod = size_per_threads % MEMCPY_CHUNK_SIZE;
    if (mod != 0)
      size_per_threads += (MEMCPY_CHUNK_SIZE - mod);

    size_t next_dst = m_cmd->GetDestination();
    size_t next_src = m_cmd->GetSource();
    for (int i = 0; i < active_threads; ++i) {
      size_t actual_size = std::min(size_per_threads, copy_size);
      memcpy_helper_[i]->Enqueue(next_dst, next_src, actual_size, 0);
      copy_size -= actual_size;
      next_dst += actual_size;
      next_src += actual_size;
    }

    // wait for all copies to finish
    for (int i = 0; i < active_threads; ++i) {
      memcpy_helper_[i]->Wait();
    }
  }
#endif
}

void NodeExecutor::ExecuteNSVMMemcpyDeviceToHost(RHACCommand *cmd) {
  assert(rhac_platform.IsHost());
  RHACCommandNSVMMemcpy *m_cmd = (RHACCommandNSVMMemcpy *)cmd;
  size_t copy_size = m_cmd->GetSize();

  rhac_platform.FinishAllRequestQueueExcept(0);

#if defined(RHAC_MEMCPY_HELPER)
  if (copy_size < MEMCPY_CHUNK_SIZE) {
#endif
#if defined(RHAC_PREFETCH)
  RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
  rhac_driver->PrefetchToCPU(m_cmd->GetSource(), m_cmd->GetSize(), -1, true);
#endif
  SVMMemcpy(m_cmd);
#if defined(RHAC_MEMCPY_HELPER)
  }
  else {
    // spread jobs to workers
    int active_threads = std::min((int)
        std::ceil((double)copy_size / MEMCPY_CHUNK_SIZE), NUM_MEMCPY_THREAD);
    size_t size_per_threads = copy_size / active_threads;

    size_t mod = size_per_threads % MEMCPY_CHUNK_SIZE;
    if (mod != 0)
      size_per_threads += (MEMCPY_CHUNK_SIZE - mod);

    size_t next_dst = m_cmd->GetDestination();
    size_t next_src = m_cmd->GetSource();
    for (int i = 0; i < active_threads; ++i) {
      size_t actual_size = std::min(size_per_threads, copy_size);
      memcpy_helper_[i]->Enqueue(next_dst, next_src, actual_size, 1);
      copy_size -= actual_size;
      next_dst += actual_size;
      next_src += actual_size;
    }

    // wait for all copies to finish
    for (int i = 0; i < active_threads; ++i) {
      memcpy_helper_[i]->Wait();
    }
  }
#endif
}

void NodeExecutor::ExecuteNSVMMemcpyHostToHost(RHACCommand *cmd)
{
  assert(rhac_platform.IsHost());
  RHACCommandNSVMMemcpy *m_cmd = (RHACCommandNSVMMemcpy *)cmd;

  rhac_platform.FinishAllRequestQueueExcept(0);

#if 1
  SVMMemcpy(m_cmd);
#else
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  libcuda->cudaMemcpy((void*)(m_cmd->GetDestination()),
                      (void *)(m_cmd->GetSource()),
                      m_cmd->GetSize(),
                      cudaMemcpyHostToHost);
#endif
}

void NodeExecutor::ExecuteNSVMMemcpyDeviceToDevice(RHACCommand *cmd)
{
  assert(rhac_platform.IsHost());
  RHACCommandNSVMMemcpy *m_cmd = (RHACCommandNSVMMemcpy *)cmd;

  rhac_platform.FinishAllRequestQueueExcept(0);

#if 1
  SVMMemcpy(m_cmd);
#else
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  libcuda->cudaMemcpy((void*)(m_cmd->GetDestination()),
                      (void *)(m_cmd->GetSource()),
                      m_cmd->GetSize(),
                      cudaMemcpyDeviceToDevice);
#endif
}

void NodeExecutor::ExecuteNSVMMemcpyAsyncHostToDevice(RHACCommand *cmd)
{
  assert(rhac_platform.IsHost());
  RHACCommandNSVMMemcpy *m_cmd = (RHACCommandNSVMMemcpy *)cmd;
  size_t copy_size = m_cmd->GetSize();

//  rhac_platform.FinishAllRequestQueueExcept(0);

#if defined(RHAC_MEMCPY_HELPER)
  if (copy_size < MEMCPY_CHUNK_SIZE) {
#endif
#if defined(RHAC_PREFETCH)
    RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
    rhac_driver->PrefetchToCPU(m_cmd->GetDestination(), copy_size, -1, true);
#endif
    SVMMemcpy(m_cmd);
#if defined(RHAC_MEMCPY_HELPER)
  }
  else {
    // spread jobs to workers
    int active_threads = std::min((int)
        std::ceil((double)copy_size / MEMCPY_CHUNK_SIZE), NUM_MEMCPY_THREAD);
    size_t size_per_threads = copy_size / active_threads;

    size_t mod = size_per_threads % MEMCPY_CHUNK_SIZE;
    if (mod != 0)
      size_per_threads += (MEMCPY_CHUNK_SIZE - mod);

    size_t next_dst = m_cmd->GetDestination();
    size_t next_src = m_cmd->GetSource();
    for (int i = 0; i < active_threads; ++i) {
      size_t actual_size = std::min(size_per_threads, copy_size);
      memcpy_helper_[i]->Enqueue(next_dst, next_src, actual_size, 0);
      copy_size -= actual_size;
      next_dst += actual_size;
      next_src += actual_size;
    }

    // wait for all copies to finish
    for (int i = 0; i < active_threads; ++i) {
      memcpy_helper_[i]->Wait();
    }
  }
#endif
}

void NodeExecutor::ExecuteNSVMMemcpyAsyncDeviceToHost(RHACCommand *cmd)
{
  assert(rhac_platform.IsHost());
  RHACCommandNSVMMemcpy *m_cmd = (RHACCommandNSVMMemcpy *)cmd;
  size_t copy_size = m_cmd->GetSize();

//  rhac_platform.FinishAllRequestQueueExcept(0);

#if defined(RHAC_MEMCPY_HELPER)
  if (copy_size < MEMCPY_CHUNK_SIZE) {
#endif
#if defined(RHAC_PREFETCH)
  RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
  rhac_driver->PrefetchToCPU(m_cmd->GetSource(), m_cmd->GetSize(), -1, true);
#endif
  SVMMemcpy(m_cmd);
#if defined(RHAC_MEMCPY_HELPER)
  }
  else {
    // spread jobs to workers
    int active_threads = std::min((int)
        std::ceil((double)copy_size / MEMCPY_CHUNK_SIZE), NUM_MEMCPY_THREAD);
    size_t size_per_threads = copy_size / active_threads;

    size_t mod = size_per_threads % MEMCPY_CHUNK_SIZE;
    if (mod != 0)
      size_per_threads += (MEMCPY_CHUNK_SIZE - mod);

    size_t next_dst = m_cmd->GetDestination();
    size_t next_src = m_cmd->GetSource();
    for (int i = 0; i < active_threads; ++i) {
      size_t actual_size = std::min(size_per_threads, copy_size);
      memcpy_helper_[i]->Enqueue(next_dst, next_src, actual_size, 1);
      copy_size -= actual_size;
      next_dst += actual_size;
      next_src += actual_size;
    }

    // wait for all copies to finish
    for (int i = 0; i < active_threads; ++i) {
      memcpy_helper_[i]->Wait();
    }
  }
#endif
}

void NodeExecutor::ExecuteNSVMMemset(RHACCommand *cmd)
{
  assert(rhac_platform.IsHost());
  RHACCommandNSVMMemset *m_cmd = (RHACCommandNSVMMemset *)cmd;

  rhac_platform.FinishAllRequestQueueExcept(0);

#if defined(RHAC_PREFETCH)
  RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
  rhac_driver->PrefetchToCPU(m_cmd->GetDestination(), m_cmd->GetSize(), -1, true);
#endif
  memset((void*)m_cmd->GetDestination(), m_cmd->GetValue(), m_cmd->GetSize());
}

void NodeExecutor::ExecuteNSVMMemsetAsync(RHACCommand *cmd)
{
  ExecuteNSVMMemsetAsync(cmd);
}

void NodeExecutor::ExecuteNSVMReserve(RHACCommand *cmd)
{
  RHACCommandNSVMReserve *n_cmd = (RHACCommandNSVMReserve *)cmd;

  size_t reserve_size = n_cmd->GetReserveSize();

  RHAC_LOG("Rank %d Handle NSVMReserve with size %zu",
      rhac_platform.GetRank(),
      reserve_size);

  ClusterSVM::ReserveSVM(reserve_size);
}

void NodeExecutor::ExecuteNSVMSync(RHACCommand *cmd)
{
  RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
  rhac_driver->Synchronize();
  RHAC_LOG("Node %d NSVM Sync Done", rhac_platform.GetRank());
}

void NodeExecutor::ExecuteNBarrier(RHACCommand *cmd)
{
  RHAC_LOG("Rank %d NBarrier(cmd id : %lu) Start ", 
      rhac_platform.GetRank(), cmd->GetCommandID());

  RHACCommandNBarrier *b_cmd = (RHACCommandNBarrier *)cmd;
  uint32_t num_wait = b_cmd->GetNumWait();
  b_cmd->ResetVargHead();
  
  for (uint32_t w = 0; w < num_wait; w++) {
    int node;
    int dev;
    rhac_command_id_t wait_id;

    b_cmd->PopWaitItem(&node, &dev, &wait_id);

    RHAC_LOG("Rank %d NBarrier(cmd id : %lu) wait id : %lu, dev : %d", 
        rhac_platform.GetRank(), b_cmd->GetCommandID(), wait_id, dev);
    rhac_platform.WaitResponse(wait_id, node, dev);
  }
}

void NodeExecutor::ExecuteNMemAdvise(RHACCommand *cmd) {
  RHAC_LOG("Rank %d NMemAdvise(cmd id : %lu) Start ",
      rhac_platform.GetRank(), cmd->GetCommandID());

  cudaError_t cuda_err;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  RHACCommandNMemAdvise *m_cmd;
  m_cmd = (RHACCommandNMemAdvise *)cmd;

  void *devPtr = (void *)(m_cmd->GetDevPtr());
  size_t count = m_cmd->GetCount();
  cudaMemoryAdvise advice = m_cmd->GetAdvice();
  int device = m_cmd->GetDevice();

  RHAC_LOG("Recv Rank %d NMemAdvise(cmd id : %lu) devPtr : %p, count %zu, advise : %d, device : %d",
      rhac_platform.GetRank(),
      m_cmd->GetCommandID(),
      devPtr, count, advice, device);

  // FIXME - cuda runtime's weired behavior
  cuda_err = libcuda->cudaSetDevice(0);
  CHECK_CUDART_ERROR(cuda_err);

  cuda_err = libcuda->cudaMemAdvise(devPtr, count, advice, 0);
  CHECK_CUDART_ERROR(cuda_err);
 
  RHAC_LOG("Rank %d NMemAdvise(cmd id : %lu) Done ",
      rhac_platform.GetRank(), cmd->GetCommandID());
}

void NodeExecutor::ExecuteNSplitVARange(RHACCommand *cmd) {
  RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
  RHACCommandNSplitVARange *m_cmd;
  m_cmd = (RHACCommandNSplitVARange *)cmd;

  uint64_t base = m_cmd->GetBase();
  size_t length = m_cmd->GetLength();

  size_t length_mod = length % SVM_ALIGNMENT;
  length = (length_mod == 0) ? length : length + SVM_ALIGNMENT - length_mod;

  rhac_driver->SplitVARange(base, length);
}

void NodeExecutor::ExecuteNSetDupFlag(RHACCommand *cmd) {
  RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
  RHACCommandNSetDupFlag *m_cmd;
  m_cmd = (RHACCommandNSetDupFlag *)cmd;

  uint64_t base = m_cmd->GetBase();
  size_t length = m_cmd->GetLength();
  bool turnon = m_cmd->GetFlag();

  if (turnon) {
    rhac_driver->SetDupFlag(base, length);
  }
  else {
    rhac_driver->UnsetDupFlag(base, length);
  }
}

void NodeExecutor::ExecuteNEventCreate(RHACCommand *cmd)
{
  // This function is only executed on Host Node's node executor
  assert(rhac_platform.IsHost());

  RHACCommandNEventCreate *e_cmd;
  cudaEvent_t *event;

  e_cmd = (RHACCommandNEventCreate *)cmd;

  event = e_cmd->GetEventPtr();

  // Create RHAC Event (cudaEventCreate is called internally)
  RHACEvent *rhac_event = new RHACEvent(event);

  rhac_platform.InsertEvent(*event, rhac_event);
}

void NodeExecutor::ExecuteNEventRecord(RHACCommand *cmd)
{
  // This function is only executed on Host Node's node executor
  assert(rhac_platform.IsHost());

  RHACCommandNEventRecord *e_cmd;
  cudaEvent_t event;
  cudaStream_t stream;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  cudaError_t cuda_err;

  e_cmd = (RHACCommandNEventRecord *)cmd;

  event = e_cmd->GetEvent();
  stream = e_cmd->GetStream();

//  cuda_err = libcuda->cudaEventRecord(event, stream);
  cuda_err = libcuda->cudaEventRecord(event, 0);
  CHECK_CUDART_ERROR(cuda_err);
}

// All Command
void NodeExecutor::ExecuteAGlobalBarrier(RHACCommand *cmd)
{
  RHAC_LOG("Rank %d Node AGlobalBarrier(cmd id : %lu) Start ", 
      rhac_platform.GetRank(), cmd->GetCommandID());

  rhac_command_id_t cmd_id;
  cmd_id = cmd->GetCommandID();

  *response_queue_ = cmd_id;

  rhac_platform.RhacBarrierWait();

  RHAC_LOG("Rank %d Node AGlobalBarrier(cmd id : %lu) Done ", 
      rhac_platform.GetRank(), cmd->GetCommandID());
}

// ====================================================================
// Device Executor class
DeviceExecutor::DeviceExecutor(volatile rhac_command_id_t* response_queue,
                               RequestQueue* request_queue,
                               int device_id)
  :Executor(response_queue, request_queue), device_id_(device_id) {
  CUresult error;
  CUdevice cu_device;
  CUcontext cu_context;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();

  error = libcuda->cuDeviceGet(&cu_device, device_id_);
  if (error != 0) {
    printf("Failed to get CUDA device %d (error = %d)\n", device_id_, error);
    return;
  }

  error = libcuda->cuDevicePrimaryCtxRetain(&cu_context, cu_device);
  if (error != 0) {
    printf("Failed to retain primary context of device %d (error = %d)\n",
        device_id_, error);
    return;
  }

  error = libcuda->cuCtxSetCurrent(cu_context);
  if (error != 0) {
    printf("Failed to set current context of device %d (error = %d)\n",
        device_id_, error);
    return;
  }

  int num_fatbins = fatbin_handler->GetNumFatbins();
  cu_modules_ = (CUmodule*)malloc(sizeof(CUmodule) * num_fatbins);
  char filename[256];

  for (int i = 0; i < num_fatbins; ++i) {
    sprintf(filename, "kernels.%d.fatbin", i+1);
    error = libcuda->cuModuleLoad(&cu_modules_[i], filename);

    if (error != 0) {
      printf("Failed to load \"%s\" for device %d (error = %d)\n",
          filename, device_id_, error);
      return;
    }
  }

#ifdef RHAC_PREFETCH
  prefetch_scheduler_ = new PrefetchScheduler(device_id_);
  prefetch_scheduler_->Run();

  prefetch_calculator_ = new PrefetchCalculator(rhac_platform.GetRank(),
      device_id_, prefetch_scheduler_);
  prefetch_calculator_->Run();
#endif
}

DeviceExecutor::~DeviceExecutor() {
#ifdef RHAC_PREFETCH
  prefetch_calculator_->Kill();
  prefetch_calculator_->Quit();
  prefetch_scheduler_->Kill();
  prefetch_scheduler_->Quit();
#endif
}

void DeviceExecutor::PrintStart()
{
  RHAC_LOG("Rank %d DeviceExecutor %d Start", rhac_platform.GetRank(), device_id_);
}

void DeviceExecutor::PrintEnd()
{
  RHAC_LOG("Rank %d DeviceExecutor %d End", rhac_platform.GetRank(), device_id_);
}

void DeviceExecutor::Execute(RHACCommand *cmd)
{
  enum CommandKind cmd_kind;
  cmd_kind = cmd->GetCommandKind();

  assert(cmd->IsDeviceCommand() || cmd->IsAllDeviceCommand() || cmd->IsAllCommand());

  switch(cmd_kind) {
    case DExit:
      break;
    case DSetDevice:
      ExecuteDSetDevice(cmd);
      break;
    case DReset:
      ExecuteDReset(cmd);
      break;
    case DSetCacheConfig:
      ExecuteDSetCacheConfig(cmd);
      break;
    case DFuncSetCacheConfig:
      ExecuteDFuncSetCacheConfig(cmd);
      break;
    case DMallocArray:
      ExecuteDMallocArray(cmd);
      break;
    case DFreeArray:
      ExecuteDFreeArray(cmd);
      break;
    case DBindTexture:
      ExecuteDBindTexture(cmd);
      break;
    case DBindTextureToArray:
      ExecuteDBindTextureToArray(cmd);
      break;
    case DKernelPartialExecution:
      ExecuteDKernelPartialExecution(cmd);
      break;
    case DEvent:
      break;
    case ADMemcpyToSymbol:
      ExecuteADMemcpyToSymbol(cmd);
      break;
    case ADMemcpy2DToArray:
      ExecuteADMemcpy2DToArray(cmd);
      break;
    case AGlobalBarrier:
      ExecuteAGlobalBarrier(cmd);
      break;
    default:
      RHAC_LOG(" Unimplemented cmd_kind = %d", cmd_kind);
      YD_TODO();
      break;
  }
}

void DeviceExecutor::ExecuteDSetDevice(RHACCommand *cmd)
{
  cudaError_t cuda_err;
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();

  cuda_err = libcuda->cudaSetDevice(device_id_);
  CHECK_CUDART_ERROR(cuda_err);

  RHAC_LOG("Rank %d Device %d %s", rhac_platform.GetRank(), device_id_, __func__);

  // FIXME
  cudaDeviceProp prop;
  libcuda->cudaGetDeviceProperties(&prop, device_id_);
  RHAC_LOG("Rank %d Device %d name : %s", rhac_platform.GetRank(), device_id_, prop.name);
}

void DeviceExecutor::ExecuteDReset(RHACCommand *cmd)
{
  cudaError_t cuda_err;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();

  cuda_err = libcuda->cudaDeviceReset();

  CHECK_CUDART_ERROR(cuda_err);

  RHAC_LOG("Rank %d Device %d %s", rhac_platform.GetRank(), device_id_, __func__);
}

void DeviceExecutor::ExecuteDSetCacheConfig(RHACCommand *cmd)
{
  RHACCommandDSetCacheConfig *c_cmd = (RHACCommandDSetCacheConfig *)cmd;
  cudaError_t cuda_err;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();

  cuda_err = libcuda->cudaDeviceSetCacheConfig(c_cmd->GetCacheConfig());
  CHECK_CUDART_ERROR(cuda_err);

  RHAC_LOG("Rank %d Device %d %s", rhac_platform.GetRank(), device_id_, __func__);
}

void DeviceExecutor::ExecuteDFuncSetCacheConfig(RHACCommand *cmd) {
  RHACCommandDFuncSetCacheConfig *m_cmd;
  int fatbin_index;
  char func_name[128];
  cudaFuncCache config;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();

  m_cmd = (RHACCommandDFuncSetCacheConfig *)cmd;

  fatbin_index = m_cmd->GetFatbinIndex();
  m_cmd->GetFuncName(func_name);
  config = m_cmd->GetCacheConfig();

  CUfunction kernel_function = GetCUfunction(fatbin_index, func_name);
  CUresult error;

  error = libcuda->cuFuncSetCacheConfig(kernel_function, (CUfunc_cache)config);
  if (error != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to set cache config of kernel in device %d (error = %d)\n",
        device_id_, error);
    return;
  }
}

void DeviceExecutor::ExecuteDMallocArray(RHACCommand *cmd) {
  RHACCommandDMallocArray *c_cmd = (RHACCommandDMallocArray *)cmd;
  cudaArray_t cuda_array;
  cudaChannelFormatDesc desc;
  size_t width;
  size_t height;
  unsigned int flags;
  cudaError_t cuda_err;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();

  desc = libcuda->cudaCreateChannelDesc(c_cmd->GetDescX(), c_cmd->GetDescY(),
      c_cmd->GetDescZ(), c_cmd->GetDescW(), c_cmd->GetDescFormat());
  width = c_cmd->GetWidth();
  height = c_cmd->GetHeight();
  flags = c_cmd->GetFlags();

  cuda_err = libcuda->cudaMallocArray(&cuda_array, &desc, width, height, flags);
  CHECK_CUDART_ERROR(cuda_err);

  // register returned cuda array to the list
  cuda_arrays_.push_back(cuda_array);
}

void DeviceExecutor::ExecuteDFreeArray(RHACCommand *cmd) {
  RHACCommandDFreeArray *c_cmd = (RHACCommandDFreeArray *)cmd;
  unsigned int array_index;
  cudaArray_t cuda_array;
  cudaError_t cuda_err;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();

  array_index = c_cmd->GetArrayIndex();
  cuda_array = cuda_arrays_[array_index];

  cuda_err = libcuda->cudaFreeArray(cuda_array);
  CHECK_CUDART_ERROR(cuda_err);
}

void DeviceExecutor::ExecuteDBindTexture(RHACCommand *cmd) {
  RHACCommandDBindTexture *m_cmd;
  int fatbin_index;
  char ref_name[128];
  int ref_filtermode;
  int ref_normalized;
  void *dev_ptr;
  int desc_x, desc_y, desc_z, desc_w;
  int channel_bitwidth, num_channels;
  unsigned int flags;
  cudaChannelFormatKind desc_format;
  int translated_format;
  size_t size;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();

  m_cmd = (RHACCommandDBindTexture *)cmd;

  fatbin_index = m_cmd->GetFatbinIndex();
  m_cmd->GetRefName(ref_name);
  ref_filtermode = m_cmd->GetRefFilterMode();
  ref_normalized = m_cmd->GetRefNormalized();
  dev_ptr = m_cmd->GetDevPtr();
  desc_x = m_cmd->GetDescX();
  desc_y = m_cmd->GetDescY();
  desc_z = m_cmd->GetDescZ();
  desc_w = m_cmd->GetDescW();
  desc_format = m_cmd->GetDescFormat();
  num_channels = 0;
  flags = 0;
  size = m_cmd->GetSize();

  CUtexref texref = GetCUtexref(fatbin_index, ref_name);
  CUresult error;

  error = libcuda->cuTexRefSetAddress(NULL, texref, (CUdeviceptr)dev_ptr, size);
  if (error != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to set address of texture in device %d (error = %d)\n",
        device_id_, error);
    return;
  }

  error = libcuda->cuTexRefSetFilterMode(texref, (CUfilter_mode)ref_filtermode);
  if (error != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to set filter mode of texture in device %d (error = %d)\n",
        device_id_, error);
    return;
  }

  if (ref_normalized) {
    flags |= CU_TRSF_NORMALIZED_COORDINATES;
  }

  if (desc_x) {
    channel_bitwidth = desc_x;
    num_channels += 1;
  }
  if (desc_y) {
    channel_bitwidth = desc_y;
    num_channels += 1;
  }
  if (desc_z) {
    channel_bitwidth = desc_z;
    num_channels += 1;
  }
  if (desc_w) {
    channel_bitwidth = desc_w;
    num_channels += 1;
  }

  switch (desc_format) {
  case cudaChannelFormatKindSigned:
    switch (channel_bitwidth) {
    case 8: translated_format = CU_AD_FORMAT_SIGNED_INT8; break;
    case 16: translated_format = CU_AD_FORMAT_SIGNED_INT16; break;
    case 32: translated_format = CU_AD_FORMAT_SIGNED_INT32; break;
    default: fprintf(stderr, "Unsupported channel bitwidth\n"); return;
    }
    flags |= CU_TRSF_READ_AS_INTEGER;
    break;
  case cudaChannelFormatKindUnsigned:
    switch (channel_bitwidth) {
    case 8: translated_format = CU_AD_FORMAT_UNSIGNED_INT8; break;
    case 16: translated_format = CU_AD_FORMAT_UNSIGNED_INT16; break;
    case 32: translated_format = CU_AD_FORMAT_UNSIGNED_INT32; break;
    default: fprintf(stderr, "Unsupported channel bitwidth\n"); return;
    }
    flags |= CU_TRSF_READ_AS_INTEGER;
    break;
  case cudaChannelFormatKindFloat:
    translated_format = CU_AD_FORMAT_FLOAT;
    break;
  case cudaChannelFormatKindNone:
    translated_format = 0;
    break;
  }

  if (translated_format) {
    error = libcuda->cuTexRefSetFormat(texref, (CUarray_format)translated_format, num_channels);
    if (error != CUDA_SUCCESS) {
      fprintf(stderr, "Failed to set format of texture in device %d (error = %d)\n",
          device_id_, error);
      return;
    }
  }

  if (flags) {
    error = libcuda->cuTexRefSetFlags(texref, flags);
    if (error != CUDA_SUCCESS) {
      fprintf(stderr, "Failed to set flags of texture in device %d (error = %d)\n",
          device_id_, error);
      return;
    }
  }
}

void DeviceExecutor::ExecuteDBindTextureToArray(RHACCommand *cmd) {
  RHACCommandDBindTextureToArray *m_cmd;
  int fatbin_index;
  char ref_name[128];
  int ref_filtermode;
  int ref_normalized;
  unsigned int array_index;
  cudaArray_t cuda_array;
  int desc_x, desc_y, desc_z, desc_w;
  int channel_bitwidth, num_channels;
  unsigned int flags;
  cudaChannelFormatKind desc_format;
  int translated_format;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();

  m_cmd = (RHACCommandDBindTextureToArray *)cmd;

  fatbin_index = m_cmd->GetFatbinIndex();
  m_cmd->GetRefName(ref_name);
  ref_filtermode = m_cmd->GetRefFilterMode();
  ref_normalized = m_cmd->GetRefNormalized();
  array_index = m_cmd->GetArrayIndex();
  cuda_array = cuda_arrays_[array_index];
  desc_x = m_cmd->GetDescX();
  desc_y = m_cmd->GetDescY();
  desc_z = m_cmd->GetDescZ();
  desc_w = m_cmd->GetDescW();
  desc_format = m_cmd->GetDescFormat();
  num_channels = 0;
  flags = 0;

  CUtexref texref = GetCUtexref(fatbin_index, ref_name);
  CUresult error;

  error = libcuda->cuTexRefSetArray(texref, (CUarray)cuda_array,
      CU_TRSA_OVERRIDE_FORMAT);
  if (error != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to set array of texture in device %d (error = %d)\n",
        device_id_, error);
    return;
  }

  error = libcuda->cuTexRefSetFilterMode(texref, (CUfilter_mode)ref_filtermode);
  if (error != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to set filter mode of texture in device %d (error = %d)\n",
        device_id_, error);
    return;
  }

  if (ref_normalized) {
    flags |= CU_TRSF_NORMALIZED_COORDINATES;
  }

  if (desc_x) {
    channel_bitwidth = desc_x;
    num_channels += 1;
  }
  if (desc_y) {
    channel_bitwidth = desc_y;
    num_channels += 1;
  }
  if (desc_z) {
    channel_bitwidth = desc_z;
    num_channels += 1;
  }
  if (desc_w) {
    channel_bitwidth = desc_w;
    num_channels += 1;
  }

  switch (desc_format) {
  case cudaChannelFormatKindSigned:
    switch (channel_bitwidth) {
    case 8: translated_format = CU_AD_FORMAT_SIGNED_INT8; break;
    case 16: translated_format = CU_AD_FORMAT_SIGNED_INT16; break;
    case 32: translated_format = CU_AD_FORMAT_SIGNED_INT32; break;
    default: fprintf(stderr, "Unsupported channel bitwidth\n"); return;
    }
    flags |= CU_TRSF_READ_AS_INTEGER;
    break;
  case cudaChannelFormatKindUnsigned:
    switch (channel_bitwidth) {
    case 8: translated_format = CU_AD_FORMAT_UNSIGNED_INT8; break;
    case 16: translated_format = CU_AD_FORMAT_UNSIGNED_INT16; break;
    case 32: translated_format = CU_AD_FORMAT_UNSIGNED_INT32; break;
    default: fprintf(stderr, "Unsupported channel bitwidth\n"); return;
    }
    flags |= CU_TRSF_READ_AS_INTEGER;
    break;
  case cudaChannelFormatKindFloat:
    translated_format = CU_AD_FORMAT_FLOAT;
    break;
  case cudaChannelFormatKindNone:
    translated_format = 0;
    break;
  }

  if (translated_format) {
    error = libcuda->cuTexRefSetFormat(texref, (CUarray_format)translated_format, num_channels);
    if (error != CUDA_SUCCESS) {
      fprintf(stderr, "Failed to set format of texture in device %d (error = %d)\n",
          device_id_, error);
      return;
    }
  }

  if (flags) {
    error = libcuda->cuTexRefSetFlags(texref, flags);
    if (error != CUDA_SUCCESS) {
      fprintf(stderr, "Failed to set flags of texture in device %d (error = %d)\n",
          device_id_, error);
      return;
    }
  }
}

void DeviceExecutor::ExecuteDKernelPartialExecution(RHACCommand *cmd)
{
  FunctionInfo *func_info;
  std::vector<VarInfo*> *var_infos;
  int fatbin_index;
  char func_name[128];
  dim3 gridDim, blockDim;
  size_t dynamic_smem;
  int num_args, num_global_vars;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();

  RHACCommandDKernelPartialExecution *k_cmd;
  k_cmd = (RHACCommandDKernelPartialExecution *)cmd;

  k_cmd->ResetVargHead(); 
  fatbin_index = k_cmd->GetFatbinIndex();
  k_cmd->GetFuncName(func_name);
  gridDim = k_cmd->GetGridDim();
  RHAC_LOG(" gridDim x : %d, y : %d", gridDim.x, gridDim.y);
  blockDim = k_cmd->GetBlockDim();
  dynamic_smem = k_cmd->GetSharedMem();

  // get kernel info
  func_info = rhac_platform.GetKernelInfo(fatbin_index, func_name);
  if (func_info == NULL) {
    fprintf(stderr, "Failed to find function info of \"%s\" - Rank %d \n", func_name, rhac_platform.GetRank());
    return;
  }

#if defined(DEBUG_NV_UVM_DRIVER)
  RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
  rhac_driver->ResetFaultStatus(device_id_);
#endif

  // get var infos
  var_infos = rhac_platform.GetVarInfos(fatbin_index);
  if (var_infos == NULL) {
    num_global_vars = 0;
  }
  else {
    num_global_vars = var_infos->size();
  }

  num_args = func_info->num_args;
  void* args[num_args + num_global_vars + 6];
  int arg_index = 0;

  // original kernel arguments
  for (int i = 0; i < num_args; ++i) {
    int arg_size = func_info->arg_sizes[i];
    args[arg_index] = k_cmd->PopArgPtr((size_t)arg_size);
    arg_index += 1;
  }

  for (int i = 0; i < num_global_vars; ++i) {
    args[arg_index] = &(var_infos->at(i)->dev_ptr);
    arg_index += 1;
  }

  int block_bound_index = arg_index;
  // single image variables
  for (int i = 0; i < 6; ++i) {
    args[arg_index] = k_cmd->PopArgPtr(sizeof(unsigned));
    arg_index += 1;
  }

#if defined(RHAC_PREFETCH)
  unsigned int num_buffers = k_cmd->PopArg<size_t>();

  unsigned int num_one_thread_expressions = 0;
  for (unsigned int i = 0; i < num_buffers; ++i) {
    size_t kernel_arg_index = k_cmd->PopArg<size_t>();
    size_t buffer_bound = k_cmd->PopArg<size_t>();
    int64_t gx_coeff = k_cmd->PopArg<int64_t>();
    int64_t gy_coeff = k_cmd->PopArg<int64_t>();
    int64_t gz_coeff = k_cmd->PopArg<int64_t>();
    int64_t lx_coeff = k_cmd->PopArg<int64_t>();
    int64_t ly_coeff = k_cmd->PopArg<int64_t>();
    int64_t lz_coeff = k_cmd->PopArg<int64_t>();
    int64_t i0_bound = k_cmd->PopArg<int64_t>();
    int64_t i0_step = k_cmd->PopArg<int64_t>();
    int64_t i1_bound = k_cmd->PopArg<int64_t>();
    int64_t i1_step = k_cmd->PopArg<int64_t>();
    int64_t const_var = k_cmd->PopArg<int64_t>();
    size_t fetch_size = k_cmd->PopArg<size_t>();
    bool one_thread = k_cmd->PopArg<size_t>();

    if (one_thread &&
        (rhac_platform.GetRank() != 0 || device_id_ != 0)) {
      // only GPU0 will prefetch for this expression
      num_one_thread_expressions += 1;
      continue;
    }

    uint64_t buf = *((uint64_t*)args[kernel_arg_index]);
    prefetch_calculator_->Calculate(i, buf, buffer_bound, gridDim, blockDim,
        dim3(
          *(unsigned *)args[block_bound_index+0],
          *(unsigned *)args[block_bound_index+2],
          *(unsigned *)args[block_bound_index+4]),
        dim3(
          *(unsigned *)args[block_bound_index+1],
          *(unsigned *)args[block_bound_index+3],
          *(unsigned *)args[block_bound_index+5]),
        gx_coeff, gy_coeff, gz_coeff,
        lx_coeff, ly_coeff, lz_coeff,
        i0_bound, i0_step, i1_bound, i1_step, const_var, fetch_size);
  }
  prefetch_scheduler_->SetNumExpressions(num_buffers-num_one_thread_expressions);

  prefetch_calculator_->WakeUp();
  prefetch_calculator_->Wait();
  prefetch_scheduler_->Wait();
#endif

  RHAC_LOG("Rank %d dev %d cmd id %lu kernel %s block x %u ~ %u, y %u ~ %u, z %u ~ %u",
      rhac_platform.GetRank(),
      device_id_,
      k_cmd->GetCommandID(),
      func_name,
      *(unsigned *)args[block_bound_index+0],
      *(unsigned *)args[block_bound_index+1],
      *(unsigned *)args[block_bound_index+2],
      *(unsigned *)args[block_bound_index+3],
      *(unsigned *)args[block_bound_index+4],
      *(unsigned *)args[block_bound_index+5]);

  CUfunction kernel_function = GetCUfunction(fatbin_index, func_name);

  CUresult error = libcuda->cuLaunchKernel(kernel_function,
      gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
      dynamic_smem, 0, args, NULL);
  if (error != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to launch kernel in device %d (error = %d)\n",
        device_id_, error);
    return;
  }

  // wait till kernel finishes
  CHECK_CUDART_ERROR(libcuda->cudaDeviceSynchronize());

#if defined(RHAC_PREFETCH)
//  prefetch_calculator_->Wait();
#endif

#if defined(DEBUG_NV_UVM_DRIVER)
  printf("Printing fault status of GPU%d for \"%s\" kernel\n",
      device_id_, func_name);
  rhac_driver->PrintFaultStatus(device_id_);
#endif

}

void DeviceExecutor::ExecuteADMemcpyToSymbol(RHACCommand *cmd)
{
  RHACCommandADMemcpyToSymbol *m_cmd;
  int fatbin_index;
  char symbol_name[128];
  void *src;
  size_t count, offset;
  cudaMemcpyKind kind;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();

  m_cmd = (RHACCommandADMemcpyToSymbol *)cmd;

  fatbin_index = m_cmd->GetFatbinIndex();
  m_cmd->GetSymbolName(symbol_name);
  kind = m_cmd->GetMemcpyKind();
  src = (void*)(m_cmd->GetSource());
  count = m_cmd->GetCount();
  offset = m_cmd->GetOffset();

  RHAC_LOG("recv fatbin idx : %d, symbol %s, src : %p, count : %zu, offset : %zu",
      fatbin_index, symbol_name, src, count, offset);

  CUdeviceptr symbol_ptr = GetCUdeviceptr(fatbin_index, symbol_name);
  CUresult error;

  switch (kind) {
  case cudaMemcpyHostToDevice:
    error = libcuda->cuMemcpyHtoD_v2(symbol_ptr + offset, src, count);
    break;
  case cudaMemcpyDeviceToHost:
    error = libcuda->cuMemcpyDtoH_v2(src, symbol_ptr + offset, count);
    break;
  default:
    fprintf(stderr, "Unsupported kind for cudaMempcyToSymbol\n");
    return;
    break;
  }
  if (error != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to copy symbol in device %d (error = %d)\n",
        device_id_, error);
    return;
  }
  //CHECK_CUDART_ERROR(libcuda->cudaMemcpyToSymbol(symbol, src, count, offset, kind));
}

void DeviceExecutor::ExecuteADMemcpy2DToArray(RHACCommand *cmd)
{
  RHACCommandADMemcpy2DToArray *c_cmd = (RHACCommandADMemcpy2DToArray *)cmd;
  unsigned int array_index;
  cudaArray_t cuda_array;
  size_t w_offset, h_offset, s_pitch, width, height;
  void *src;
  cudaMemcpyKind kind;
  cudaError_t cuda_err;
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();

  array_index = c_cmd->GetArrayIndex();
  cuda_array = cuda_arrays_[array_index];
  w_offset = c_cmd->GetWOffset();
  h_offset = c_cmd->GetHOffset();
  src = c_cmd->GetSource();
  s_pitch = c_cmd->GetSPitch();
  width = c_cmd->GetWidth();
  height = c_cmd->GetHeight();
  kind = c_cmd->GetMemcpyKind();

  cuda_err = libcuda->cudaMemcpy2DToArray(cuda_array, w_offset, h_offset, src,
      s_pitch, width, height, kind);
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "Failed to copy to array in device %d (error = %d)\n",
        device_id_, cuda_err);
    return;
  }
}

void DeviceExecutor::ExecuteAGlobalBarrier(RHACCommand *cmd)
{
  RHAC_LOG("Rank %d device %d AGlobalBarrier(cmd id : %lu) Start ", 
      rhac_platform.GetRank(), device_id_, cmd->GetCommandID());

  rhac_command_id_t cmd_id;
  cmd_id = cmd->GetCommandID();

  assert(*response_queue_ < cmd_id);
  *response_queue_ = cmd_id;

  rhac_platform.RhacBarrierWait();

  RHAC_LOG("Rank %d device %d AGlobalBarrier(cmd id : %lu) Done ", 
      rhac_platform.GetRank(), device_id_, cmd->GetCommandID());
}


CUfunction DeviceExecutor::GetCUfunction(int fatbin_index,
    const char *func_name)
{
  CUresult error;
  CUfunction kernel_function;
  std::map<std::pair<int, std::string>, CUfunction>::iterator MI;
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();

  MI = cuda_function_map_cu_.find(std::make_pair(fatbin_index, func_name));

  // no function found, make new one
  if (MI == cuda_function_map_cu_.end()) {
    error = libcuda->cuModuleGetFunction(&kernel_function,
        cu_modules_[fatbin_index-1], func_name);
    if (error != CUDA_SUCCESS) {
      fprintf(stderr, "Failed to get function \"%s\" from fatbin %d from module in device %d (error = %d)\n",
          func_name, fatbin_index, device_id_, error);
      return kernel_function;
    }
    cuda_function_map_cu_[std::make_pair(fatbin_index, func_name)] =
      kernel_function;
  }
  else {
    kernel_function = MI->second;
  }

  return kernel_function;
}

CUdeviceptr DeviceExecutor::GetCUdeviceptr(int fatbin_index,
    const char *symbol_name)
{
  CUresult error;
  CUdeviceptr symbol_ptr;
  std::map<std::pair<int, std::string>, CUdeviceptr>::iterator MI;
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();

  MI = cuda_var_map_cu_.find(std::make_pair(fatbin_index, symbol_name));

  // no symbol found, make new one
  if (MI == cuda_var_map_cu_.end()) {
    error = libcuda->cuModuleGetGlobal(&symbol_ptr, NULL,
        cu_modules_[fatbin_index-1], symbol_name);
    if (error != CUDA_SUCCESS) {
      fprintf(stderr, "Failed to get global \"%s\" from module in device %d (error = %d)\n",
          symbol_name, device_id_, error);
      return symbol_ptr;
    }
    cuda_var_map_cu_[std::make_pair(fatbin_index, symbol_name)] =
      symbol_ptr;
  }
  else {
    symbol_ptr = MI->second;
  }

  return symbol_ptr;
}

CUtexref DeviceExecutor::GetCUtexref(int fatbin_index,
    const char *ref_name)
{
  CUresult error;
  CUtexref tex_ref;
  std::map<std::pair<int, std::string>, CUtexref>::iterator MI;
  LibCUDA* libcuda = LibCUDA::GetLibCUDA();

  MI = cuda_tex_map_cu_.find(std::make_pair(fatbin_index, ref_name));

  // no symbol found, make new one
  if (MI == cuda_tex_map_cu_.end()) {
    error = libcuda->cuModuleGetTexRef(&tex_ref,
        cu_modules_[fatbin_index-1], ref_name);
    if (error != CUDA_SUCCESS) {
      fprintf(stderr, "Failed to get texture reference \"%s\" from module in device %d (error = %d)\n",
          ref_name, device_id_, error);
      return tex_ref;
    }
    cuda_tex_map_cu_[std::make_pair(fatbin_index, ref_name)] = tex_ref;
  }
  else {
    tex_ref = MI->second;
  }

  return tex_ref;
}
