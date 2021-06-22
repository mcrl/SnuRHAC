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

#ifndef __RHAC_EXECUTOR_H__
#define __RHAC_EXECUTOR_H__

#include "rhac.h"
#include "thread.h"
#include "utils.h"
#include "memcpy_helper.h"
#include "prefetch_calculator.h"
#include "prefetch_scheduler.h"

#include <cuda.h>
#include <map>

class Executor : public Thread {
  public:
    Executor(volatile rhac_command_id_t* response_queue,
             RequestQueue* request_queue);
    ~Executor();
    void run_();
    bool CheckQuit(RHACCommand *cmd);

    virtual void Execute(RHACCommand *cmd) = 0;
    virtual void PrintStart() = 0;
    virtual void PrintEnd() = 0;

  protected:
    volatile rhac_command_id_t* response_queue_;
    RequestQueue* request_queue_;
};

class NodeExecutor : public Executor {
  public:
    NodeExecutor(volatile rhac_command_id_t *response_queue_,
                 RequestQueue *request_queue);
    ~NodeExecutor();

    void Execute(RHACCommand *cmd);
    void ExecuteNSVMMemcpyHostToDevice(RHACCommand *cmd);
    void ExecuteNSVMMemcpyDeviceToHost(RHACCommand *cmd);
    void ExecuteNSVMMemcpyHostToHost(RHACCommand *cmd);
    void ExecuteNSVMMemcpyDeviceToDevice(RHACCommand *cmd);
    void ExecuteNSVMMemcpyAsyncHostToDevice(RHACCommand *cmd);
    void ExecuteNSVMMemcpyAsyncDeviceToHost(RHACCommand *cmd);
    void ExecuteNSVMMemset(RHACCommand *cmd);
    void ExecuteNSVMMemsetAsync(RHACCommand *cmd);
    void ExecuteNSVMReserve(RHACCommand *cmd);
    void ExecuteNSVMSync(RHACCommand *cmd);
    void ExecuteNBarrier(RHACCommand *cmd);
    void ExecuteNMemAdvise(RHACCommand *cmd);
    void ExecuteNSplitVARange(RHACCommand *cmd);
    void ExecuteNSetDupFlag(RHACCommand *cmd);
    void ExecuteNEventCreate(RHACCommand *cmd);
    void ExecuteNEventRecord(RHACCommand *cmd);

    void ExecuteAGlobalBarrier(RHACCommand *cmd);

    void PrintStart();
    void PrintEnd();

  private:
    void SVMMemcpy(RHACCommandNSVMMemcpy *m_cmd);
#ifdef RHAC_MEMCPY_HELPER
    MemcpyHelper* memcpy_helper_[NUM_MEMCPY_THREAD];
#endif
};

class DeviceExecutor : public Executor {
  public:
    DeviceExecutor(volatile rhac_command_id_t *response_queue,
                   RequestQueue *request_queue,
                   int device_id);
    ~DeviceExecutor();

    void Execute(RHACCommand *cmd);
    void PrintStart();
    void PrintEnd();

    void ExecuteDSetDevice(RHACCommand *cmd);
    void ExecuteDReset(RHACCommand *cmd);
    void ExecuteDSetCacheConfig(RHACCommand *cmd);
    void ExecuteDFuncSetCacheConfig(RHACCommand *cmd);
    void ExecuteDMallocArray(RHACCommand *cmd);
    void ExecuteDFreeArray(RHACCommand *cmd);
    void ExecuteDBindTexture(RHACCommand *cmd);
    void ExecuteDBindTextureToArray(RHACCommand *cmd);
    void ExecuteDKernelPartialExecution(RHACCommand *cmd);

    void ExecuteADMemcpyToSymbol(RHACCommand *cmd);
    void ExecuteADMemcpy2DToArray(RHACCommand *cmd);

    void ExecuteAGlobalBarrier(RHACCommand *cmd);

    CUfunction GetCUfunction(int fatbin_index, const char *func_name);
    CUdeviceptr GetCUdeviceptr(int fatbin_index, const char *symbol_name);
    CUtexref GetCUtexref(int fatbin_index, const char *ref_name);
    CUsurfref GetCUsurfref(int fatbin_index, const char *ref_name);
    cudaArray_t GetCUDAArray(unsigned int index) { return cuda_arrays_[index]; }

  private:
    int device_id_;
    CUmodule* cu_modules_;
    std::map<std::pair<int, std::string>, CUfunction> cuda_function_map_cu_;
    std::map<std::pair<int, std::string>, CUdeviceptr> cuda_var_map_cu_;
    std::map<std::pair<int, std::string>, CUtexref> cuda_tex_map_cu_;
    std::map<std::pair<int, std::string>, CUsurfref> cuda_surf_map_cu_;
    std::vector<cudaArray_t> cuda_arrays_;
#ifdef RHAC_PREFETCH
    PrefetchScheduler* prefetch_scheduler_;
    PrefetchCalculator* prefetch_calculator_;
#endif
};

#endif // __RHAC_EXECUTOR_H__
