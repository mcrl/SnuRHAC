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

#ifndef __RHAC_PLATFORM_H__
#define __RHAC_PLATFORM_H__

#include "rhac.h"
#include "utils.h"
#include "clusterSVM.h"

#include <map>
#include <vector>
#include <stdint.h>
#include <pthread.h>

class Platform {
  public:
    Platform();
    ~Platform();

    bool IsHost();
    int GetRank();
    int GetNodeNumDevices();
    int GetNumDevicesIn(int node);
    int GetClusterNumNodes();
    int GetClusterNumDevices();
    int GetClusterDeviceIndex(int node, int dev);
    rhac_command_id_t GenerateCommandID();
    rhac_command_id_t GenerateCommandID(int n);

    volatile rhac_command_id_t* GetResponseQueue(int node);
    volatile rhac_command_id_t* GetResponseQueue(int node, int dev);

    bool QueryResponse(rhac_command_id_t wait, int node);
    bool QueryResponse(rhac_command_id_t wait, int node, int dev);
    void WaitResponse(rhac_command_id_t wait, int node);
    void WaitResponse(rhac_command_id_t wait, int node, int dev);

    RequestQueue* GetRequestQueue(int node);
    RequestQueue* GetRequestQueue(int node, int dev);

    void EnqueueCommand(RHACCommand *cmd);
    void EnqueueCommand(RHACCommand *cmd, int node);
    void EnqueueCommand(RHACCommand *cmd, int node, int dev);

    // TODO
    void FinishRequestQueue(int node);
    void FinishRequestQueue(int node, int dev);
    void FinishAllRequestQueue();
    void FinishAllRequestQueueExcept(int node);
    void FinishAllRequestQueueExcept(int node, int dev);

    void InsertEvent(cudaEvent_t cuda_event, RHACEvent *rhac_event);
    void EraseEvent(cudaEvent_t cuda_event);
    RHACEvent* GetEvent(cudaEvent_t cuda_event);

    void ReadKernelInfoFile();
    void ReadVarInfoFile();
    FunctionInfo* GetKernelInfo(int fatbin_index, const char *kernel_name);
    std::vector<VarInfo*>* GetVarInfos(int fatbin_index);

    void SetCudaCacheConfig(cudaFuncCache cuda_cache_config);
    cudaFuncCache GetCudaCacheConfig();

    unsigned int GetNextArrayIndex();

#ifdef READDUP_FLAG_CACHING
    bool IsInReadDupList(uint64_t base);
    void AddToReadDupList(uint64_t base);
    void RemoveFromReadDupList(uint64_t base);
    void ClearReadDupList();

    std::vector<uint64_t>::iterator readdup_begin() { return readdup_bufs_.begin(); }
    std::vector<uint64_t>::iterator readdup_end() { return readdup_bufs_.end(); }
#endif

    // Barrier Functions
    void RhacBarrierWait();
    void RhacBarrierRegister(RHACBarrier *b);
    bool RhacBarrierGet(RHACBarrier **b);
    void RhacBarrierDeleteFront();
    void RhacBarrierEnqueue();

  private:
    void CreateResponseQueue();
    void CreateRequestQueues();
    void AllocGlobalVars();

    int size_;                  // MPI COMM WORLD SIZE
    int rank_;                  // MPI COMM WORLD RANK
    int &num_nodes_ = size_;    // 

    int num_devices_ = -1;          // Number of devices 
    int cluster_num_devices_ = -1;  // Number of devices in cluster
    int *num_devices_per_node_ = NULL;  // Number of devices list in cluster
    int *cluster_device_offsets_;   // offset - just for host

    std::vector<volatile rhac_command_id_t*> node_response_queues_;
    std::vector<volatile rhac_command_id_t*> device_response_queues_;

    std::vector<RequestQueue*>     node_request_queues_;
    std::vector<RequestQueue*>     device_request_queues_;

    // Barrier queue (host thread <->transmitter)
    QueueSPSC<RHACBarrier>* barrier_queue_;

    // mutex for Commmand ID
    mutex_t                               mutex_command_id_;
    rhac_command_id_t                     command_id_cnt_ = 1;

    // RHAC Driver Handler
    RHACDriver                            *rhac_driver_;

    // SVM
    void                                  *node_svm_ptr_;
    void                                  *cluster_svm_ptr_;
    ClusterSVM                            *cluster_svm_;

    // Executors
    NodeExecutor                          *node_executor_;
    //FIXME
  public:
    std::vector<DeviceExecutor*>          device_executors_;
  private:
    Transmitter*                          transmitter_;
    Receiver*                             receiver_;

    // CUDA libraries' singlton pointer
    LibCUDA *libcuda_;

    std::map<std::pair<int, std::string>, FunctionInfo*> func_info_map;
    std::map<int, std::vector<VarInfo*>> var_info_map;

    // default value for cache config
    cudaFuncCache cuda_cache_config_ = cudaFuncCachePreferNone;

    unsigned int array_index_;

#ifdef READDUP_FLAG_CACHING
    std::vector<uint64_t> readdup_bufs_;
#endif

    // RHAC Events 
    std::map<cudaEvent_t, RHACEvent *> event_map_;

    // barrier for rhac threads 
    pthread_barrier_t rhac_barrier_;
};

#endif // __RHAC_PLATFORM_H__
