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

#include "platform.h"
#include "libcuda.h"
#include "libmapa.h"
#include "utils.h"
#include "executor.h"
#include "transmitter.h"
#include "receiver.h"
#include "communicator.h"
#include "rhac_command.h"
#include "rhac_response.h"
#include "clusterSVM.h"
#include "rhac_driver.h"
#include "fatbin_handler.h"
#include "rhac_barrier.h"

#include <mpi.h>
#include <stdint.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <pthread.h>

Platform rhac_platform;

Platform::Platform() 
{
  int dummy;
  /* MPI Initialization */
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &dummy);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  RHAC_LOG("Comm size : %d, Comm rank : %d start", size_, rank_);

  /* get singletons */
  libcuda_ = LibCUDA::GetLibCUDA();
#if defined(RHAC_PREFETCH)
  LibMAPA::GetLibMAPA();
#endif

  /* Handle Binary File */
  FatbinHandler *fatbin_handler = FatbinHandler::GetFatbinHandler();
  int num_fatbins;

  if (IsHost()) {
    fatbin_handler->CreateWrapper();
    num_fatbins = fatbin_handler->GetNumFatbins();
  }
  MPI_Bcast(&num_fatbins, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!IsHost()) {
    fatbin_handler->SetNumFatbins(num_fatbins);
  }

  /* Read Kernel Info file */
  ReadKernelInfoFile();

  ReadVarInfoFile();

  /* Create NodeSVMPtr */
  node_svm_ptr_ = ClusterSVM::CreateNodeSVMPtr(DEFAULT_MALLOC_MANAGED_SIZE);

  /* Create ClusterSVMPtr */
  cluster_svm_ptr_ = ClusterSVM::CreateClusterSVMPtr(node_svm_ptr_);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Load Driver */
#ifdef DEBUG_NV_UVM_DRIVER
  rhac_driver_ = RHACDriver::GetRHACDriver();
  rhac_driver_->LoadNVDriver();
#endif

  rhac_driver_ = RHACDriver::GetRHACDriver(); 
  rhac_driver_->LoadDriver();
  MPI_Barrier(MPI_COMM_WORLD);

  /* Set ClusterSVM - notify to driver */
  rhac_driver_->InitClusterSVM(cluster_svm_ptr_, size_, rank_,
      GetNodeNumDevices());
  MPI_Barrier(MPI_COMM_WORLD);

  /* Reserve ClusterSVM First with the default size */
  ClusterSVM::ReserveSVM(DEFAULT_CLUSTERSVM_RESERVE_SIZE);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Create Cluster SVM */
  if (IsHost()) {
    cluster_svm_ = ClusterSVM::GetClusterSVM(cluster_svm_ptr_, 
                                             DEFAULT_CLUSTERSVM_RESERVE_SIZE);
  }
  
  /* Get Num Devices */
  num_devices_ = GetNodeNumDevices();
  RHAC_LOG("Rank %d - devs %d", rank_, num_devices_);

  /* Gather Num Devices in cluster */
  if (IsHost()) {
    num_devices_per_node_ = new int[size_];
    cluster_device_offsets_ = new int[size_];
  }
  else
    cluster_device_offsets_ = new int;

  GetClusterNumDevices();

  /* Create Queues */
  CreateResponseQueue();
  CreateRequestQueues();


  /* RHAC Barrier */
  if (num_nodes_ > 1) {
    // device threads + node thread + transmitter/receiver
    pthread_barrier_init(&rhac_barrier_, NULL, num_devices_ + 2);
  }
  else {
    // device threads + node thread
    pthread_barrier_init(&rhac_barrier_, NULL, num_devices_ + 1);
  }

  /* Create Executors */
  // FIXME Queues
  node_executor_ = new NodeExecutor(node_response_queues_[0],
                                    node_request_queues_[0]);

  for (int d = 0; d < num_devices_; d++) {
    DeviceExecutor* device_executor;
    device_executor = new DeviceExecutor(device_response_queues_[d],
                                         device_request_queues_[d],
                                         d);
    device_executors_.push_back(device_executor);
  }

  /* Run Executors */
  node_executor_->Run();
  for (unsigned int e = 0; e < device_executors_.size(); e++) {
    device_executors_[e]->Run();
  }

  /* Run Reciver */
  if (!IsHost()) {
    receiver_ = new Receiver();
    // remote process becomes a receiver
    receiver_->run_();
    MPI_Finalize();
    rhac_driver_->UnloadDriver();

    pthread_barrier_destroy(&rhac_barrier_);

    exit(0);
  }

  array_index_ = 0;

  /* Create and Run Transmitter and Receiver */
  if (size_ > 1) {
    // CAUTION : Queues for remote nodes are only one
    /* all queues to remote node share ONE queue */
    transmitter_ = new Transmitter(node_request_queues_[1]);
    transmitter_->Run();

    // barrier queue
    barrier_queue_ = new QueueSPSC<RHACBarrier>(QUEUE_SIZE);
  }

  // This forces the transmitter and device executor is running 
  /* Send Set device Command */
  for (int n = 0; n < num_nodes_; n++) {
    for (int d = 0; d < GetNumDevicesIn(n); d++) {
      RHACCommand *cmd;
      rhac_command_id_t cmd_id = GenerateCommandID();
      cmd = new RHACCommand();
      cmd->SetDefaultInfo(cmd_id, DSetDevice, n, d);
      EnqueueCommand(cmd);
      WaitResponse(cmd_id, n, d);
    }
  }

  AllocGlobalVars();
}

Platform::~Platform()
{
  // FIXME
  if (IsHost()) {

    RHACCommand *cmd;
    rhac_command_id_t cmd_id;
    for (int n = 0; n < num_nodes_; n++) {
      cmd = new RHACCommand();
      cmd->SetDefaultInfo(GenerateCommandID(), NExit, n, 0);
      cmd_id = cmd->GetCommandID();
      EnqueueCommand(cmd);
      RHAC_LOG("Host WaitResponse node %d", n);
      WaitResponse(cmd_id, n); 

      for (int d = 0; d < num_devices_per_node_[n]; d++) {
        cmd = new RHACCommand();
        cmd->SetDefaultInfo(GenerateCommandID(), DExit, n, d);
        cmd_id = cmd->GetCommandID();
        EnqueueCommand(cmd);
        RHAC_LOG("Host WaitResponse node %d dev %d ", n, d);
        WaitResponse(cmd_id, n, d); 
      }
    }

    if (num_nodes_ > 1) {
      transmitter_->Kill();
      transmitter_->Quit();
    }

    MPI_Finalize();
    rhac_driver_->UnloadDriver();
#ifdef DEBUG_NV_UVM_DRIVER
    rhac_driver_->UnloadNVDriver();
#endif
    /* destroy rhac barrier */
    pthread_barrier_destroy(&rhac_barrier_);

    exit(0);
  }


  if (IsHost()) {
    delete[] num_devices_per_node_;
    delete[] cluster_device_offsets_;
  }
  else
    delete cluster_device_offsets_;

  /* delete queues */
  delete node_response_queues_[0];
  delete node_request_queues_[0];
  
  for (int d = 0; d < num_devices_; d++) {
    delete device_request_queues_[d];
    delete device_response_queues_[d];
  }

  if (IsHost()) {
    // only use one request queue for all remote nodes
    if (size_ > 1) {
      delete node_request_queues_[1];
      delete barrier_queue_;
    }

    /* delete response queues */
    for (int n = 1; n < num_nodes_; n++) {
      delete node_response_queues_[n];
      for (int d = 0; d < num_devices_per_node_[n]; d++) {
        delete device_response_queues_[d];
      }
    }
  }
}

bool Platform::IsHost()
{
  return (rank_ == HOST_NODE) ? true : false;
}

int Platform::GetRank()
{
  return rank_;
}

int Platform::GetNodeNumDevices()
{
  int ret;

  if (num_devices_ < 0) {
    cudaError_t err;
    err = libcuda_->cudaGetDeviceCount(&ret);
    CHECK_CUDART_ERROR(err);
  }
  else {
    ret = num_devices_;
  }

  return ret;
}

int Platform::GetNumDevicesIn(int node)
{
  return num_devices_per_node_[node];
}

int Platform::GetClusterNumNodes()
{
  return num_nodes_;
}

// Remote should call this only once at initialization
int Platform::GetClusterNumDevices()
{
  if (cluster_num_devices_ > 0)
    return cluster_num_devices_;

  int num_devices = GetNodeNumDevices();
  int err;

  err = MPI_Gather(&num_devices, 1, MPI_INT, 
                      num_devices_per_node_, 1, MPI_INT, 
                      HOST_NODE,
                      MPI_COMM_WORLD);
  CHECK_MPI_ERROR(err);

  // get cluster offset
  if (IsHost()) {
    cluster_num_devices_ = 0;
    for (int n = 0; n < size_; n++) {
      cluster_device_offsets_[n] = cluster_num_devices_;
      cluster_num_devices_ += num_devices_per_node_[n];
    }

    // FIXME
    for (int n = 0; n < size_; n++) {
      RHAC_LOG("Host : node %d cluster_offset %d (%d)",
          n, cluster_device_offsets_[n], 
          num_devices_per_node_[n]);
    }
    RHAC_LOG("Host : total %d", cluster_num_devices_);
  }
  else {
    cluster_device_offsets_[0] = 0;
  }

  return cluster_num_devices_;
}

int Platform::GetClusterDeviceIndex(int node, int dev)
{
  assert(IsHost());

  return cluster_device_offsets_[node]+dev;
}

rhac_command_id_t Platform::GenerateCommandID()
{
  rhac_command_id_t ret;
  mutex_command_id_.lock();
  ret = command_id_cnt_++;
  mutex_command_id_.unlock();

  return ret;
}

rhac_command_id_t Platform::GenerateCommandID(int n)
{
  rhac_command_id_t ret;
  mutex_command_id_.lock();
  ret = command_id_cnt_;
  command_id_cnt_ += n;
  mutex_command_id_.unlock();

  return ret;
}

// =====================================================================
// functions related to queue
volatile rhac_command_id_t* Platform::GetResponseQueue(int node)
{
  if (!IsHost())
    node = 0;

  return node_response_queues_[node];
}

volatile rhac_command_id_t* Platform::GetResponseQueue(int node, int dev)
{
  if (!IsHost())
    node = 0;

  int cluster_dev_idx;
  cluster_dev_idx = cluster_device_offsets_[node] + dev;

  return device_response_queues_[cluster_dev_idx];
}

bool Platform::QueryResponse(rhac_command_id_t wait, int node)
{
  return *GetResponseQueue(node) >= wait;
}

bool Platform::QueryResponse(rhac_command_id_t wait, int node, int dev)
{
  return *GetResponseQueue(node, dev) >= wait;
}

void Platform::WaitResponse(rhac_command_id_t wait, int node)
{
  RHAC_LOG("Wait Response start with %zu < Wait(%zu) - node %d",
      *GetResponseQueue(node), wait, node);

  while (*GetResponseQueue(node) < wait);

  RHAC_LOG("Wait Response End with %zu < Wait(%zu) - node %d",
      *GetResponseQueue(node), wait, node);
}

void Platform::WaitResponse(rhac_command_id_t wait, int node, int dev)
{
  RHAC_LOG("Wait Response start with %zu < Wait(%zu) - node %d dev %d",
      *GetResponseQueue(node, dev), wait, node, dev);

  while (*GetResponseQueue(node, dev) < wait);

  RHAC_LOG("Wait Response End with %zu < Wait(%zu) - node %d dev %d",
      *GetResponseQueue(node, dev), wait, node, dev);
}

RequestQueue* Platform::GetRequestQueue(int node) 
{
  if (!IsHost())
    node = 0;

  return node_request_queues_[node];
}

RequestQueue* Platform::GetRequestQueue(int node, int dev)
{
  if (!IsHost())
    node = 0;

  int cluster_dev_idx;
  cluster_dev_idx = cluster_device_offsets_[node] + dev;

  return device_request_queues_[cluster_dev_idx];
}

void Platform::EnqueueCommand(RHACCommand *cmd)
{
  // FIXME
  enum CommandKind cmd_kind = cmd->GetCommandKind(); 
  assert(cmd_kind != ADMemcpyToSymbol && cmd_kind != ADMemcpy2DToArray);

  int node = cmd->GetTargetNode();
  int dev = cmd->GetTargetDevice();
  RequestQueue *request_queue;

  if (cmd->IsNodeCommand() || cmd->IsAllCommand())
    request_queue = GetRequestQueue(node);
  else
    request_queue = GetRequestQueue(node, dev);

  while(!request_queue->Enqueue(cmd));
}

void Platform::EnqueueCommand(RHACCommand *cmd, int node)
{
  RequestQueue *request_queue;

  request_queue = GetRequestQueue(node);

  while(!request_queue->Enqueue(cmd));
}

void Platform::EnqueueCommand(RHACCommand *cmd, int node, int dev)
{
  RequestQueue *request_queue;

  request_queue = GetRequestQueue(node, dev);

  while(!request_queue->Enqueue(cmd));
}

void Platform::FinishRequestQueue(int node)
{
  assert(IsHost());
  rhac_command_id_t last_id;
  RequestQueue *request_queue;

  request_queue = GetRequestQueue(node);
  last_id = request_queue->GetLastEnqueueID();

  RHAC_LOG("%s : node %d last id : %zu", __func__, node, last_id);
  WaitResponse(last_id, node);
}

void Platform::FinishRequestQueue(int node, int dev)
{
  rhac_command_id_t last_id;
  RequestQueue *request_queue;

  request_queue = GetRequestQueue(node, dev);
  last_id = request_queue->GetLastEnqueueID();

  WaitResponse(last_id, node, dev);
}

void Platform::FinishAllRequestQueue()
{
  int n, d;

  for (n = 0; n < num_nodes_; n++) {
    FinishRequestQueue(n);

    for (d = 0; d < GetNumDevicesIn(n); d++) {
      FinishRequestQueue(n, d);
    }
  }
}

void Platform::FinishAllRequestQueueExcept(int node)
{
  assert(IsHost());
  int n, d;

  for (n = 0; n < num_nodes_; n++) {
    if (n != node)
      FinishRequestQueue(n);

    for (d = 0; d < GetNumDevicesIn(n); d++) {
      FinishRequestQueue(n, d);
    }
  }
}

void Platform::FinishAllRequestQueueExcept(int node, int dev)
{
  assert(IsHost());
  int n, d;

  for (n = 0; n < num_nodes_; n++) {
    FinishRequestQueue(n);

    for (d = 0; d < GetNumDevicesIn(n); d++) {
      if (n != node || d != dev)
        FinishRequestQueue(n, d);
    }
  }
}
// ==================================================================
// functions related to the event handling
void Platform::InsertEvent(cudaEvent_t cuda_event, RHACEvent *rhac_event)
{
  event_map_.insert(std::make_pair(cuda_event, rhac_event));
}

void Platform::EraseEvent(cudaEvent_t cuda_event)
{
  std::map<cudaEvent_t, RHACEvent*>::iterator it;
  it = event_map_.find(cuda_event);
  assert(it != event_map_.end());
  event_map_.erase(it);
}

RHACEvent* Platform::GetEvent(cudaEvent_t cuda_event)
{
  std::map<cudaEvent_t, RHACEvent*>::iterator it;
  it = event_map_.find(cuda_event);
  assert(it != event_map_.end());
  return it->second;
}

// ==================================================================
// functions related to the kernel parsing
void Platform::ReadKernelInfoFile()
{
  FunctionInfo *func_info = NULL;
  std::string line;
  std::string func_name;
  int fatbin_index;
  int local_stage = 0;

  std::ifstream info_file(KERNEL_INFO_FILE_NAME);
  if (!info_file.is_open()) {
    fprintf(stderr, "Failed to read %s\n", KERNEL_INFO_FILE_NAME);
    return;
  }

  while (std::getline(info_file, line)) {
    std::istringstream iss(line);
    do {
      std::string word;
      iss >> word;

      if (word.size() == 0)
        continue;

      switch (local_stage) {
        case 0:
          fatbin_index = strtol(word.c_str(), NULL, 10);
          func_info = new FunctionInfo;
          local_stage = 1;
          break;
        case 1:
          func_name = word;
          local_stage = 2;
          break;
        case 2:
          func_info->has_global_atomics = strtol(word.c_str(), NULL, 10);
          local_stage = 3;
          break;
        case 3:
          func_info->num_args = strtol(word.c_str(), NULL, 10);
          local_stage = 4;
          break;
        case 4:
          func_info->arg_sizes.push_back(strtol(word.c_str(), NULL, 10));
          local_stage = 4;
          break;
      }

    } while (iss);

    if (func_info != NULL)
      func_info_map[std::make_pair(fatbin_index, func_name)] = func_info;

//  RHAC_LOG("func info map name       : %s", func_name.c_str());
//  RHAC_LOG("func info map fatbin idx : %d", fatbin_index);
//  RHAC_LOG("func info map num args   : %d", func_info->num_args);
//  std::string tmp_str;
//  tmp_str.clear();
//  std::vector<int>::iterator ti;
//  for (ti = func_info->arg_sizes.begin();
//       ti != func_info->arg_sizes.end();
//       ti++)
//  {
//    tmp_str.append(std::to_string(*ti) + " ");
//  }
//  RHAC_LOG("func info map num args   : %s", tmp_str.c_str());


    func_info = NULL;
    local_stage = 0;
  }

  info_file.close();
}

void Platform::ReadVarInfoFile()
{
  VarInfo *var_info = NULL;
  std::string line;
  std::string var_name;
  int fatbin_index;
  int local_stage = 0;

  std::ifstream info_file(VAR_INFO_FILE_NAME);
  if (!info_file.is_open()) {
    fprintf(stderr, "Failed to read %s\n", VAR_INFO_FILE_NAME);
    return;
  }

  while (std::getline(info_file, line)) {
    std::istringstream iss(line);
    do {
      std::string word;
      iss >> word;

      if (word.size() == 0)
        continue;

      switch (local_stage) {
        case 0:
          fatbin_index = strtol(word.c_str(), NULL, 10);
          var_info = new VarInfo;
          local_stage = 1;
          break;
        case 1:
          var_info->var_name = word;
          local_stage = 2;
          break;
        case 2:
          var_info->type_bitwidth = strtol(word.c_str(), NULL, 10);
          local_stage = 3;
          break;
        case 3:
          var_info->array_width = strtol(word.c_str(), NULL, 10);
          local_stage = 3;
          break;
      }
    } while (iss);

    if (var_info != NULL) {
      std::vector<VarInfo*>& var_infos = var_info_map[fatbin_index];
      var_infos.push_back(var_info);
    }

    var_info = NULL;
    local_stage = 0;
  }

  info_file.close();
}

FunctionInfo* Platform::GetKernelInfo(int fatbin_index, const char *kernel_name)
{
  FunctionInfo* FI = NULL;

  if (kernel_name)
    FI = func_info_map[std::make_pair(fatbin_index, kernel_name)];
  return FI;
}

std::vector<VarInfo*>* Platform::GetVarInfos(int fatbin_index) {
  std::vector<VarInfo*> *ret;

  if (var_info_map.find(fatbin_index) == var_info_map.end()) {
    ret = NULL;
  }
  else {
    ret = &var_info_map[fatbin_index];
  }

  return ret;
}

cudaFuncCache Platform::GetCudaCacheConfig()
{
  return cuda_cache_config_;
}

void Platform::SetCudaCacheConfig(cudaFuncCache cuda_cache_config)
{
  cuda_cache_config_ = cuda_cache_config;
}

unsigned int Platform::GetNextArrayIndex() {
  return array_index_++;
}

#ifdef READDUP_FLAG_CACHING
bool Platform::IsInReadDupList(uint64_t base) {
  for (std::vector<uint64_t>::iterator I = readdup_bufs_.begin(),
      E = readdup_bufs_.end(); I != E; ++I) {
    if (*I == base) {
      return true;
    }
  }
  return false;
}

void Platform::AddToReadDupList(uint64_t base) {
  readdup_bufs_.push_back(base);
}

void Platform::RemoveFromReadDupList(uint64_t base) {
  for (std::vector<uint64_t>::iterator I = readdup_bufs_.begin(),
      E = readdup_bufs_.end(); I != E; ++I) {
    if (*I == base) {
      readdup_bufs_.erase(I);
      return;
    }
  }
}

void Platform::ClearReadDupList() {
  readdup_bufs_.clear();
}
#endif

// ========================================================
// functions related to barrier
void Platform::RhacBarrierWait()
{
  pthread_barrier_wait(&rhac_barrier_);
}

void Platform::RhacBarrierRegister(RHACBarrier *b)
{
  while(!barrier_queue_->Enqueue(b));
}

bool Platform::RhacBarrierGet(RHACBarrier **b)
{
  return barrier_queue_->Peek(b);
}

void Platform::RhacBarrierDeleteFront()
{
  RHACBarrier *b;
  assert(barrier_queue_->Dequeue(&b));
}

void Platform::RhacBarrierEnqueue()
{
  int n, d;
  int nNodes, nDevs;
  RHACCommand *gb_cmd;
  rhac_command_id_t gb_cmd_id;

  assert(IsHost());

  nNodes = GetClusterNumNodes();

  // send Global Barrier
  RHACBarrier *barrier;
  if (rhac_platform.GetClusterNumNodes() > 1)
    barrier = new RHACBarrier;

  // send global barrier for host executors
  nDevs = rhac_platform.GetNumDevicesIn(0);
  gb_cmd_id = rhac_platform.GenerateCommandID(nDevs + 1); // 1 for node executor
  gb_cmd = new RHACCommand;
  gb_cmd->SetDefaultInfo(gb_cmd_id,
                         AGlobalBarrier,
                         0, nDevs+1); // nDevs+1 == ref count
  rhac_platform.EnqueueCommand(gb_cmd, 0);

  for (d = 0; d < nDevs; d++)
    rhac_platform.EnqueueCommand(gb_cmd, 0, d);

  // Register CommandID info to Barrier object
  if (rhac_platform.GetClusterNumNodes() > 1) {
    // Node Executer Barrier Info
    barrier->RegisterBarrierCommandID(gb_cmd_id, 0);
    for (d = 0; d < nDevs; d++) {
      // Device Executor Barrier Info
      barrier->RegisterBarrierCommandID(gb_cmd_id, 0, d);
    }
  }

  for (n = 1; n < nNodes; n++) {
    // Global barrier command for each node
    nDevs = rhac_platform.GetNumDevicesIn(n);
    gb_cmd_id = rhac_platform.GenerateCommandID(nDevs + 1); // 1 for node executor
    gb_cmd = new RHACCommand;
    gb_cmd->SetDefaultInfo(gb_cmd_id,
                           AGlobalBarrier,
                           n, -1);
    rhac_platform.EnqueueCommand(gb_cmd);

    // Node Executer Barrier Info
    barrier->RegisterBarrierCommandID(gb_cmd_id, n);
    for (d = 0; d < nDevs; d++) {
      // Device Executor Barrier Info
      barrier->RegisterBarrierCommandID(gb_cmd_id, n, d);
    }
  }

  if (rhac_platform.GetClusterNumNodes() > 1)
    rhac_platform.RhacBarrierRegister(barrier);
}

// ========================================================
// private functions
void Platform::CreateResponseQueue()
{
  /* My node response queue */
  node_response_queues_.push_back(new volatile rhac_command_id_t(0));

  /* My device response queues */
  int d, n_d = GetNodeNumDevices();
  for (d = 0; d < n_d; d ++) {
    device_response_queues_.push_back(new volatile rhac_command_id_t(0));
  }

  /* Remote Node Response Queues */
  if (IsHost()) {
    for (int n = 1; n < num_nodes_; n++) {
      node_response_queues_.push_back(new volatile rhac_command_id_t(0));

      for (int d = 0; d < num_devices_per_node_[n]; d++) {
        device_response_queues_.push_back(new volatile rhac_command_id_t(0));
      }
    }
  }
}

void Platform::CreateRequestQueues()
{
  /* My node request queue */
  node_request_queues_.push_back(new RequestQueue(QUEUE_SIZE));

  /* My Node's devices request queues */
  int d, n_d = GetNodeNumDevices();
  for (d = 0; d < n_d; d++) {
    device_request_queues_.push_back(new RequestQueue(QUEUE_SIZE));
  }

  /* Remote Node Request Queues */
  if (IsHost()) {
    /* all queues to remote node share ONE queue */
    QueueSPSC<RHACCommand>* remote_queue;
    remote_queue = new QueueSPSC<RHACCommand>(QUEUE_SIZE);
    for (int n = 1; n < num_nodes_; n++) {
      node_request_queues_.push_back(new RequestQueue(remote_queue));

      for (int d = 0; d < num_devices_per_node_[n]; d++) {
        device_request_queues_.push_back(new RequestQueue(remote_queue));
      }
    }
  }
}

void Platform::AllocGlobalVars() {
  ClusterSVM* cluster_svm = ClusterSVM::GetClusterSVM();

  for (std::map<int, std::vector<VarInfo*>>::iterator MI = var_info_map.begin(),
      ME = var_info_map.end(); MI != ME; ++MI) {
    std::vector<VarInfo*>* var_infos = &MI->second;

    // calculate the total required size for a fatbin
    size_t total_size = 0;
    for (std::vector<VarInfo*>::iterator VI = var_infos->begin(),
        VE = var_infos->end(); VI != VE; ++VI) {
      VarInfo* info = *VI;
      size_t alloc_size = info->type_bitwidth * info->array_width / 8;
      size_t alloc_mod = alloc_size % 512;
      if (alloc_mod != 0)
        alloc_size += (512 - alloc_mod);
      total_size += alloc_size;
    }

    void* dev_ptr = cluster_svm->MallocClusterSVM(total_size);
    cudaMemset(dev_ptr, 0, total_size);

    for (std::vector<VarInfo*>::iterator VI = var_infos->begin(),
        VE = var_infos->end(); VI != VE; ++VI) {
      VarInfo* info = *VI;
      size_t alloc_size = info->type_bitwidth * info->array_width / 8;
      size_t alloc_mod = alloc_size % 512;
      if (alloc_mod != 0)
        alloc_size += (512 - alloc_mod);

      info->dev_ptr = dev_ptr;
      dev_ptr = (void*)((uint64_t)dev_ptr + alloc_size);
    }
  }
}
