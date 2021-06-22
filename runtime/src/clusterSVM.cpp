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

#include "clusterSVM.h"
#include "config.h"
#include "libcuda.h"
#include "platform.h"
#include "rhac_command.h"
#include "rhac_driver.h"

#include <mpi.h>
#include <algorithm>

// ========================================================
// This class does not care about the race condition.
// ========================================================

ClusterSVM* ClusterSVM::singleton_ = NULL;
mutex_t ClusterSVM::mutex_;

// ========================================================
// static functions for general purpose

ClusterSVM* ClusterSVM::GetClusterSVM(void *cluster_svm_ptr, size_t reserved_size)
{
  assert(rhac_platform.IsHost());
  /* Init and return singleton */
  mutex_.lock();
  assert(singleton_ == NULL);
  singleton_ = new ClusterSVM(cluster_svm_ptr, reserved_size);
  mutex_.unlock();

  return singleton_;
}
ClusterSVM* ClusterSVM::GetClusterSVM() 
{
  assert(rhac_platform.IsHost());
  assert(singleton_ != NULL);

  return singleton_;
}

void* ClusterSVM::CreateNodeSVMPtr(size_t size)
{
  void *node_svm;
  cudaError_t cuda_err;
  LibCUDA *lib_cuda = LibCUDA::GetLibCUDA();

  cuda_err = lib_cuda->cudaMallocManaged(&node_svm,
                                         size,
                                         cudaMemAttachGlobal);
  CHECK_CUDART_ERROR(cuda_err);

  return node_svm;
}

// called by initialization of platform
void* ClusterSVM::CreateClusterSVMPtr(void* node_svm_ptr)
{
  int mpi_err;
  uint64_t cluster_svm_ptr, t_node_svm_ptr;
  uint64_t *node_svm_ptrs;
  int num_nodes;

  t_node_svm_ptr = (uint64_t)node_svm_ptr;

  if (rhac_platform.IsHost()) {
    num_nodes = rhac_platform.GetClusterNumNodes();
    node_svm_ptrs = new uint64_t[num_nodes];
  }

  /* gather all nodes' NodeSVM Pointers */
  mpi_err = MPI_Gather(&t_node_svm_ptr, 1, MPI_UINT64_T,
                       node_svm_ptrs, 1, MPI_UINT64_T,
                       HOST_NODE, MPI_COMM_WORLD);
  CHECK_MPI_ERROR(mpi_err);

  /* get cluster SVM */
  if (rhac_platform.IsHost()) {
    cluster_svm_ptr = *std::max_element(node_svm_ptrs, node_svm_ptrs + num_nodes);
    cluster_svm_ptr = (uint64_t)AlignSVM((void *)cluster_svm_ptr, SVM_ALIGNMENT);
  }

  /* Broadcast clusterSVM */
  mpi_err = MPI_Bcast(&cluster_svm_ptr, 1, MPI_UINT64_T,
                      HOST_NODE, MPI_COMM_WORLD);
  CHECK_MPI_ERROR(mpi_err);

  assert(cluster_svm_ptr % SVM_ALIGNMENT == 0L);

  if (rhac_platform.IsHost()) {
    delete[] node_svm_ptrs;
  }

  RHAC_LOG("Rank %d NodeSVM %p clusterSVM %p", 
      rhac_platform.GetRank(), node_svm_ptr, (void *)cluster_svm_ptr);

  return (void *)cluster_svm_ptr;
}

void ClusterSVM::FreeNodeSVM(void *node_svm)
{
  cudaError_t cuda_err;
  LibCUDA* lib_cuda = LibCUDA::GetLibCUDA();

  cuda_err = lib_cuda->cudaFree(node_svm);
  CHECK_CUDART_ERROR(cuda_err);
}

void* ClusterSVM::AlignSVM(void *prev, size_t align)
{
  uint64_t ret;
  ret = (uint64_t)prev;

  if (ret % align != 0L) {
    ret = ((ret / align) + 1)*align;
  }

  return (void *)ret;
}

void ClusterSVM::ReserveSVM(size_t size)
{
  RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
  rhac_driver->ReserveClusterSVM(size);
  RHAC_LOG("Rank %d call ReserveSVM with size : %zu", 
      rhac_platform.GetRank(), size);
}

// ===========================================================
// functions for Host's cluster SVM
ClusterSVM::ClusterSVM(void *cluster_svm_ptr, 
                       size_t reserved_size) 
{
  // TODO check this singleton implementations
  cluster_svm_ptr_ = cluster_svm_ptr;
  reserved_size_ = reserved_size;
  alloc_head_ = cluster_svm_ptr;
}

ClusterSVM::~ClusterSVM() 
{
}

void ClusterSVM::PushPtr(void *ptr, size_t alloc_size) {
  assert(rhac_platform.IsHost());

  alloc_ptrs_.push_back(std::pair<void*, size_t>(ptr, alloc_size));
}

void ClusterSVM::PrintAllocPtrs(void) {
  assert(rhac_platform.IsHost());
  std::vector<std::pair<void*, size_t>>::iterator s, e, i;
  s = alloc_ptrs_.begin();
  e = alloc_ptrs_.end();
  int cnt = 0;

  for (i = s; i != e; i++) {
    RHAC_LOG("Host alloc pointers %d : %p (%lubytes)", cnt, i->first, i->second);
    cnt++;
  }
}

void* ClusterSVM::MallocClusterSVM(size_t size) {
  void *ret;
  assert(rhac_platform.IsHost());

  // FIXME
  RHAC_LOG("Malloc called");

  while ((uint64_t)alloc_head_ + size 
      >= (uint64_t)cluster_svm_ptr_ + reserved_size_)
  {
    RequestReserve(reserved_size_ * 2L);
    reserved_size_ *= 2L;
    RHAC_LOG("Host New Reseved Size : %zu", reserved_size_);
  }
  // TODO check reserved space is enough
  // alloc_head_ + size >= cluster_svm_ptr + reserve ?

  assert((uint64_t)alloc_head_ % SVM_ALIGNMENT == 0);

  ret = alloc_head_;
  PushPtr(ret, size);

  alloc_head_ = (void*)((uint64_t)alloc_head_ + (uint64_t)size);
  alloc_head_ = AlignSVM(alloc_head_, SVM_ALIGNMENT);

  assert((uint64_t)alloc_head_ % SVM_ALIGNMENT == 0);

  // FIXME
  RHAC_LOG("Malloc called");
  PrintAllocPtrs();

  return ret;
}

void ClusterSVM::FreeClusterSVM(void *ptr)
{
  assert(rhac_platform.IsHost());

  // TODO
  RHAC_LOG("TODO call FreeClusterSVM to pointer %p", ptr);
}

void ClusterSVM::RequestReserve(size_t size)
{
  assert(rhac_platform.IsHost());
  RHAC_LOG("Reqeust Reserve Start");

  RHACCommandNSVMReserve *cmd;
  rhac_command_id_t cmd_id;
  int n, cluster_n = rhac_platform.GetClusterNumNodes();
  int *wait = new int[cluster_n];

  for (n = 0; n < cluster_n; n++) {
    cmd_id = rhac_platform.GenerateCommandID();
    cmd = new RHACCommandNSVMReserve();
    cmd->SetDefaultInfo(cmd_id, NSVMReserve, n, -1);
    cmd->SetReserveSize(size);
    wait[n] = cmd_id;

    rhac_platform.EnqueueCommand((RHACCommand*)cmd);
  }

  // FIXME - WaitAllRequest
  for (n = 0; n < cluster_n; n++) {
    rhac_platform.WaitResponse(wait[n], n);
  }

  delete[] wait;

  RHAC_LOG("Reqeust Reserve Done");
}

void* ClusterSVM::GetBase(void *ptr) {
  assert(rhac_platform.IsHost());

  for (std::vector<std::pair<void*, size_t>>::iterator I = alloc_ptrs_.begin(),
      E = alloc_ptrs_.end(); I != E; ++I) {
    uint64_t elem_base = (uint64_t)I->first;
    size_t elem_length = (size_t)I->second;

    if (elem_base <= (uint64_t)ptr &&
        elem_base + elem_length > (uint64_t)ptr) {
      return (void*)elem_base;
    }
  }

  RHAC_ERROR("Failed to get base of ptr %lu", (uint64_t)ptr);
  return NULL;
}

size_t ClusterSVM::GetLength(void *ptr) {
  assert(rhac_platform.IsHost());

  for (std::vector<std::pair<void*, size_t>>::iterator I = alloc_ptrs_.begin(),
      E = alloc_ptrs_.end(); I != E; ++I) {
    uint64_t elem_base = (uint64_t)I->first;
    size_t elem_length = (size_t)I->second;

    if (elem_base <= (uint64_t)ptr &&
        elem_base + elem_length > (uint64_t)ptr) {
      return elem_length;
    }
  }

  RHAC_ERROR("Failed to get length of ptr %lu", (uint64_t)ptr);
  return 0;
}

void ClusterSVM::GetBaseAndLength(void *ptr, uint64_t *base, size_t *length) {
  assert(rhac_platform.IsHost());

  for (std::vector<std::pair<void*, size_t>>::iterator I = alloc_ptrs_.begin(),
      E = alloc_ptrs_.end(); I != E; ++I) {
    uint64_t elem_base = (uint64_t)I->first;
    size_t elem_length = (size_t)I->second;

    if (elem_base <= (uint64_t)ptr &&
        elem_base + elem_length > (uint64_t)ptr) {
      *base = elem_base;
      *length = elem_length;
      return;
    }
  }

  RHAC_ERROR("Failed to get length of ptr %lu", (uint64_t)ptr);
}
