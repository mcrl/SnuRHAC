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

#include "rhac_driver.h"
#include "config.h"
#include "platform.h"

#include <assert.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "rhac_ioctl.h"

#define CHECK_IOCTL(err)                \
  if (err != 0) {                       \
    RHAC_ERROR("IOCTL err : %d", err);  \
    assert(0);                          \
  }

RHACDriver* RHACDriver::singleton_ = NULL;
mutex_t RHACDriver::mutex_;

RHACDriver* RHACDriver::GetRHACDriver()
{
  mutex_.lock();
  if (singleton_ == NULL) {
    singleton_ = new RHACDriver();
  }
  mutex_.unlock();

  return singleton_;
}

RHACDriver::RHACDriver()
{
}

RHACDriver::~RHACDriver()
{
}

void RHACDriver::LoadDriver()
{
  rhac_driver_fd_ = open(RHAC_DRIVER_FILE, O_RDWR);

  if (rhac_driver_fd_ == -1) {
    RHAC_ERROR("Failed to open rhac driver fd");
    exit(0);
  }
}

void RHACDriver::UnloadDriver()
{
  RHAC_LOG("Rank %d close rhac driver", rhac_platform.GetRank());
  close(rhac_driver_fd_);
}

void RHACDriver::InitClusterSVM(void *cluster_svm, int node, int rank,
    int num_local_gpus) {
  int ioctl_err;

  rhac_iocx_init_param_t init = {
    .vaddr_base = (uint64_t)cluster_svm,
    .num_nodes = (uint32_t)node,
    .node_id = (uint32_t)rank,
    .num_local_gpus = (uint32_t)num_local_gpus,
  };

  ioctl_err = ioctl(rhac_driver_fd_, RHAC_IOCX_INIT, &init);
  CHECK_IOCTL(ioctl_err);
}

void RHACDriver::ReserveClusterSVM(size_t reserve_capacity)
{
  int ioctl_err;

  rhac_iocx_reserve_param_t reserve = {
    .capacity = (uint64_t) reserve_capacity,
  };
  ioctl_err = ioctl(rhac_driver_fd_, RHAC_IOCX_RESERVE, &reserve); 
  CHECK_IOCTL(ioctl_err);
}

void RHACDriver::Synchronize()
{
  int ioctl_err;
  ioctl_err = ioctl(rhac_driver_fd_, RHAC_IOCX_SYNC);
  CHECK_IOCTL(ioctl_err);
}

void RHACDriver::MemcpyHostToSVM()
{
}

void RHACDriver::MemcpySVMToHost()
{
}

void RHACDriver::SplitVARange(uint64_t base, uint64_t length) {
  int ioctl_err;

  rhac_iocx_split_va_range_param_t cmd = {
    .vaddr = base,
    .length = length,
  };

  ioctl_err = ioctl(rhac_driver_fd_, RHAC_IOCX_SPLIT_VA_RANGE, &cmd);
  CHECK_IOCTL(ioctl_err);
}

void RHACDriver::SetDupFlag(uint64_t base, uint64_t length) {
#if defined(RHAC_PREFETCH)
  int ioctl_err;

  rhac_iocx_toggle_dup_flag_param_t toggle = {
    .vaddr = base,
    .size = length,
    .turnon_flag = 1,
  };

  ioctl_err = ioctl(rhac_driver_fd_, RHAC_IOCX_TOGGLE_DUP_FLAG, &toggle);
  CHECK_IOCTL(ioctl_err);
#endif
}

void RHACDriver::UnsetDupFlag(uint64_t base, uint64_t length) {
#if defined(RHAC_PREFETCH)
  int ioctl_err;

  rhac_iocx_toggle_dup_flag_param_t toggle = {
    .vaddr = base,
    .size = length,
    .turnon_flag = 0,
  };

  ioctl_err = ioctl(rhac_driver_fd_, RHAC_IOCX_TOGGLE_DUP_FLAG, &toggle);
  CHECK_IOCTL(ioctl_err);
#endif
}

void RHACDriver::PrefetchToCPU(uint64_t vaddr, uint64_t size, uint32_t device_id, bool is_async) {
#if defined(RHAC_PREFETCH)
  int ioctl_err;

  rhac_iocx_prefetch_to_cpu_param_t cmd = {
    .vaddr = vaddr,
    .size = size,
    .device_id = device_id,
    .is_async = is_async,
  };

  ioctl_err = ioctl(rhac_driver_fd_, RHAC_IOCX_PREFETCH_TO_CPU, &cmd);
  CHECK_IOCTL(ioctl_err);
#endif
}

void RHACDriver::PrefetchToGPU(uint64_t vaddr, uvm_page_mask_t *page_mask, uint32_t device_id) {
#if defined(RHAC_PREFETCH)
  int ioctl_err;

  rhac_iocx_prefetch_to_gpu_param_t cmd = {
    .vaddr = vaddr,
    .page_mask = *page_mask,
    .device_id = device_id,
  };

  ioctl_err = ioctl(rhac_driver_fd_, RHAC_IOCX_PREFETCH_TO_GPU, &cmd);
  CHECK_IOCTL(ioctl_err);
#endif
}

#ifdef DEBUG_NV_UVM_DRIVER
void RHACDriver::LoadNVDriver() {
  nv_uvm_driver_fd_ = open(NV_UVM_DRIVER_FILE, O_RDWR);
  if (nv_uvm_driver_fd_ == -1) {
    RHAC_ERROR("Failed to open nvidia-uvm device\n");
    exit(0);
  }
}

void RHACDriver::UnloadNVDriver() {
  close(nv_uvm_driver_fd_);
}

void RHACDriver::ResetFaultStatus(int device_id) {
  int ioctl_err;
  ioctl_err = ioctl(nv_uvm_driver_fd_, 0x50000001, device_id);
  CHECK_IOCTL(ioctl_err);
}

void RHACDriver::PrintFaultStatus(int device_id) {
  uint64_t *data = (uint64_t*)malloc(sizeof(uint64_t) * 2 * 32);

  int ioctl_err;
  ioctl_err = ioctl(nv_uvm_driver_fd_, 0x50000002, data);
  CHECK_IOCTL(ioctl_err);

  printf("[GPU%d] Read fault: %lu, Write fault: %lu\n",
      device_id, data[device_id * 2 + 0], data[device_id * 2 + 1]);

  free(data);
}
#endif
