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

#ifndef __RHAC_DRIVER_H__
#define __RHAC_DRIVER_H__

#include "utils.h"
#include "bitmap.h"

class RHACDriver {
  public:
    void LoadDriver();
    void UnloadDriver();
    void InitClusterSVM(void *cluster_svm, int node, int rank, int num_local_gpus);
    void ReserveClusterSVM(size_t size);
    void Synchronize();
    void MemcpyHostToSVM();
    void MemcpySVMToHost();
    void SplitVARange(uint64_t base, uint64_t length);
    void SetDupFlag(uint64_t base, uint64_t length);
    void UnsetDupFlag(uint64_t base, uint64_t length);
    void PrefetchToCPU(uint64_t vaddr, uint64_t size, uint32_t device_id, bool is_async);
    void PrefetchToGPU(uint64_t vaddr, uvm_page_mask_t *page_mask, uint32_t device_id);

  private:
    int rhac_driver_fd_ = -1;

#ifdef DEBUG_NV_UVM_DRIVER
  public:
    void LoadNVDriver();
    void UnloadNVDriver();
    void ResetFaultStatus(int device_id);
    void PrintFaultStatus(int device_id);

  private:
    int nv_uvm_driver_fd_ = -1;
#endif

  // methods and members for singleton implementation
  public:
    static RHACDriver* GetRHACDriver();

  private:
    RHACDriver();
    ~RHACDriver();
    static RHACDriver *singleton_;
    static mutex_t mutex_;
};

#endif // __RHAC_DRIVER_H__
