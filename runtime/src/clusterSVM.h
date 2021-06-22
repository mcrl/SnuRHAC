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

#ifndef __RHAC_CLUSTERSVM_H__
#define __RHAC_CLUSTERSVM_H__

#include "utils.h"
#include "rhac.h"

#include <vector>

// ========================================================
// This class does not care about the race condition.
// ========================================================

class ClusterSVM {
  public:
    static ClusterSVM*  GetClusterSVM(void *cluster_svm_ptr, size_t reserved_size);
    static ClusterSVM*  GetClusterSVM();
    static void*        CreateNodeSVMPtr(size_t size);
    static void*        CreateClusterSVMPtr(void *node_svm_ptr);
    static void         FreeNodeSVM(void *node_svm);
    static void*        AlignSVM(void *prev, size_t align);
    static void         ReserveSVM(size_t size);

  private:
    static ClusterSVM*  singleton_;
    static mutex_t      mutex_;

  public:
    void PushPtr(void *ptr, size_t alloc_size);
    void PrintAllocPtrs(void);
    void* MallocClusterSVM(size_t size);
    void FreeClusterSVM(void *ptr); 
    void RequestReserve(size_t size);
    void* GetBase(void *ptr);
    size_t GetLength(void *ptr);
    void GetBaseAndLength(void *ptr, uint64_t *base, size_t *length);

  private:
    ClusterSVM(void *cluster_svm_ptr, size_t reserved_size);
    ~ClusterSVM();

    void                *cluster_svm_ptr_;
    size_t              reserved_size_;
    void                *alloc_head_;
    std::vector<std::pair<void *, size_t>> alloc_ptrs_;
};

#endif // __RHAC_CLUSTERSVM_H__
