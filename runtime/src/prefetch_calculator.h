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

#ifndef __RHAC_PREFETCH_CALCULATOR_H__
#define __RHAC_PREFETCH_CALCULATOR_H__

#include "rhac.h"
#include "thread.h"
#include "utils.h"
#include "prefetch_scheduler.h"

#include <cuda.h>
#include <semaphore.h>
#include <sys/time.h>

enum IndexKind {
  GroupX, // blockIdx.x
  GroupY, // blockIdx.y
  GroupZ, // blockIdx.z
  LocalX, // threadIdx.x
  LocalY, // threadIdx.y
  LocalZ, // threadIdx.z
  Iter0,  // iteration variable (inner)
  Iter1   // iteration variable (outer)
};

typedef struct {
  IndexKind kind;
  int64_t coeff;
  size_t dimension;
  size_t bound_lower;
  size_t bound_upper;
  size_t last_value;  // this value varies between [bound_lower, bound_upper]
                      // used when index is distinguished as non-invariant
} IndexInfo;

typedef struct {
  uint32_t buffer_id;
  uint64_t buffer_base;
  size_t buffer_bound;
  std::vector<IndexInfo>* indices_invariant;
  std::vector<IndexInfo>* indices_variant;
  std::vector<IndexInfo>* indices_iteration;
  int64_t const_var;
  size_t fetch_size;
  size_t left_offset;
  size_t left_length;
  PrefetchRequest prefetch_request;
  bool need_to_recalculate;
  bool reached_end;
  int64_t offset_invariants;
  int64_t length_invariants;
  bool calculated_invariants;
  bool wakenup_scheduler;
} BufferInfo;

class PrefetchCalculator : public Thread {
  public:
    PrefetchCalculator(int node_id, int device_id, PrefetchScheduler *sched);
    ~PrefetchCalculator();
    void run_();

    void Kill();
    void Calculate(uint32_t buffer_id,
        uint64_t buffer_base, size_t buffer_bound,
        dim3 gridDim, dim3 blockDim,
        dim3 block_bound_lower, dim3 block_bound_upper,
        int64_t gx_coeff, int64_t gy_coeff, int64_t gz_coeff,
        int64_t lx_coeff, int64_t ly_coeff, int64_t lz_coeff,
        int64_t i0_bound, int64_t i0_step, int64_t i1_bound, int64_t i1_step,
        int64_t const_var, size_t fetch_size);

    void WakeUp();
    void Wait();
    void EnqueueToScheduler(BufferInfo *binfo);
    
  private:
    int node_id_;
    int device_id_;
    sem_t sem_prefetch_;
    volatile bool threads_running_;
    volatile bool jobs_done_;
    
    QueueSPSC<BufferInfo> *request_queue_;
    std::vector<BufferInfo*> inflight_queue_;
    PrefetchScheduler* scheduler_;
};

#endif // __RHAC_PREFETCH_CALCULATOR_H__
