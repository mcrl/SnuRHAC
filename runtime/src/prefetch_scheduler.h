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

#ifndef __RHAC_PREFETCH_SCHEDULER_H__
#define __RHAC_PREFETCH_SCHEDULER_H__

#include "rhac.h"
#include "thread.h"
#include "utils.h"
#include "bitmap.h"

#include <cuda.h>
#include <semaphore.h>
#include <sys/time.h>

typedef struct {
  uint64_t addr;
  uvm_page_mask_t page_mask;
} PrefetchRequest;
  
class PrefetchQueue {
public:
  PrefetchQueue(unsigned long size)
    : size_(size), idx_r_(0), idx_w_(0) {
      elements_ = (volatile PrefetchRequest*)(new PrefetchRequest[size]);
    }

  virtual ~PrefetchQueue() {
    delete elements_;
  }

  virtual bool Enqueue(PrefetchRequest* element) {
    unsigned long next_idx_w = (idx_w_ + 1) % size_;
    if (next_idx_w == idx_r_) return false;
    memcpy((void*)&elements_[idx_w_], element, sizeof(PrefetchRequest));
    __sync_synchronize();
    idx_w_ = next_idx_w;
    return true;
  }

  void Dequeue() {
    unsigned long next_idx_r = (idx_r_ + 1) % size_;
    idx_r_ = next_idx_r;
  }

  bool Dequeue(PrefetchRequest* element) {
    if (idx_r_ == idx_w_) return false;
    unsigned long next_idx_r = (idx_r_ + 1) % size_;
    if (element)
      memcpy(element, (void*)&elements_[idx_r_], sizeof(PrefetchRequest));
    idx_r_ = next_idx_r;
    return true;
  }

  bool Peek(PrefetchRequest** element) {
    if (idx_r_ == idx_w_) return false;
    *element = (PrefetchRequest*) &elements_[idx_r_];
    return true;
  }

  unsigned long Size() {
    if(idx_w_ >= idx_r_) return idx_w_ - idx_r_;
    return size_ - idx_r_ + idx_w_;
  }

protected:
  unsigned long size_;
  volatile PrefetchRequest* elements_;
  volatile unsigned long idx_r_;
  volatile unsigned long idx_w_;
};

class PrefetchScheduler : public Thread {
  public:
    PrefetchScheduler(int id);
    ~PrefetchScheduler();
    void run_();

    void Kill();
    void Enqueue(unsigned int queue_index, PrefetchRequest *request);
    void WakeUp();
    void Wait();
    void SetNumExpressions(unsigned int num);

  private:
    int id_;
    sem_t sem_prefetch_;
    volatile bool threads_running_;
    volatile bool jobs_done_;
    volatile unsigned int num_expressions_;
    
    PrefetchQueue *request_queue_[MAX_NUM_EXPRESSIONS];
};

#endif // __RHAC_PREFETCH_SCHEDULER_H__
