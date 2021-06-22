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

#include "memcpy_helper.h"
#include "platform.h"
#include "utils.h"
#include "rhac_command.h"
#include "rhac_response.h"
#include "libcuda.h"
#include "rhac_driver.h"
#include "fatbin_handler.h"
#include "config.h"
#include "rhac_event.h"
#include <string.h>

MemcpyHelper::MemcpyHelper(int id)
  : Thread(AFFINITY_TYPE_2) {
  id_ = id;
  threads_running_ = true;
  sem_init(&sem_memcpy_, 0, 0);
}

MemcpyHelper::~MemcpyHelper() {
  sem_destroy(&sem_memcpy_);
}

void MemcpyHelper::Kill() {
  threads_running_ = false;
  sem_post(&sem_memcpy_);
}

void MemcpyHelper::Enqueue(uint64_t dst, uint64_t src, size_t size, int prefetch_flag) {
  jobs_done_ = false;

  dst_ = dst;
  src_ = src;
  size_ = size;
  prefetch_flag_ = prefetch_flag;

  // wake-up the thread
  sem_post(&sem_memcpy_);
}

void MemcpyHelper::Wait() {
  // busy-wait
  while (jobs_done_ == false);
}

void MemcpyHelper::run_() {
  while(threads_running_) {
    sem_wait(&sem_memcpy_);

#if defined(RHAC_PREFETCH)
    RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
    rhac_driver->PrefetchToCPU(prefetch_flag_ == 0 ? dst_ : src_, size_, -1, true);
#endif

    memcpy((void*)dst_, (void*)src_, size_);
    jobs_done_ = true;
  }
}
