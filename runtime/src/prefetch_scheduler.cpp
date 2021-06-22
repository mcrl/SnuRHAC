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

#include "prefetch_scheduler.h"
#include "platform.h"
#include "config.h"
#include "utils.h"
#include "libcuda.h"
#include "fatbin_handler.h"
#include "rhac_command.h"
#include "rhac_driver.h"
#include "rhac_event.h"
#include "rhac_response.h"
#include <string.h>
#include <algorithm>
#include <math.h>

using namespace std;

PrefetchScheduler::PrefetchScheduler(int id)
  : Thread(AFFINITY_TYPE_1) {
  id_ = id;
  jobs_done_ = true;
  threads_running_ = true;
  for (unsigned int i = 0; i < MAX_NUM_EXPRESSIONS; ++i) {
    request_queue_[i] =
      new PrefetchQueue(PREFETCH_SCHEDULING_QUEUE_SIZE);
  }
  sem_init(&sem_prefetch_, 0, 0);
}

PrefetchScheduler::~PrefetchScheduler() {
  for (unsigned int i = 0; i < MAX_NUM_EXPRESSIONS; ++i)
    delete request_queue_[i];
  sem_destroy(&sem_prefetch_);
}

void PrefetchScheduler::Kill() {
  threads_running_ = false;
  sem_post(&sem_prefetch_);
}

void PrefetchScheduler::Enqueue(unsigned int queue_index,
    PrefetchRequest *request) {
  // busy-wait when the queue is full
  while (request_queue_[queue_index]->Enqueue(request) == false);
}

void PrefetchScheduler::WakeUp() {
  sem_post(&sem_prefetch_);
}

void PrefetchScheduler::Wait() {
  // busy-wait
  while (jobs_done_ == false);
}

void PrefetchScheduler::SetNumExpressions(unsigned int num) {
  num_expressions_ = num;
}

void PrefetchScheduler::run_() {
  RHACDriver *rhac_driver = RHACDriver::GetRHACDriver();
  PrefetchRequest req;
  unsigned int num_expressions;

  while(threads_running_) {
    sem_wait(&sem_prefetch_);
    jobs_done_ = false;
    num_expressions = num_expressions_;

    while (1) {
      bool exit_loop = true;

      for (unsigned int i = 0; i < num_expressions; ++i) {
        if (request_queue_[i]->Dequeue(&req)) {
          rhac_driver->PrefetchToGPU(req.addr, &req.page_mask, id_);
          exit_loop = false;
        }
      }
      if (exit_loop == true)
        break;
    }

    jobs_done_ = true;
  }
}
