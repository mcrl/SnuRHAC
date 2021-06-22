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

#include "thread.h"
#include <stdio.h>
#include <atomic>

std::atomic<int> affinity_type0_counter {0};
std::atomic<int> affinity_type1_counter {0};
std::atomic<int> affinity_type2_counter {0};

Thread::Thread(enum AffinityType affinity_type) {
  affinity_type_ = affinity_type;

  switch (affinity_type) {
  case AFFINITY_TYPE_0:
    affinity_id_ = affinity_type0_counter.fetch_add(1);
    break;
  case AFFINITY_TYPE_1:
    affinity_id_ = affinity_type1_counter.fetch_add(1);
    break;
  case AFFINITY_TYPE_2:
    affinity_id_ = affinity_type2_counter.fetch_add(1);
    break;
  }
}

Thread::~Thread() {
}

void Thread::Run() {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  switch (affinity_type_) {
  case AFFINITY_TYPE_0:
    CPU_SET(affinity_id_ * 2 + 0, &cpuset);
    CPU_SET(affinity_id_ * 2 + 1, &cpuset);
    break;
  case AFFINITY_TYPE_1:
  case AFFINITY_TYPE_2:
    CPU_SET(20 + affinity_id_ * 2 + 0, &cpuset);
    CPU_SET(20 + affinity_id_ * 2 + 1, &cpuset);
    break;
  }

  pthread_attr_t thread_attr;
  pthread_attr_init(&thread_attr);
  pthread_attr_setaffinity_np(&thread_attr, sizeof(cpuset), &cpuset);
  pthread_create(&thread_, &thread_attr, &Thread::run, (void *)this);
}

void* Thread::run(void *arg) {
  Thread *my = (Thread*)arg;
  my->run_();
  return 0;
}

void Thread::Quit() {
  pthread_join(thread_, NULL);
}

