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

#ifndef __RHAC_UTILS_H__
#define __RHAC_UTILS_H__

#include "config.h"
#include "rhac_command.h"

#include <stdio.h>
#include <pthread.h>
#include <assert.h>

#define RHAC_PRINT(fmt, ...) fprintf(stderr, "" fmt "\n", ## __VA_ARGS__)

#ifdef RHAC_LOGGING
#define RHAC_LOG(fmt, ...) fprintf(stderr, "[RHAC Runtime Log - %s:%d] " fmt "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#define RHAC_ERROR(fmt, ...) fprintf(stderr, "[RHAC Runtime Error - %s:%d] " fmt "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#else
#define RHAC_LOG(fmt, ...) 
#define RHAC_ERROR(fmt, ...) fprintf(stderr, "[RHAC Runtime Error - %s:%d] " fmt "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#endif // RHAC_LOGGING

#define YD_TODO() \
  fprintf(stderr, " %s:%d - %s not implemented yet \n", __FILE__, __LINE__, __func__);\
  assert(0);


#define CHECK_CUDART_ERROR(err)                                   \
  if (err != cudaSuccess) {                                       \
    RHAC_ERROR("CUDA runtime error: %d", err);                    \
    assert(0);                                                    \
  }

#define CHECK_MPI_ERROR(err)            \
  if (err != MPI_SUCCESS) {             \
    RHAC_ERROR("MPI error : %d", err);  \
    assert(0);                          \
  }

// Single Producer & Single Comsumer Queue
template<typename T>
class QueueSPSC {
  public:
    QueueSPSC(unsigned long size) 
      : size_(size), idx_r_(0), idx_w_(0) {
        elements_ = (volatile T**)(new T*[size]);
      }

    virtual ~QueueSPSC() {
      delete[] elements_;
    }

    virtual bool Enqueue(T* element) {
      unsigned long next_idx_w = (idx_w_ + 1) % size_;
      if(next_idx_w == idx_r_) return false;
      elements_[idx_w_] = element;
      __sync_synchronize();
      idx_w_ = next_idx_w;
      return true;
    }

    bool Dequeue(T** element) {
      if(idx_r_ == idx_w_) return false;
      unsigned long next_idx_r = (idx_r_ + 1) % size_;
      *element = (T*)elements_[idx_r_];
      idx_r_ = next_idx_r;
      return true;
    }

    bool Peek(T** element) {
      if(idx_r_ == idx_w_) return false;
      *element = (T*) elements_[idx_r_];
      return true;
    }

    unsigned long Size() {
      if(idx_w_ >= idx_r_) return idx_w_ - idx_r_;
      return size_ - idx_r_ + idx_w_;
    }

  protected:
    unsigned long size_;

    volatile T** elements_;
    volatile unsigned long idx_r_;
    volatile unsigned long idx_w_;

};

// Multiple Producers & Single Consumer Queue
template <typename T>
class QueueMPSC: public QueueSPSC<T>
{
  public:
    QueueMPSC(unsigned long size) 
      : QueueSPSC<T>(size), idx_w_cas_(0) {
      }

    ~QueueMPSC() {
    }

    bool Enqueue(T* element) {
      while(true) {
        unsigned long prev_idx_w = idx_w_cas_;
        unsigned long next_idx_w = (prev_idx_w + 1) % this->size_;
        if(next_idx_w == this->idx_r_) return false;
        if(__sync_bool_compare_and_swap(&idx_w_cas_, 
              prev_idx_w, next_idx_w)) {
          this->elements_[prev_idx_w] = element;
          while(!__sync_bool_compare_and_swap(&this->idx_w_, 
                prev_idx_w, next_idx_w)) {}
          break;
        }
      }
      return true;
    }

    bool Enqueue(T* element, int& idx) {
      while(true) {
        unsigned long prev_idx_w = idx_w_cas_;
        unsigned long next_idx_w = (prev_idx_w + 1) % this->size_;
        if(next_idx_w == this->idx_r_) return false;
        if(__sync_bool_compare_and_swap(&idx_w_cas_, 
              prev_idx_w, next_idx_w)) {
          this->elements_[prev_idx_w] = element;
          idx = prev_idx_w;
          //printf("idx = %d\n", idx);
          while(!__sync_bool_compare_and_swap(&this->idx_w_, 
                prev_idx_w, next_idx_w)) {}
          break;
        }
      }
      return true;
    }


  private:
    volatile unsigned long idx_w_cas_;  
};

// FIXME - move source code to ~.cpp 
class RequestQueue {
  public:
    RequestQueue(unsigned long size) {
      queue_ = new QueueSPSC<RHACCommand>(size);
      last_enqueue_id_ = 0;
    }
    RequestQueue() {
      queue_ = new QueueSPSC<RHACCommand>(QUEUE_SIZE);
      last_enqueue_id_ = 0;
    }
    RequestQueue(QueueSPSC<RHACCommand>* queue) {
      queue_ = queue;
      last_enqueue_id_ = 0;
    }
    ~RequestQueue() {
    }

    bool Enqueue(RHACCommand *element) {
      last_enqueue_id_ = element->GetCommandID();
      return queue_->Enqueue(element);
    }

    bool Dequeue(RHACCommand **element) {
      return queue_->Dequeue(element);
    }

    rhac_command_id_t GetLastEnqueueID() {
      return last_enqueue_id_;
    }

  private:
    rhac_command_id_t last_enqueue_id_;
    QueueSPSC<RHACCommand> *queue_;
};


class mutex_t {
  public:
    mutex_t() { pthread_mutex_init(&mutex_, 0); }
    ~mutex_t() { pthread_mutex_destroy(&mutex_); }
    void lock() { pthread_mutex_lock(&mutex_); }
    void unlock() { pthread_mutex_unlock(&mutex_); }
  private:
    pthread_mutex_t mutex_;
};

#endif // __RHAC_UTILS_H__
