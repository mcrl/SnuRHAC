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

#include "prefetch_calculator.h"
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
#include <algorithm>
#include <math.h>

using namespace std;

PrefetchCalculator::PrefetchCalculator(int node_id, int device_id,
    PrefetchScheduler *sched)
  : Thread(AFFINITY_TYPE_1) {
  node_id_ = node_id;
  device_id_ = device_id;
  scheduler_ = sched;
  jobs_done_ = true;
  threads_running_ = true;
  request_queue_ = new QueueSPSC<BufferInfo>(MAX_NUM_EXPRESSIONS);
  sem_init(&sem_prefetch_, 0, 0);
}

PrefetchCalculator::~PrefetchCalculator() {
  delete request_queue_;
  sem_destroy(&sem_prefetch_);
}

void PrefetchCalculator::Kill() {
  threads_running_ = false;
  sem_post(&sem_prefetch_);
}

bool CoeffCompare(IndexInfo a, IndexInfo b) {
  return a.coeff < b.coeff;
}

bool KindCompare(IndexInfo a, IndexInfo b) {
  return a.kind < b.kind;
}

#define COEFF(x)  indices_invariant->at(x).coeff
#define DIM(x)    indices_invariant->at(x).dimension
void PrefetchCalculator::Calculate(uint32_t buffer_id,
    uint64_t buffer_base, size_t buffer_bound,
    dim3 gridDim, dim3 blockDim,
    dim3 block_bound_lower, dim3 block_bound_upper,
    int64_t gx_coeff, int64_t gy_coeff, int64_t gz_coeff,
    int64_t lx_coeff, int64_t ly_coeff, int64_t lz_coeff,
    int64_t i0_bound, int64_t i0_step, int64_t i1_bound, int64_t i1_step,
    int64_t const_var, size_t fetch_size) {
  std::vector<IndexInfo>* indices_invariant = new std::vector<IndexInfo>();
  std::vector<IndexInfo>* indices_variant = new std::vector<IndexInfo>();
  std::vector<IndexInfo>* indices_iteration = new std::vector<IndexInfo>();
  {
    // group variables
    if (gx_coeff != 0) {
      indices_invariant->push_back((IndexInfo){GroupX, gx_coeff,
          block_bound_upper.x - block_bound_lower.x + 1,
          block_bound_lower.x, block_bound_upper.x,
          block_bound_lower.x});
    }
    if (gy_coeff != 0) {
      indices_invariant->push_back((IndexInfo){GroupY, gy_coeff,
          block_bound_upper.y - block_bound_lower.y + 1,
          block_bound_lower.y, block_bound_upper.y,
          block_bound_lower.y});
    }
    if (gz_coeff != 0) {
      indices_invariant->push_back((IndexInfo){GroupZ, gz_coeff,
          block_bound_upper.z - block_bound_lower.z + 1,
          block_bound_lower.z, block_bound_upper.z,
          block_bound_lower.z});
    }
  
    // local variables
    if (lx_coeff != 0) {
      indices_invariant->push_back((IndexInfo){LocalX, lx_coeff,
          blockDim.x, 0, blockDim.x - 1, 0});
    }
    if (ly_coeff != 0) {
      indices_invariant->push_back((IndexInfo){LocalY, ly_coeff,
          blockDim.y, 0, blockDim.y - 1, 0});
    }
    if (lz_coeff != 0) {
      indices_invariant->push_back((IndexInfo){LocalZ, lz_coeff,
          blockDim.z, 0, blockDim.z - 1, 0});
    }
  
    // iteration variables
    if (i0_step != 0) {
      indices_invariant->push_back((IndexInfo){Iter0, i0_step,
          (size_t)ceil((float)i0_bound/i0_step),
          0, (size_t)ceil((float)i0_bound/i0_step) - 1, 0});
    }
    if (i1_step != 0) {
      indices_invariant->push_back((IndexInfo){Iter1, i1_step,
          (size_t)ceil((float)i1_bound/i1_step),
          0, (size_t)ceil((float)i1_bound/i1_step) - 1, 0});
    }

    // sort coefficients in ascending order 
    sort(indices_invariant->begin(), indices_invariant->end(), CoeffCompare);
  }

  int num_indices = indices_invariant->size();
  if (num_indices >= 8)
    assert("Number of indices is gte to 8");
  else if (num_indices == 0)
    assert("Number of indices is zero");

  float density_term[7] = { 0 };
  switch (num_indices) {
  case 7: density_term[6] =
          min(1.0f, max(4096.0f, (float)(
                  COEFF(0) * DIM(0) + COEFF(1) * DIM(1) +
                  COEFF(2) * DIM(2) + COEFF(3) * DIM(3) +
                  COEFF(4) * DIM(4) + COEFF(5) * DIM(5))) / COEFF(6));
  case 6: density_term[5] =
          min(1.0f, max(4096.0f, (float)(
                  COEFF(0) * DIM(0) + COEFF(1) * DIM(1) +
                  COEFF(2) * DIM(2) + COEFF(3) * DIM(3) +
                  COEFF(4) * DIM(4))) / COEFF(5));
  case 5: density_term[4] =
          min(1.0f, max(4096.0f, (float)(
                  COEFF(0) * DIM(0) + COEFF(1) * DIM(1) +
                  COEFF(2) * DIM(2) + COEFF(3) * DIM(3))) / COEFF(4));
  case 4: density_term[3] =
          min(1.0f, max(4096.0f, (float)(
                  COEFF(0) * DIM(0) + COEFF(1) * DIM(1) +
                  COEFF(2) * DIM(2))) / COEFF(3));
  case 3: density_term[2] =
          min(1.0f, max(4096.0f, (float)(
                  COEFF(0) * DIM(0) + COEFF(1) * DIM(1))) / COEFF(2));
  case 2: density_term[1] =
          min(1.0f, max(4096.0f, (float)(
                  COEFF(0) * DIM(0))) / COEFF(1));
  case 1: density_term[0] =
          min(1.0f, 4096.0f / COEFF(0));
  }

  int threshold_index = 0;
  switch (num_indices) {
  case 7:
    if (density_term[6] *
        density_term[5] * density_term[4] * density_term[3] *
        density_term[2] * density_term[1] * density_term[0] >=
        ACCESS_DENSITY_THRESHOLD) {
      threshold_index = 7;
      break;
    }
  case 6:
    if (density_term[5] * density_term[4] * density_term[3] *
        density_term[2] * density_term[1] * density_term[0] >=
        ACCESS_DENSITY_THRESHOLD) {
      threshold_index = 6;
      break;
    }
  case 5:
    if (density_term[4] * density_term[3] *
        density_term[2] * density_term[1] * density_term[0] >=
        ACCESS_DENSITY_THRESHOLD) {
      threshold_index = 5;
      break;
    }
  case 4:
    if (density_term[3] *
        density_term[2] * density_term[1] * density_term[0] >=
        ACCESS_DENSITY_THRESHOLD) {
      threshold_index = 4;
      break;
    }
  case 3:
    if (density_term[2] * density_term[1] * density_term[0] >=
        ACCESS_DENSITY_THRESHOLD) {
      threshold_index = 3;
      break;
    }
  case 2:
    if (density_term[1] * density_term[0] >= ACCESS_DENSITY_THRESHOLD) {
      threshold_index = 2;
      break;
    }
  case 1:
    if (density_term[0] >= ACCESS_DENSITY_THRESHOLD) {
      threshold_index = 1;
      break;
    }
  }

  if (threshold_index == 0) {
    if (node_id_ == 0 && device_id_ == 0) {
//      printf("Prefetching expr %u buffer %lu (indices %d) ignored\n",
//          buffer_id, buffer_base, num_indices);
//      printf("density_term[0..7] = {%f, %f, %f, %f, %f, %f, %f, %f}\n",
//          density_term[0], density_term[1], density_term[2], density_term[3],
//          density_term[4], density_term[5], density_term[6], density_term[7]);
    }
    return;
  }

  // move indices that do not satisfy the threshold
  for (std::vector<IndexInfo>::iterator VI =
      (indices_invariant->begin() + threshold_index);
      VI != indices_invariant->end(); ) {
    IndexInfo item = *VI;
#if 0
    if (item.kind >= Iter0)
      indices_iteration->push_back(item);
    else
      indices_variant->push_back(item);
    VI = indices_invariant->erase(VI);
#else
    indices_variant->push_back(item);
    VI = indices_invariant->erase(VI);
#endif
  }

//  // sort iteration indices
//  sort(indices_iteration->begin(), indices_iteration->end(), KindCompare);
  
  if (node_id_ == 0 && device_id_ == 0) {
//    printf("Prefetching expr %u buffer %lu (%lu inv, %lu vari, %lu iter)\n",
//        buffer_id, buffer_base, indices_invariant->size(), indices_variant->size(),
//        indices_iteration->size());
//    printf("threshold_index=%d, density_term[0..7] = {%f, %f, %f, %f, %f, %f, %f, %f}\n",
//        threshold_index,
//        density_term[0], density_term[1], density_term[2], density_term[3],
//        density_term[4], density_term[5], density_term[6], density_term[7]);
  }

  BufferInfo* binfo = new BufferInfo();
  binfo->buffer_id = buffer_id;
  binfo->buffer_base = buffer_base;
  binfo->buffer_bound = buffer_bound;
  binfo->indices_invariant = indices_invariant;
  binfo->indices_variant = indices_variant;
  binfo->indices_iteration = indices_iteration;
  binfo->const_var = const_var;
  binfo->fetch_size = fetch_size;
  binfo->left_offset = 0;
  binfo->left_length = 0;
  binfo->prefetch_request.addr = 0;
  zero_page_mask(&binfo->prefetch_request.page_mask);
  binfo->need_to_recalculate = true;
  binfo->reached_end = false;
  binfo->offset_invariants = 0;
  binfo->length_invariants = 0;
  binfo->calculated_invariants = false;
  binfo->wakenup_scheduler = false;
  request_queue_->Enqueue(binfo);
}

void PrefetchCalculator::WakeUp() {
  jobs_done_ = false;
  sem_post(&sem_prefetch_);

  // wait until prefetch thread reads all enqueued requests
  while (request_queue_->Size() != 0);
}

void PrefetchCalculator::Wait() {
  // busy-wait
  while (jobs_done_ == false);
}

void PrefetchCalculator::EnqueueToScheduler(BufferInfo *binfo) {
  scheduler_->Enqueue(binfo->buffer_id, &binfo->prefetch_request);
  if (binfo->wakenup_scheduler == false) {
    scheduler_->WakeUp();
    binfo->wakenup_scheduler = true;
  }
}

void PrefetchCalculator::run_() {
  BufferInfo* binfo;
  bool handle_same_expression = false;

  while(threads_running_) {
    sem_wait(&sem_prefetch_);
    
    // lookup request queue and move the request to inflight queue
    while (request_queue_->Dequeue(&binfo))
      inflight_queue_.push_back(binfo);

    while (1) { 
      // handle requests in inflight queue
      for (std::vector<BufferInfo*>::iterator RI =
          inflight_queue_.begin(); RI != inflight_queue_.end(); ) {
        binfo = *RI;

        if (binfo->need_to_recalculate == true) {
          if (binfo->reached_end) {
            if (binfo->left_length != 0)
              EnqueueToScheduler(binfo);

            RI = inflight_queue_.erase(RI);
            delete binfo->indices_invariant;
            delete binfo->indices_variant;
            delete binfo->indices_iteration;
            delete binfo;
            continue;
          }

          int64_t offset_variants = 0;
          int64_t prefetch_offset;
          int64_t prefetch_length;
          bool inc_next_index = true;
  
          // handle variant indices first
          for (std::vector<IndexInfo>::iterator
              II = binfo->indices_variant->begin(),
              IE = binfo->indices_variant->end(); II != IE; ++II) {
            IndexInfo *iinfo = &(*II);
            offset_variants += iinfo->coeff * iinfo->last_value;
  
            if (inc_next_index) {
              if (iinfo->last_value + 1 > iinfo->bound_upper) {
                iinfo->last_value = iinfo->bound_lower;
                inc_next_index = true;
              }
              else {
                iinfo->last_value += 1;
                inc_next_index = false;
              }
            }
          }
  
          // handle iteration indices next
          for (std::vector<IndexInfo>::iterator
              II = binfo->indices_iteration->begin(),
              IE = binfo->indices_iteration->end(); II != IE; ++II) {
            IndexInfo *iinfo = &(*II);
            offset_variants += iinfo->coeff * iinfo->last_value;
  
            if (inc_next_index) {
              if (iinfo->last_value + 1 > iinfo->bound_upper) {
                iinfo->last_value = iinfo->bound_lower;
                inc_next_index = true;
              }
              else {
                iinfo->last_value += 1;
                inc_next_index = false;
              }
            }
          }
          
          // we need to calculate access range comes from invariant variables
          // only once
          if (binfo->calculated_invariants == false) {
            int64_t offset_min = 0;
            int64_t offset_max = 0;

            // calculate access range from invariant indices
            for (std::vector<IndexInfo>::iterator
                II = binfo->indices_invariant->begin(),
                IE = binfo->indices_invariant->end(); II != IE; ++II) {
              IndexInfo *iinfo = &(*II);

              int64_t value_lower = iinfo->coeff * iinfo->bound_lower;
              int64_t value_upper = iinfo->coeff * iinfo->bound_upper;
              int64_t value_min = std::min(value_lower, value_upper);
              int64_t value_max = std::max(value_lower, value_upper);

              offset_min += value_min;
              offset_max += value_max;
            }

            offset_max += (binfo->fetch_size - 1);

            prefetch_offset = offset_min + binfo->const_var;
            prefetch_length = offset_max - offset_min;

            binfo->offset_invariants = prefetch_offset;
            binfo->length_invariants = prefetch_length;
            binfo->calculated_invariants = true;
          }
          else {
            prefetch_offset = binfo->offset_invariants;
            prefetch_length = binfo->length_invariants;
          }

          prefetch_offset += offset_variants;

          // when prefetch offset is negative, ignore the negative part
          if (prefetch_offset < 0) {
            prefetch_length += prefetch_offset;
            prefetch_offset = 0;
          }

          // some filtering cases
          if (prefetch_length < 0) {
//            printf("Prefetch length is negative! offset=%ld, length=%ld\n",
//                prefetch_offset, prefetch_length);
            prefetch_length = 0;
          }
          else if (prefetch_offset >= (int64_t)binfo->buffer_bound) {
//            printf("Prefetch offset exceed bound! offset=%ld, length=%ld, bound=%lu\n",
//                prefetch_offset, prefetch_length, binfo->buffer_bound);
            prefetch_length = 0;
          }
          else if (prefetch_offset + prefetch_length >= (int64_t)binfo->buffer_bound) {
//            printf("Prefetch length exceed bound! offset=%ld, length=%ld, bound=%lu\n",
//                prefetch_offset, prefetch_length, binfo->buffer_bound);
            prefetch_length = binfo->buffer_bound - prefetch_offset;
          }

          if (prefetch_length == 0) {
            // prefetch noting and calculate next access range
            binfo->need_to_recalculate = true;
            handle_same_expression = true;
          }
          else {
            uint64_t prefetch_addr = binfo->buffer_base + prefetch_offset;
            uint64_t prefetch_block_start =
              prefetch_addr >> UVM_VA_BLOCK_SIZE_BITS;
            uint64_t prefetch_block_end =
              (prefetch_addr + prefetch_length - 1) >> UVM_VA_BLOCK_SIZE_BITS;

            // if last last request exists, process it first
            if (binfo->left_length != 0) {
              uint64_t last_addr = binfo->buffer_base + binfo->left_offset;
              uint64_t last_block = last_addr >> UVM_VA_BLOCK_SIZE_BITS;

              if (last_block != prefetch_block_start) {
                EnqueueToScheduler(binfo);
                binfo->left_offset = 0;
                binfo->left_length = 0;
                zero_page_mask(&binfo->prefetch_request.page_mask);
              }
              else {
                if (prefetch_block_start == prefetch_block_end) {
                  // maintain page_mask of prefetch_request
                }
                else {
                  uint32_t partial_length =
                    UVM_VA_BLOCK_SIZE - (prefetch_addr & UVM_VA_BLOCK_SIZE_MASK);

                  set_page_mask(&binfo->prefetch_request.page_mask, prefetch_addr, partial_length);
                  EnqueueToScheduler(binfo);
                  binfo->left_offset = 0;
                  binfo->left_length = 0;
                  zero_page_mask(&binfo->prefetch_request.page_mask);

                  prefetch_addr += partial_length;
                  prefetch_offset += partial_length;
                  prefetch_length -= partial_length;
                  prefetch_block_start = prefetch_addr >> UVM_VA_BLOCK_SIZE_BITS;
                }
              }
            }

            // when this request ranges across multiple NVIDIA blocks,
            // process the range in the predecessor block first
            if (prefetch_block_start != prefetch_block_end) {
              uint32_t partial_length =
                UVM_VA_BLOCK_SIZE - (prefetch_addr & UVM_VA_BLOCK_SIZE_MASK);

              binfo->prefetch_request.addr = prefetch_addr;
              set_page_mask(&binfo->prefetch_request.page_mask, prefetch_addr, partial_length);
              EnqueueToScheduler(binfo);
              zero_page_mask(&binfo->prefetch_request.page_mask);

              prefetch_addr += partial_length;
              prefetch_offset += partial_length;
              prefetch_length -= partial_length;

              if (prefetch_length >= UVM_VA_BLOCK_SIZE) {
                binfo->need_to_recalculate = false;
                handle_same_expression = false;
              }
              else {
                binfo->need_to_recalculate = true;
                handle_same_expression = false;
              }
            }
            else {
              binfo->need_to_recalculate = true;
              handle_same_expression = true;
            }

            binfo->left_offset = prefetch_offset;
            binfo->left_length = prefetch_length;

            if (binfo->need_to_recalculate == true) {
              binfo->prefetch_request.addr = prefetch_addr;
              set_page_mask(&binfo->prefetch_request.page_mask, prefetch_addr,
                  prefetch_length);
            }
          }
  
          // when we reached the very end
          if (inc_next_index == true) {
            binfo->reached_end = true;
  
            if (binfo->left_length == 0) {
              RI = inflight_queue_.erase(RI);
              delete binfo->indices_invariant;
              delete binfo->indices_variant;
              delete binfo->indices_iteration;
              delete binfo;
              continue;
            }
          }
  
          // when handle_same_expression is on, peek next access range of same expression
          // rather than going to next expression
          if (handle_same_expression == true) {
            handle_same_expression = false;
          }
          else {
            ++RI;
          }
        }

        // we prefetch rest of the calculated access range
        else {
          size_t prefetch_offset = binfo->left_offset;
          size_t prefetch_length = binfo->left_length;
          uint64_t prefetch_addr = binfo->buffer_base + prefetch_offset;

          binfo->prefetch_request.addr = prefetch_addr;
          fill_page_mask(&binfo->prefetch_request.page_mask);
          EnqueueToScheduler(binfo);

          prefetch_addr += UVM_VA_BLOCK_SIZE;
          prefetch_offset += UVM_VA_BLOCK_SIZE;
          prefetch_length -= UVM_VA_BLOCK_SIZE;
          binfo->left_offset = prefetch_offset;
          binfo->left_length = prefetch_length;

          if (prefetch_length == 0) {
            binfo->need_to_recalculate = true;
          }
          else {
            if (prefetch_length >= UVM_VA_BLOCK_SIZE) {
              binfo->need_to_recalculate = false;
            }
            else {
              binfo->need_to_recalculate = true;
              binfo->prefetch_request.addr = prefetch_addr;
              zero_page_mask(&binfo->prefetch_request.page_mask);
              set_page_mask(&binfo->prefetch_request.page_mask, prefetch_addr,
                  prefetch_length);
            }
          }
          ++RI;
        }

      }

      // when there is no more expression to calculate access range, we exit
      if (inflight_queue_.size() == 0)
        break;
    }

    jobs_done_ = true;
  }
}
