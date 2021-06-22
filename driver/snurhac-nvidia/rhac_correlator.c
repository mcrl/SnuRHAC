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

#include <linux/uaccess.h>

#include "rhac_correlator.h"
#include "rhac_ctx.h"
#include "rhac_utils.h"
#include "rhac_ioctl.h"
#include "rhac_nvidia_symbols.h"

#include "nvidia-uvm/uvm8_va_range.h"
#include "nvidia-uvm/uvm8_va_space.h"

uvm_va_space_t *correlator_va_space = NULL;
correlator_table_t correlator_table[UVM_ID_MAX_GPUS];
correlator_thread_t correlator_thread[UVM_ID_MAX_GPUS];

volatile NvU64 correlation_update_queue[UVM_ID_MAX_GPUS][RHAC_CORRELATOR_QUEUE];
volatile NvU32 correlation_queue_head[UVM_ID_MAX_GPUS] = { 0 };
volatile NvU32 correlation_queue_tail[UVM_ID_MAX_GPUS] = { 0 };
volatile NvU64* correlation_last_successors[UVM_ID_MAX_GPUS][RHAC_CORRELATOR_LEVEL];

void rhac_create_correlation_tables(void) {
  unsigned int i;
  unsigned int num_local_gpus;
  num_local_gpus = rhac_ctx_get_global()->num_local_gpus;

  // save va_space information
  struct vm_area_struct *vma;
  uvm_va_space_t *va_space = NULL;

  if (current->mm == NULL) {
    RHAC_BUG();
    return;
  }
  else {
    vma = current->mm->mmap;
    while (vma != NULL) {
      if (rhacuvm_uvm_file_is_nvidia_uvm(vma->vm_file)) {
        va_space = (uvm_va_space_t *)(vma->vm_file)->private_data;
        break;
      }
      vma = vma->vm_next;
    }
  }

  if (va_space) {
    correlator_va_space = va_space;
  }
  else {
    RHAC_BUG();
    return;
  }

  for (i = 0; i < num_local_gpus; ++i) {
    correlator_row_t *rows =
      vzalloc(RHAC_CORRELATOR_ROWS * sizeof(correlator_row_t));
    correlator_table[i] = rows;
  }
}

void rhac_destroy_correlation_tables(void) {
  unsigned int i;
  unsigned int num_local_gpus;

  num_local_gpus = rhac_ctx_get_global()->num_local_gpus;
  correlator_va_space = NULL;

  for (i = 0; i < num_local_gpus; ++i)
    vfree(correlator_table[i]);
}

void rhac_clear_correlation_tables(unsigned int gpu_id) {
  memset(correlator_table[gpu_id], 0,
      RHAC_CORRELATOR_ROWS * sizeof(correlator_row_t));
}

void rhac_clear_correlation_successors(unsigned int gpu_id) {
  memset(correlation_last_successors[gpu_id], 0,
      RHAC_CORRELATOR_LEVEL * sizeof(NvU64*));
}

bool rhac_correlation_queue_is_empty(unsigned int gpu_id) {
  return (correlation_queue_head[gpu_id] == correlation_queue_tail[gpu_id]);
}

bool rhac_correlation_queue_is_full(unsigned int gpu_id) {
  return ((correlation_queue_head[gpu_id] - correlation_queue_tail[gpu_id])
      == RHAC_CORRELATOR_QUEUE);
}

void rhac_correlation_queue_clear(unsigned int gpu_id) {
  correlation_queue_tail[gpu_id] = correlation_queue_head[gpu_id];
}

void rhac_push_update_request(unsigned int gpu_id, NvU64 fault_address) {
  NvU64 new_head = correlation_queue_head[gpu_id] + 1;
  NvU32 new_index = new_head & RHAC_CORRELATOR_QUEUE_MASK;

  correlation_update_queue[gpu_id][new_index] = fault_address;
  correlation_queue_head[gpu_id] = new_head;
}

NvU64 rhac_pop_update_request(unsigned int gpu_id) {
  NvU64 new_tail = correlation_queue_tail[gpu_id] + 1;
  NvU32 new_index = new_tail & RHAC_CORRELATOR_QUEUE_MASK;

  NvU64 fault_address = correlation_update_queue[gpu_id][new_index];
  correlation_queue_tail[gpu_id] = new_tail;

  return fault_address;
}

NvU32 rhac_get_correlators(uvm_gpu_t *gpu,
    uvm_fault_buffer_entry_t *current_entry,
    uvm_fault_buffer_entry_t *new_entry) {
  uvm_va_range_t *va_range;
  correlator_row_t *correlator_row;
  NvU64 fault_address;
  NvU64 fault_tag;
  unsigned int row_index, added_index;
  unsigned int i, j;
  unsigned int gpu_id;
  int assoc_index;

  fault_address = current_entry->fault_address;
  va_range = rhacuvm_uvm_va_range_find(correlator_va_space, fault_address);
  if (!va_range) {
    RHAC_BUG();
    return 0;
  }

  if (va_range->read_duplication == UVM_READ_DUPLICATION_ENABLED)
    return 0;

  gpu_id = uvm_id_gpu_index(gpu->id);

  // push correlation table update request to worker thread
  // we do not update correlation table here to minimize response time
  while (rhac_correlation_queue_is_full(gpu_id));
  rhac_push_update_request(gpu_id, fault_address);

  fault_tag = fault_address >> 12; // page offset
  row_index = fault_tag & RHAC_CORRELATOR_ROWS_MASK;
  fault_tag >>= RHAC_CORRELATOR_ROWS_BITS;

  correlator_row = &correlator_table[gpu_id][row_index];

  assoc_index = -1;
  for (i = 0; i < RHAC_CORRELATOR_ASSOC; ++i) {
    if (correlator_row->tag[i] == fault_tag) {
      assoc_index = i;
      break;
    }
  }

  // when tag matching failed, return empty-handed
  if (assoc_index == -1)
    return 0;

  added_index = 0;
  for (i = 0; i < RHAC_CORRELATOR_LEVEL; ++i) {
    for (j = 0; j < RHAC_CORRELATOR_SUCCS; ++j) {
      if (correlator_row->successor[assoc_index][i][j] != 0) {
        // first copy all information from faulted entry
        memcpy(&new_entry[added_index], current_entry,
            sizeof(uvm_fault_buffer_entry_t));

        // replace some intel
        new_entry[added_index].fault_address =
          correlator_row->successor[assoc_index][i][j];
        INIT_LIST_HEAD(&new_entry[added_index].merged_instances_list);
        added_index += 1;
      }
    }
  }

  return added_index;
}

NvU64 *rhac_update_correlation_table(unsigned int gpu_id,
    NvU64 fault_address, NvU64 **last_successors) {
  NvU64 fault_tag;
  correlator_row_t *correlator_row;
  unsigned int row_index;
  unsigned int i, j;
  unsigned int copy_count;
  int assoc_match_index, assoc_new_index;

  // update row of level of successors
  for (i = 0; i < RHAC_CORRELATOR_LEVEL; ++i) {
    if (last_successors[i]) {
      // shift successors for MRU policy
      #if RHAC_CORRELATOR_SUCCS > 1
      copy_count = RHAC_CORRELATOR_SUCCS - 1;
      for (j = 0; j < RHAC_CORRELATOR_SUCCS; ++j) {
        if (last_successors[i][j] == fault_address) {
          copy_count = j;
          break;
        }
      }
      for (j = copy_count; j > 0; --j) {
        last_successors[i][j] = last_successors[i][j-1];
      }
      #endif
      last_successors[i][0] = fault_address;
    }
  }

  fault_tag = fault_address >> 12; // page offset
  row_index = fault_tag & RHAC_CORRELATOR_ROWS_MASK;
  fault_tag >>= RHAC_CORRELATOR_ROWS_BITS;

  correlator_row = &correlator_table[gpu_id][row_index];
  correlator_row->global_counter += 1;

  assoc_match_index = -1;
  assoc_new_index = -1;
  for (i = 0; i < RHAC_CORRELATOR_ASSOC; ++i) {
    NvU64 existing_tag = correlator_row->tag[i];
    if (existing_tag == 0) {
      assoc_new_index = i;
      break;
    }
    else if (existing_tag == fault_tag) {
      assoc_match_index = i;
    }
  }

  // entry already exits so we just update the counter of entry for LRU
  if (assoc_match_index != -1) {
    correlator_row->entry_counter[assoc_match_index] =
      correlator_row->global_counter;
    return correlator_row->successor[assoc_match_index][0];
  }

  // when tag matching failed
  else {
    // evict existing entry before we add new one
    if (assoc_new_index == -1) {
      // pick entry that has lowest counter
      assoc_new_index = 0;
      for (i = 1; i < RHAC_CORRELATOR_ASSOC; ++i) {
        if (correlator_row->entry_counter[i] <
            correlator_row->entry_counter[assoc_new_index]) {
          assoc_new_index = i;
        }
      }

      // initialize successors
      memset(correlator_row->successor[assoc_new_index], 0,
          RHAC_CORRELATOR_LEVEL * RHAC_CORRELATOR_SUCCS * sizeof(NvU64));
    }
    
    // add new entry: update tag field
    correlator_row->tag[assoc_new_index] = fault_tag;
    correlator_row->entry_counter[assoc_new_index] =
      correlator_row->global_counter;
    return correlator_row->successor[assoc_new_index][0];
  }
}

static int correlator_handler(void *arg) {
  correlator_thread_t *correlator_thread = (correlator_thread_t*)arg;
  NvU64* last_successors[RHAC_CORRELATOR_LEVEL];
  NvU64* new_successor;
  NvU64 fault_address;
  unsigned int gpu_id;
  unsigned int i;
  int err;

  //RHAC_LOG("%s has launched", correlator_thread->name);
  gpu_id = correlator_thread->gpu_id;

  for (;;) {
    err = wait_event_interruptible(correlator_thread->queue.wait_queue,
        atomic_read(&correlator_thread->queue.cnt) > 0);

    if (atomic_read(&correlator_thread->should_stop)) {
      break;
    }
    else if (atomic_read(&correlator_thread->should_flush)) {
      rhac_correlation_queue_clear(gpu_id);
      rhac_clear_correlation_tables(gpu_id);
      rhac_clear_correlation_successors(gpu_id);
      atomic_set(&correlator_thread[i].should_flush, 0);
      atomic_dec(&correlator_thread->queue.cnt);
      continue;
    }

    if (kthread_should_stop()) {
      return 0;
    }
    // update local tracker from global tracker
    memcpy(last_successors, correlation_last_successors[gpu_id],
        RHAC_CORRELATOR_LEVEL * sizeof(NvU64*));

    while (!rhac_correlation_queue_is_empty(gpu_id)) {
      fault_address = rhac_pop_update_request(gpu_id);
      new_successor = rhac_update_correlation_table(gpu_id,
          fault_address, last_successors);

      // exit loop immediately when device release is called
      if (atomic_read(&correlator_thread->should_stop)) {
        goto loop_exit;
      }
      else if (atomic_read(&correlator_thread->should_flush)) {
        rhac_correlation_queue_clear(gpu_id);
        rhac_clear_correlation_tables(gpu_id);
        rhac_clear_correlation_successors(gpu_id);
        atomic_set(&correlator_thread[gpu_id].should_flush, 0);
        atomic_dec(&correlator_thread->queue.cnt);
        goto next_loop;
      }

      // update last successors
      #if RHAC_CORRELATOR_LEVEL > 1
      for (i = RHAC_CORRELATOR_LEVEL-1; i > 0; --i) {
        last_successors[i] = last_successors[i-1];
      }
      for (i = 1; i < RHAC_CORRELATOR_LEVEL; ++i) {
        if (last_successors[i])
          last_successors[i] += RHAC_CORRELATOR_SUCCS;
      }
      #endif
      last_successors[0] = new_successor;
    }

    // update global tracker from local tracker
    memcpy(correlation_last_successors[gpu_id], last_successors,
        RHAC_CORRELATOR_LEVEL * sizeof(NvU64*));
next_loop:
    atomic_dec(&correlator_thread->queue.cnt);
  }
loop_exit:

  while (!kthread_should_stop())
    schedule();

  return 0;
}

void rhac_create_correlator_threads(void) {
  unsigned int i;
  for (i = 0; i < UVM_ID_MAX_GPUS; ++i) {
    memset(&correlator_thread[i], 0, sizeof(correlator_thread_t));

    sprintf(correlator_thread[i].name, "CORRELATOR_GPU%u", i);
    correlator_thread[i].gpu_id = i;
    atomic_set(&correlator_thread[i].should_stop, 0);
    atomic_set(&correlator_thread[i].should_flush, 0);

    init_waitqueue_head(&correlator_thread[i].queue.wait_queue);
    atomic_set(&correlator_thread[i].queue.cnt, 0);

    correlator_thread[i].kthread =
      kthread_run(correlator_handler, &correlator_thread[i], correlator_thread[i].name);
  }
}

void rhac_destroy_correlator_threads(void) {
  unsigned int i;
  for (i = 0; i < UVM_ID_MAX_GPUS; ++i) {
    atomic_set(&correlator_thread[i].should_stop, 1);
    atomic_inc(&correlator_thread[i].queue.cnt);
    wake_up_interruptible_sync(&correlator_thread[i].queue.wait_queue);
    kthread_stop(correlator_thread[i].kthread);
  }
}

void rhac_clear_correlator_threads(void) {
  unsigned int i;
  unsigned int num_local_gpus;
  num_local_gpus = rhac_ctx_get_global()->num_local_gpus;

  for (i = 0; i < num_local_gpus; ++i) {
    atomic_set(&correlator_thread[i].should_flush, 1);
    atomic_inc(&correlator_thread[i].queue.cnt);
    wake_up_interruptible_sync(&correlator_thread[i].queue.wait_queue);
  }
}

void rhac_wakeup_correlator_thread(uvm_gpu_t *gpu) {
  unsigned int gpu_id;
  gpu_id = uvm_id_gpu_index(gpu->id);
  atomic_inc(&correlator_thread[gpu_id].queue.cnt);
  wake_up_interruptible_sync(&correlator_thread[gpu_id].queue.wait_queue);
}
