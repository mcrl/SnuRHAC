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

#ifndef __RHAC_CORRELATOR_H__
#define __RHAC_CORRELATOR_H__

#include <linux/cdev.h>
#include <linux/mm.h>
#include <linux/kthread.h>

#include "rhac_config.h"
#include "rhac_nvidia_decl.h"

typedef struct {
  NvU64 tag[RHAC_CORRELATOR_ASSOC];
  NvU64 successor[RHAC_CORRELATOR_ASSOC][RHAC_CORRELATOR_LEVEL][RHAC_CORRELATOR_SUCCS];
  NvU64 entry_counter[RHAC_CORRELATOR_ASSOC];
  NvU64 global_counter;
} correlator_row_t;

typedef correlator_row_t* correlator_table_t;

typedef struct {
	char name[32];
  unsigned int gpu_id;
	atomic_t should_stop;   // flag to stop all workloads and stop the thread
  atomic_t should_flush;  // flag to flush all workloads and wait the thread
	struct {
		wait_queue_head_t wait_queue;
		atomic_t cnt;
	} queue;
	struct task_struct *kthread;
} correlator_thread_t;

void rhac_create_correlation_tables(void);
void rhac_destroy_correlation_tables(void);
void rhac_clear_correlation_tables(unsigned int gpu_id);
void rhac_clear_correlation_successors(unsigned int gpu_id);

bool rhac_correlation_queue_is_empty(unsigned int gpu_id);
bool rhac_correlation_queue_is_full(unsigned int gpu_id);
void rhac_correlation_queue_clear(unsigned int gpu_id);

void rhac_push_update_request(unsigned int gpu_id, NvU64 fault_address);
NvU64 rhac_pop_update_request(unsigned int gpu_id);

NvU32 rhac_get_correlators(uvm_gpu_t *gpu,
    uvm_fault_buffer_entry_t *current_entry,
    uvm_fault_buffer_entry_t *new_entry);

NvU64 *rhac_update_correlation_table(unsigned int gpu_id,
    NvU64 fault_address, NvU64 **last_successors);

void rhac_create_correlator_threads(void);
void rhac_destroy_correlator_threads(void);
void rhac_clear_correlator_threads(void);
void rhac_wakeup_correlator_thread(uvm_gpu_t *gpu);

#endif //__RHAC_CORRELATOR_H__
