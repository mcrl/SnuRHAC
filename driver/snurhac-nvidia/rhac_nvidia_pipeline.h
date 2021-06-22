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

#ifndef __RHAC_NVIDIA_ISR_H__
#define __RHAC_NVIDIA_ISR_H__


#include <linux/types.h>
#include "nvidia-uvm/uvm8_forward_decl.h"
#include "nvidia-uvm/uvm8_processors.h"
#include "nvidia-uvm/uvm8_gpu.h"
#include "nvidia-uvm/uvm8_va_block.h"

enum {
	RHAC_NVIDIA_PIPELINE_FAULT_START = 0,
	RHAC_NVIDIA_PIPELINE_FAULT_MIGRATE,
	RHAC_NVIDIA_PIPELINE_FAULT_WRAPUP,
	RHAC_NVIDIA_PIPELINE_FAULT_DONE,
	RHAC_NVIDIA_PIPELINE_PREFETCH_START,
	RHAC_NVIDIA_PIPELINE_PREFETCH_MIGRATE,
	RHAC_NVIDIA_PIPELINE_PREFETCH_WRAPUP,
	RHAC_NVIDIA_PIPELINE_PREFETCH_DONE,
	RHAC_NVIDIA_PIPELINE_MAX,
};

struct rhac_comm;

struct rhac_nvidia_isr_ctx {
	struct rhac_comm *comm;
	uvm_va_block_t *va_block;
	uvm_service_block_context_t *block_context;
	uvm_service_block_context_t service_context;
	uvm_fault_service_batch_context_t *batch_context;

	uvm_gpu_t *gpu;
	struct mm_struct *mm;
	NvU32 first_fault_index;
	bool is_write;
	bool done;

	uvm_va_block_region_t region;
  uvm_page_mask_t page_mask;
	uvm_va_block_retry_t va_block_retry;
	uvm_processor_id_t dst_id;
	uvm_migrate_mode_t mode;
	uvm_tracker_t *out_tracker;
	uvm_mutex_t *lock;

	struct work_struct work;
};

int rhac_nvidia_pipeline_init(void);
void rhac_nvidia_pipeline_deinit(void);
void rhac_nvidia_pipeline_flush(void);

void rhac_nvidia_pipeline_enqueue(struct rhac_comm *comm, int step);

NV_STATUS rhac_nvidia_pipeline_cpu_fault(struct rhac_comm *, uvm_va_block_t *, NvU64 , bool , uvm_service_block_context_t *);
NV_STATUS rhac_nvidia_pipeline_gpu_fault(struct rhac_comm *, uvm_gpu_t *, struct mm_struct *, uvm_va_block_t *, NvU32 , uvm_fault_service_batch_context_t *, NvU32 *, uvm_mutex_t*);


NV_STATUS rhac_nvidia_pipeline_prefetch( struct rhac_comm *, uvm_va_block_t *, uvm_va_block_region_t, uvm_page_mask_t *, uvm_processor_id_t, uvm_migrate_mode_t, uvm_tracker_t *, uvm_mutex_t*);
#endif //__RHAC_NVIDIA_ISR_H__
