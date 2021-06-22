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

#include <linux/slab.h>
#include "rhac_ctx.h"
#include "rhac_nvidia_pipeline.h"
#include "rhac_nvidia_cpu.h"
#include "rhac_nvidia_gpu.h"
#include "rhac_nvidia_common.h"
#include "rhac_nvidia_prefetch.h"
#include "rhac_nvidia_mm.h"
#include "rhac_comm.h"
#include "rhac_nvidia_helpers.h"
#include "rhac_nvidia_symbols.h"

#include "rhac_protocol.h"
#include "rhac_utils.h"

#include "nvidia-uvm/uvm8_forward_decl.h"
#include "nvidia-uvm/uvm8_va_block.h"
#include "nvidia-uvm/uvm8_va_range.h"
#include "nvidia-uvm/uvm8_gpu.h"

uvm_service_block_context_t service_context_ro;
uvm_service_block_context_t service_context_rw;

static atomic_t num;
static atomic_t cpu_fault_cnt;
static atomic_t gpu_fault_cnt;
static atomic_t fault_cnt[RHAC_NVIDIA_PIPELINE_MAX];
static void nvidia_pipeline_fault_start(struct work_struct *work);
static void nvidia_pipeline_fault_migrate(struct work_struct *work);
static void nvidia_pipeline_fault_wrapup(struct work_struct *work);
static void nvidia_pipeline_fault_done(struct work_struct *work);
static void nvidia_pipeline_prefetch_start(struct work_struct *work);
static void nvidia_pipeline_prefetch_migrate(struct work_struct *work);
static void nvidia_pipeline_prefetch_wrapup(struct work_struct *work);
static void nvidia_pipeline_prefetch_done(struct work_struct *work);

static struct kmem_cache *isr_ctx_cache;
typedef void (*handler_t)(struct work_struct *);
static handler_t nvidia_pipeline_handlers[RHAC_NVIDIA_PIPELINE_MAX] = {
	[RHAC_NVIDIA_PIPELINE_FAULT_START] = nvidia_pipeline_fault_start,
	[RHAC_NVIDIA_PIPELINE_FAULT_MIGRATE] = nvidia_pipeline_fault_migrate,
	[RHAC_NVIDIA_PIPELINE_FAULT_WRAPUP] = nvidia_pipeline_fault_wrapup,
	[RHAC_NVIDIA_PIPELINE_FAULT_DONE] = nvidia_pipeline_fault_done,
	[RHAC_NVIDIA_PIPELINE_PREFETCH_START] = nvidia_pipeline_prefetch_start,
	[RHAC_NVIDIA_PIPELINE_PREFETCH_MIGRATE] = nvidia_pipeline_prefetch_migrate,
	[RHAC_NVIDIA_PIPELINE_PREFETCH_WRAPUP] = nvidia_pipeline_prefetch_wrapup,
	[RHAC_NVIDIA_PIPELINE_PREFETCH_DONE] = nvidia_pipeline_prefetch_done,
};

//static struct workqueue_struct* fault_queue;
static struct workqueue_struct* workqueue[RHAC_NVIDIA_PIPELINE_MAX];

atomic_t total_cnt;

inline static struct workqueue_struct* get_queue(int type)
{
  //return fault_queue;
	return workqueue[type];
}

NV_STATUS rhac_nvidia_pipeline_prefetch(
		struct rhac_comm *pa,
		uvm_va_block_t *va_block,
		uvm_va_block_region_t region,
    uvm_page_mask_t *page_mask,
		uvm_processor_id_t dst_id,
		uvm_migrate_mode_t mode,
		uvm_tracker_t *out_tracker,
		uvm_mutex_t *lock)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = kmem_cache_alloc(isr_ctx_cache, in_interrupt() ? GFP_ATOMIC : GFP_KERNEL);
	RHAC_ASSERT(isr_ctx);
	if (!isr_ctx) return NV_ERR_GENERIC;

	struct rhac_comm *comm = rhac_comm_spawn(pa);
	RHAC_ASSERT(comm);
	comm->cur = -2;

	comm->isr_ctx = isr_ctx;
	isr_ctx->comm = comm;
	isr_ctx->va_block = va_block;
	isr_ctx->block_context = &isr_ctx->service_context;
	isr_ctx->region = region;
  isr_ctx->page_mask = *page_mask;
	isr_ctx->dst_id = dst_id;
	isr_ctx->mode = mode;
	isr_ctx->out_tracker = out_tracker;
	isr_ctx->lock = lock;

	int err;
  comm->next = RHAC_NVIDIA_PIPELINE_PREFETCH_START;
  if (rhac_ctx_get_global()->num_nodes <= 1) {
    rhac_nvidia_pipeline_enqueue(isr_ctx->comm, RHAC_NVIDIA_PIPELINE_PREFETCH_START);
  } else {
    err = rhac_protocol_post_lock(isr_ctx->comm, isr_ctx->va_block->start);
    RHAC_ASSERT(!err);
  }


	return NV_OK;
}

static void nvidia_pipeline_prefetch_start(struct work_struct *work)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = container_of(work,
			struct rhac_nvidia_isr_ctx,
			work);
	struct rhac_comm *comm = isr_ctx->comm;

	comm->cur = RHAC_NVIDIA_PIPELINE_PREFETCH_START;

	rhac_nvidia_mm_lock_blk(isr_ctx->va_block->start);
	rhac_nvidia_pipeline_enqueue(isr_ctx->comm, RHAC_NVIDIA_PIPELINE_PREFETCH_MIGRATE);
}

static void nvidia_pipeline_prefetch_migrate(struct work_struct *work)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = container_of(work,
			struct rhac_nvidia_isr_ctx,
			work);
	struct rhac_comm *comm = isr_ctx->comm;

	comm->cur = RHAC_NVIDIA_PIPELINE_PREFETCH_MIGRATE;

	int err;

	comm->next = RHAC_NVIDIA_PIPELINE_PREFETCH_WRAPUP;
	rhacuvm_uvm_va_block_retry_init(&isr_ctx->va_block_retry);
	err = rhac_uvm_va_block_migrate_global(
			comm,
			isr_ctx->va_block,
			&isr_ctx->va_block_retry,
			&isr_ctx->block_context->block_context,
			isr_ctx->region,
      &isr_ctx->page_mask,
			isr_ctx->dst_id,
			isr_ctx->mode);
	RHAC_ASSERT(!err);

	if (!comm->processing) {
		rhac_nvidia_pipeline_enqueue(isr_ctx->comm, RHAC_NVIDIA_PIPELINE_PREFETCH_DONE);
	}
}

static void nvidia_pipeline_prefetch_wrapup(struct work_struct *work)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = container_of(work,
			struct rhac_nvidia_isr_ctx,
			work);
	struct rhac_comm *comm = isr_ctx->comm;

	comm->cur = RHAC_NVIDIA_PIPELINE_PREFETCH_WRAPUP;

	int err;

	uvm_page_mask_t *copy_mask = (uvm_page_mask_t*)comm->copy_mask;
	uvm_page_mask_t added_mask;

  // FIXME
  uvm_page_mask_andnot(&added_mask, copy_mask, &isr_ctx->page_mask);
  uvm_page_mask_or(&isr_ctx->page_mask, &isr_ctx->page_mask, &added_mask);
  //uvm_page_mask_and(copy_mask, copy_mask, &isr_ctx->page_mask);


	err = rhac_nvidia_make_resident_from_cpu(
			isr_ctx->va_block,
			&isr_ctx->block_context->block_context,
			isr_ctx->dst_id,
			isr_ctx->region,
      &isr_ctx->page_mask,
      copy_mask,
			uvm_va_range_is_read_duplicate(isr_ctx->va_block->va_range) ?
			UVM_VA_BLOCK_TRANSFER_MODE_COPY :
			UVM_VA_BLOCK_TRANSFER_MODE_MOVE
			);
	RHAC_ASSERT(!err);

	err = rhac_uvm_va_block_migrate_wrapup(
			comm,
			isr_ctx->va_block,
			&isr_ctx->block_context->block_context,
			isr_ctx->region,
      &isr_ctx->page_mask,
			isr_ctx->dst_id,
			isr_ctx->mode);
	RHAC_ASSERT(!err);


	if (isr_ctx->out_tracker) {
		NV_STATUS status;
		uvm_mutex_lock(isr_ctx->lock);
		status = rhacuvm_uvm_tracker_add_tracker_safe(isr_ctx->out_tracker, &isr_ctx->va_block->tracker);
		uvm_mutex_unlock(isr_ctx->lock);
		RHAC_ASSERT(status == NV_OK);
	}

	rhac_nvidia_pipeline_enqueue(isr_ctx->comm, RHAC_NVIDIA_PIPELINE_PREFETCH_DONE);
}

static void nvidia_pipeline_prefetch_done(struct work_struct *work)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = container_of(work,
			struct rhac_nvidia_isr_ctx,
			work);
	struct rhac_comm *comm = isr_ctx->comm;

	comm->cur = RHAC_NVIDIA_PIPELINE_PREFETCH_DONE;

	int err;

	rhacuvm_uvm_va_block_retry_deinit(&isr_ctx->va_block_retry, isr_ctx->va_block);

	rhac_nvidia_mm_unlock_blk(isr_ctx->va_block->start);

  if (rhac_ctx_get_global()->num_nodes > 1) {
    //if (atomic_read(&comm->req_unlock) || !comm->processing) {
      err = rhac_protocol_post_unlock(comm, isr_ctx->va_block->start);
      RHAC_ASSERT(!err);
    //}
  }

	rhac_comm_free(comm);

	kmem_cache_free(isr_ctx_cache, isr_ctx);
}

static uint32_t compute_block_faults(uvm_va_block_t *va_block,
		NvU32 first_fault_index,
		uvm_fault_service_batch_context_t *batch_context)
{
	NvU32 i;
	uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache;
	uvm_va_space_t *va_space = va_block->va_range->va_space;
	for (i = first_fault_index;
			i < batch_context->num_coalesced_faults &&
			ordered_fault_cache[i]->va_space == va_space &&
			ordered_fault_cache[i]->fault_address <= va_block->end;
			++i)
		;

	return i - first_fault_index;
}

NV_STATUS rhac_nvidia_pipeline_gpu_fault(
		struct rhac_comm *pa,
		uvm_gpu_t *gpu, 
		struct mm_struct *mm,
		uvm_va_block_t *va_block,
		NvU32 first_fault_index,
		uvm_fault_service_batch_context_t *batch_context,
		NvU32 *block_faults,
		uvm_mutex_t *lock
		)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = kmem_cache_alloc(isr_ctx_cache, in_interrupt() ? GFP_ATOMIC : GFP_KERNEL);
	RHAC_ASSERT(isr_ctx);
	if (!isr_ctx) return NV_ERR_GENERIC;

	struct rhac_comm *comm = rhac_comm_spawn(pa);
	RHAC_ASSERT(comm);
	comm->cur = -3;

	comm->isr_ctx = isr_ctx;
	isr_ctx->comm = comm;
	isr_ctx->gpu = gpu;
	isr_ctx->mm = mm;
	isr_ctx->va_block = va_block;
	isr_ctx->first_fault_index = first_fault_index;
	isr_ctx->block_context = &isr_ctx->service_context;
	isr_ctx->batch_context = batch_context;
	isr_ctx->dst_id = gpu->id;
	isr_ctx->out_tracker = NULL;
	isr_ctx->lock = lock;


	*block_faults = compute_block_faults(va_block, first_fault_index, batch_context);

	comm->number = atomic_fetch_inc(&num);

	int err;
  comm->next = RHAC_NVIDIA_PIPELINE_FAULT_START;
  if (rhac_ctx_get_global()->num_nodes <= 1) {
    comm->cur = -18;
    rhac_nvidia_pipeline_enqueue(isr_ctx->comm, RHAC_NVIDIA_PIPELINE_FAULT_START);
  } else {
    err = rhac_protocol_post_lock(isr_ctx->comm, isr_ctx->va_block->start);
    RHAC_ASSERT(!err);
    comm->cur = -18;
  }


	return NV_OK;
}

NV_STATUS rhac_nvidia_pipeline_cpu_fault(
		struct rhac_comm *pa,
		uvm_va_block_t *va_block,
		NvU64 fault_addr,
		bool is_write,
		uvm_service_block_context_t *service_context
		)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = kmem_cache_alloc(isr_ctx_cache, in_interrupt() ? GFP_ATOMIC : GFP_KERNEL);
	RHAC_ASSERT(isr_ctx);
	if (!isr_ctx) return NV_ERR_GENERIC;

	struct rhac_comm *comm = rhac_comm_spawn(pa);
	RHAC_ASSERT(comm);
	comm->cur = -4;

	comm->isr_ctx = isr_ctx;
	isr_ctx->comm = comm;
	isr_ctx->va_block = va_block;
	isr_ctx->block_context = service_context;
	isr_ctx->first_fault_index = uvm_va_block_cpu_page_index(va_block, fault_addr);
	isr_ctx->is_write = is_write;
	isr_ctx->dst_id = UVM_ID_CPU;
	isr_ctx->out_tracker = NULL;
	isr_ctx->lock = NULL;




	comm->number = atomic_fetch_inc(&num);
  comm->next = RHAC_NVIDIA_PIPELINE_FAULT_START;
  if (rhac_ctx_get_global()->num_nodes <= 1) {
    rhac_nvidia_pipeline_enqueue(isr_ctx->comm, RHAC_NVIDIA_PIPELINE_FAULT_START);
  } else {
    int err;
    err = rhac_protocol_post_lock(isr_ctx->comm, isr_ctx->va_block->start);
    RHAC_ASSERT(!err);
  }

	return NV_OK;
}

static void nvidia_pipeline_fault_start(struct work_struct *work)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = container_of(work,
			struct rhac_nvidia_isr_ctx,
			work);
	struct rhac_comm *comm = isr_ctx->comm;

	comm->cur = RHAC_NVIDIA_PIPELINE_FAULT_START;

	rhac_nvidia_mm_lock_blk(isr_ctx->va_block->start);
	nvidia_pipeline_fault_migrate(work);
	//rhac_nvidia_pipeline_enqueue(isr_ctx->comm, RHAC_NVIDIA_PIPELINE_FAULT_MIGRATE);

}

static void nvidia_pipeline_fault_migrate(struct work_struct *work)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = container_of(work,
			struct rhac_nvidia_isr_ctx,
			work);
	struct rhac_comm *comm = isr_ctx->comm;

	comm->cur = RHAC_NVIDIA_PIPELINE_FAULT_MIGRATE;


	int err;
	comm->next = RHAC_NVIDIA_PIPELINE_FAULT_WRAPUP;
	if (UVM_ID_IS_CPU(isr_ctx->dst_id)) {
		err = rhac_uvm_cpu_fault_start(
				isr_ctx->comm,
				isr_ctx->va_block,
				uvm_va_block_cpu_page_address(isr_ctx->va_block, isr_ctx->first_fault_index),
				isr_ctx->is_write,
				isr_ctx->block_context);
	} else {
		err = rhac_uvm_gpu_fault_start(
				isr_ctx->comm,
				isr_ctx->gpu,
				isr_ctx->mm,
				isr_ctx->va_block,
				isr_ctx->first_fault_index, 
				&isr_ctx->service_context,
				isr_ctx->batch_context);
	}
  RHAC_ASSERT(!err);

  comm->cur = -1204;

  if (!comm->processing) {
    rhac_nvidia_pipeline_enqueue(isr_ctx->comm, RHAC_NVIDIA_PIPELINE_FAULT_DONE);
	} 
  comm->cur = -577;
}


static void nvidia_pipeline_fault_wrapup(struct work_struct *work)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = container_of(work,
			struct rhac_nvidia_isr_ctx,
			work);
	struct rhac_comm *comm = isr_ctx->comm;

	comm->cur = RHAC_NVIDIA_PIPELINE_FAULT_WRAPUP;

	int err;
	if (UVM_ID_IS_CPU(isr_ctx->dst_id)) {
		atomic_inc(&cpu_fault_cnt);
	} else {
		atomic_inc(&gpu_fault_cnt);
	}

	err = rhac_nvidia_mm_make_resident_cpu(
			isr_ctx->va_block->start,
			isr_ctx->dst_id,
			isr_ctx->comm->copy_mask
			);
	RHAC_ASSERT(!err);

	err = rhac_uvm_va_block_service_locked_local(
			isr_ctx->comm,
			isr_ctx->dst_id,
			isr_ctx->va_block,
			isr_ctx->block_context);
	RHAC_ASSERT(!err);

	// START acquire LOCK, so START release the LOCK
	rhac_nvidia_pipeline_enqueue(isr_ctx->comm, RHAC_NVIDIA_PIPELINE_FAULT_DONE);
}

static void nvidia_pipeline_fault_done(struct work_struct *work)
{
	struct rhac_nvidia_isr_ctx *isr_ctx = container_of(work,
			struct rhac_nvidia_isr_ctx,
			work);
	struct rhac_comm *comm = isr_ctx->comm;

	comm->cur = RHAC_NVIDIA_PIPELINE_FAULT_DONE;

	int err;
	if (UVM_ID_IS_CPU(isr_ctx->dst_id)) {
		err = rhac_uvm_cpu_fault_done(isr_ctx->comm, isr_ctx->va_block);
	} else  {
		err = rhac_uvm_gpu_fault_done(isr_ctx->comm, isr_ctx->mm, isr_ctx->va_block, isr_ctx->batch_context, isr_ctx->lock);
	}
	RHAC_ASSERT(!err);

	RHAC_ASSERT(isr_ctx->comm->number <= atomic_fetch_inc(&num));

	rhac_nvidia_mm_unlock_blk(isr_ctx->va_block->start);

  if (rhac_ctx_get_global()->num_nodes > 1) {
    //if (atomic_read(&comm->req_unlock) || !comm->processing) {
      err = rhac_protocol_post_unlock(comm, isr_ctx->va_block->start);
      RHAC_ASSERT(!err);
    //}
  }

	rhac_comm_free(comm);

	kmem_cache_free(isr_ctx_cache, isr_ctx);
}

void rhac_nvidia_pipeline_enqueue(struct rhac_comm *comm, int step)
{
	RHAC_ASSERT(step < RHAC_NVIDIA_PIPELINE_MAX);
	int err;
	struct rhac_nvidia_isr_ctx *isr_ctx = comm->isr_ctx;

	INIT_WORK(&isr_ctx->work, nvidia_pipeline_handlers[step]);
  err = queue_work(get_queue(step), &isr_ctx->work);
	if (!err) {
		RHAC_ASSERT(false);
		rhac_comm_fail(comm, -EINVAL);
	}
}

int rhac_nvidia_pipeline_init(void)
{
	atomic_set(&cpu_fault_cnt, 0);
	atomic_set(&gpu_fault_cnt, 0);
	atomic_set(&num, 0);

	int i;
	atomic_set(&total_cnt, 0);
	for (i = 0; i < RHAC_NVIDIA_PIPELINE_MAX; i++) {
		atomic_set(&fault_cnt[i], 0);
	}
	for (i = 0; i < PAGES_PER_UVM_VA_BLOCK; ++i) {
		service_context_ro.access_type[i] = UVM_FAULT_ACCESS_TYPE_READ;
		service_context_rw.access_type[i] = UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG;
	}

	isr_ctx_cache = KMEM_CACHE(rhac_nvidia_isr_ctx, SLAB_HWCACHE_ALIGN);
	if (!isr_ctx_cache) return -EINVAL;

	for (i = 0; i < RHAC_NVIDIA_PIPELINE_MAX; i++) {
		int max_active = 20;
		workqueue[i] = alloc_workqueue("rhac-pipeline %d",  WQ_UNBOUND | WQ_MEM_RECLAIM, max_active, i);
	}
  //fault_queue = alloc_workqueue("rhac-pipeline",  WQ_HIGHPRI | WQ_UNBOUND | WQ_MEM_RECLAIM, 20);

	return 0;

	rhac_nvidia_pipeline_deinit();
	return -EINVAL;
}

void rhac_nvidia_pipeline_deinit(void)
{
	RHAC_LOG("# CPU FAULT: %u", atomic_read(&cpu_fault_cnt));
	RHAC_LOG("# GPU FAULT: %u", atomic_read(&gpu_fault_cnt));

	atomic_set(&cpu_fault_cnt, 0);
	atomic_set(&gpu_fault_cnt, 0);
	atomic_set(&num, 0);

	int i;
	for (i = 0; i < RHAC_NVIDIA_PIPELINE_MAX; i++) {
		atomic_set(&fault_cnt[i], 0);
	}
	atomic_set(&total_cnt, 0);

	for (i = 0; i < RHAC_NVIDIA_PIPELINE_MAX; i++) {
		if (workqueue[i]) {
			flush_workqueue(workqueue[i]);
			destroy_workqueue(workqueue[i]);
		}
		workqueue[i] = NULL;
	}
	/*
  if (fault_queue) { 
    flush_workqueue(fault_queue);
    destroy_workqueue(fault_queue);
  }
  fault_queue = NULL;
  */

	if (isr_ctx_cache)
		kmem_cache_destroy(isr_ctx_cache);
	isr_ctx_cache = NULL;

}

void rhac_nvidia_pipeline_flush(void)
{
	int i;
	for (i = 0; i < RHAC_NVIDIA_PIPELINE_MAX; i++) {
		if (workqueue[i]) {
			flush_workqueue(workqueue[i]);
		}
	}
	/*
  if (fault_queue) {
    flush_workqueue(fault_queue);
  }
  */
}
