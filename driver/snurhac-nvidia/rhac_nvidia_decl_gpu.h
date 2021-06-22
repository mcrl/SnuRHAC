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

/*******************************************************************************
    Copyright (c) 2015-2019 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef __RHAC_NVIDIA_DECL_GPU_H__
#define __RHAC_NVIDIA_DECL_GPU_H__

#include "uvm8_forward_decl.h"
#include "uvm8_api.h"
#include "uvm8_global.h"
#include "uvm8_forward_decl.h"
#include "uvm8_va_space.h"
#include "uvm8_va_space_mm.h"
#include "uvm8_va_range.h"
#include "uvm8_va_block.h"
#include "uvm8_mmu.h"
#include "uvm8_hal.h"
#include "uvm8_processors.h"
#include "uvm8_ats_faults.h"
#include "uvm8_procfs.h"
#include "uvm8_tools.h"
#include "nv-kref.h"

typedef enum
{
	// Use this mode when calling from the normal fault servicing path
	FAULT_SERVICE_MODE_REGULAR,

	// Use this mode when servicing faults from the fault cancelling algorithm.
	// In this mode no replays are issued
	FAULT_SERVICE_MODE_CANCEL,
} fault_service_mode_t;

typedef enum
{
	// Fetch a batch of faults from the buffer.
	FAULT_FETCH_MODE_BATCH_ALL,

	// Fetch a batch of faults from the buffer. Stop at the first entry that is
	// not ready yet
	FAULT_FETCH_MODE_BATCH_READY,

	// Fetch all faults in the buffer before PUT. Wait for all faults to become
	// ready
	FAULT_FETCH_MODE_ALL,
} fault_fetch_mode_t;

typedef enum
{
	// Only cancel faults flagged as fatal
	FAULT_CANCEL_MODE_FATAL,

	// Cancel all faults in the batch unconditionally
	FAULT_CANCEL_MODE_ALL,
} fault_cancel_mode_t;

static NV_STATUS cancel_fault_precise_va(uvm_gpu_t *gpu,
		uvm_fault_buffer_entry_t *fault_entry,
		uvm_fault_cancel_va_mode_t cancel_va_mode)
{
	NV_STATUS status;
	uvm_gpu_va_space_t *gpu_va_space;
	uvm_gpu_phys_address_t pdb;
	uvm_push_t push;
	uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->fault_buffer_info.replayable;
	NvU64 offset;

	UVM_ASSERT(gpu->replayable_faults_supported);
	UVM_ASSERT(fault_entry->fatal_reason != UvmEventFatalReasonInvalid);
	UVM_ASSERT(!fault_entry->filtered);

	gpu_va_space = uvm_gpu_va_space_get(fault_entry->va_space, gpu);
	UVM_ASSERT(gpu_va_space);
	pdb = uvm_page_tree_pdb(&gpu_va_space->page_tables)->addr;

	// Record fatal fault event
	rhacuvm_uvm_tools_record_gpu_fatal_fault(gpu->id, fault_entry->va_space, fault_entry, fault_entry->fatal_reason);

	status = uvm_push_begin_acquire(gpu->channel_manager,
			UVM_CHANNEL_TYPE_MEMOPS,
			&replayable_faults->replay_tracker,
			&push,
			"Precise cancel targeting PDB {0x%llx:%s} VA 0x%llx VEID %u with access type %s",
			pdb.address,
			rhacuvm_uvm_aperture_string(pdb.aperture),
			fault_entry->fault_address,
			fault_entry->fault_source.ve_id,
			rhacuvm_uvm_fault_access_type_string(fault_entry->fault_access_type));

	// UVM aligns fault addresses to PAGE_SIZE as it is the smallest mapping
	// and coherence tracking granularity. However, the cancel method requires
	// the original address (4K-aligned) reported in the packet, which is lost
	// at this point. Since the access permissions are the same for the whole
	// 64K page, we issue a cancel per 4K range to make sure that the HW sees
	// the address reported in the packet.
	for (offset = 0; offset < PAGE_SIZE; offset += UVM_PAGE_SIZE_4K) {
		gpu->host_hal->cancel_faults_va(&push, pdb, fault_entry, cancel_va_mode);
		fault_entry->fault_address += UVM_PAGE_SIZE_4K;
	}
	fault_entry->fault_address = UVM_PAGE_ALIGN_DOWN(fault_entry->fault_address - 1);

	// We don't need to put the cancel in the GPU replay tracker since we wait
	// on it immediately.
	status = rhacuvm_uvm_push_end_and_wait(&push);
	if (status != NV_OK) {
		UVM_ERR_PRINT("Failed to wait for pushed VA global fault cancel: %s, GPU %s\n",
				rhacuvm_nvstatusToString(status), gpu->name);
	}

	uvm_tracker_clear(&replayable_faults->replay_tracker);

	return status;
}

static NV_STATUS cancel_faults_precise_va(uvm_gpu_t *gpu,
		uvm_fault_service_batch_context_t *batch_context,
		fault_cancel_mode_t cancel_mode,
		UvmEventFatalReason reason)
{
	NV_STATUS status = NV_OK;
	uvm_va_space_t *va_space = NULL;
	NvU32 i;

	UVM_ASSERT(gpu->fault_cancel_va_supported);
	if (cancel_mode == FAULT_CANCEL_MODE_ALL)
		UVM_ASSERT(reason != UvmEventFatalReasonInvalid);

	for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
		uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];

		UVM_ASSERT(current_entry->va_space);

		if (current_entry->va_space != va_space) {
			// Fault on a different va_space, drop the lock of the old one...
			if (va_space != NULL)
				uvm_va_space_up_read(va_space);

			va_space = current_entry->va_space;

			// ... and take the lock of the new one
			uvm_va_space_down_read(va_space);

			// We don't need to check whether a buffer flush is required
			// (due to VA range destruction).
			// - For cancel_mode == FAULT_CANCEL_MODE_FATAL, once a fault is
			// flagged as fatal we need to cancel it, even if its VA range no
			// longer exists.
			// - For cancel_mode == FAULT_CANCEL_MODE_ALL we don't care about
			// any of this, we just want to trigger RC in RM.
		}

		if (!uvm_processor_mask_test(&va_space->registered_gpu_va_spaces, gpu->id)) {
			// If there is no GPU VA space for the GPU, ignore the fault.
			// This can happen if the GPU VA did not exist in
			// service_fault_batch(), or it was destroyed since then.
			// This is to avoid targetting a PDB that might have been reused
			// by another process.
			continue;
		}

		// Cancel the fault
		if (cancel_mode == FAULT_CANCEL_MODE_ALL || current_entry->is_fatal) {
			uvm_fault_cancel_va_mode_t cancel_va_mode = current_entry->replayable.cancel_va_mode;

			// If cancelling unconditionally and the fault was not fatal,
			// set the cancel reason passed to this function
			if (!current_entry->is_fatal) {
				current_entry->fatal_reason = reason;
				cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
			}

			status = cancel_fault_precise_va(gpu, current_entry, cancel_va_mode);
			if (status != NV_OK)
				break;
		}
	}

	if (va_space != NULL)
		uvm_va_space_up_read(va_space);

	return status;
}

static inline int cmp_fault_instance_ptr(const uvm_fault_buffer_entry_t *a,
		const uvm_fault_buffer_entry_t *b)
{
	int result = uvm_gpu_phys_addr_cmp(a->instance_ptr, b->instance_ptr);
	// On Volta+ we need to sort by {instance_ptr + subctx_id} pair since it can
	// map to a different VA space
	if (result != 0)
		return result;
	return UVM_CMP_DEFAULT(a->fault_source.ve_id, b->fault_source.ve_id);
}
static void fetch_fault_buffer_merge_entry(uvm_fault_buffer_entry_t *current_entry,
    uvm_fault_buffer_entry_t *last_entry)
{
  UVM_ASSERT(last_entry->num_instances > 0);

  ++last_entry->num_instances;
  uvm_fault_access_type_mask_set(&last_entry->access_type_mask, current_entry->fault_access_type);

  if (current_entry->fault_access_type > last_entry->fault_access_type) {
    // If the new entry has a higher access type, it becomes the
    // fault to be serviced. Add the previous one to the list of instances
    current_entry->access_type_mask = last_entry->access_type_mask;
    current_entry->num_instances = last_entry->num_instances;
    last_entry->filtered = true;

    // We only merge faults from different uTLBs if the new fault has an
    // access type with the same or lower level of intrusiveness.
    UVM_ASSERT(current_entry->fault_source.utlb_id == last_entry->fault_source.utlb_id);

    list_replace(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
    list_add(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
  }
  else {
    // Add the new entry to the list of instances for reporting purposes
    current_entry->filtered = true;
    list_add(&current_entry->merged_instances_list, &last_entry->merged_instances_list);
  }
}

static bool fetch_fault_buffer_try_merge_entry(uvm_fault_buffer_entry_t *current_entry,
    uvm_fault_service_batch_context_t *batch_context,
    uvm_fault_utlb_info_t *current_tlb,
    bool is_same_instance_ptr)
{
  uvm_fault_buffer_entry_t *last_tlb_entry = current_tlb->last_fault;
  uvm_fault_buffer_entry_t *last_global_entry = batch_context->last_fault;

  // Check the last coalesced fault and the coalesced fault that was
  // originated from this uTLB
  const bool is_last_tlb_fault = current_tlb->num_pending_faults > 0 &&
    cmp_fault_instance_ptr(current_entry, last_tlb_entry) == 0 &&
    current_entry->fault_address == last_tlb_entry->fault_address;

  // We only merge faults from different uTLBs if the new fault has an
  // access type with the same or lower level of intrusiveness. This is to
  // avoid having to update num_pending_faults on both uTLBs and recomputing
  // last_fault.
  const bool is_last_fault = is_same_instance_ptr &&
    current_entry->fault_address == last_global_entry->fault_address &&
    current_entry->fault_access_type <= last_global_entry->fault_access_type;

  if (is_last_tlb_fault) {
    fetch_fault_buffer_merge_entry(current_entry, last_tlb_entry);
    if (current_entry->fault_access_type > last_tlb_entry->fault_access_type)
      current_tlb->last_fault = current_entry;

    return true;
  }
  else if (is_last_fault) {
    fetch_fault_buffer_merge_entry(current_entry, last_global_entry);
    if (current_entry->fault_access_type > last_global_entry->fault_access_type)
      batch_context->last_fault = current_entry;

    return true;
  }

  return false;
}

static void write_get(uvm_gpu_t *gpu, NvU32 get)
{
  uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->fault_buffer_info.replayable;

  UVM_ASSERT(mutex_is_locked(&gpu->isr.replayable_faults.service_lock.m));

  // Write get on the GPU only if it's changed.
  if (replayable_faults->cached_get == get)
    return;

  replayable_faults->cached_get = get;

  // Update get pointer on the GPU
  gpu->fault_buffer_hal->write_get(gpu, get);
}

static NvU32 is_fatal_fault_in_buffer(uvm_fault_service_batch_context_t *batch_context,
		uvm_fault_buffer_entry_t *fault)
{
	NvU32 i;

	// Fault filtering is not allowed in the TLB-based fault cancel path
	UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

	for (i = 0; i < batch_context->num_cached_faults; ++i) {
		uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];
		if (cmp_fault_instance_ptr(current_entry, fault) == 0 &&
				current_entry->fault_address == fault->fault_address &&
				current_entry->fault_access_type == fault->fault_access_type &&
				current_entry->fault_source.utlb_id == fault->fault_source.utlb_id) {
			return true;
		}
	}

	return false;
}

static NvU32 find_fatal_fault_in_utlb(uvm_fault_service_batch_context_t *batch_context,
		NvU32 utlb_id)
{
	NvU32 i;

	// Fault filtering is not allowed in the TLB-based fault cancel path
	UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

	for (i = 0; i < batch_context->num_cached_faults; ++i) {
		if (batch_context->fault_cache[i].is_fatal &&
				batch_context->fault_cache[i].fault_source.utlb_id == utlb_id)
			return i;
	}

	return i;
}

static bool is_first_fault_in_utlb(uvm_fault_service_batch_context_t *batch_context, NvU32 fault_index)
{
	NvU32 i;
	NvU32 utlb_id = batch_context->fault_cache[fault_index].fault_source.utlb_id;

	for (i = 0; i < fault_index; ++i) {
		uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

		// We have found a prior fault in the same uTLB
		if (current_entry->fault_source.utlb_id == utlb_id)
			return false;
	}

	return true;
}

static void faults_for_page_in_utlb(uvm_fault_service_batch_context_t *batch_context,
		uvm_va_space_t *va_space,
		NvU64 addr,
		NvU32 utlb_id,
		NvU32 *fatal_faults,
		NvU32 *non_fatal_faults)
{
	NvU32 i;

	*fatal_faults = 0;
	*non_fatal_faults = 0;

	// Fault filtering is not allowed in the TLB-based fault cancel path
	UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

	for (i = 0; i < batch_context->num_cached_faults; ++i) {
		uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

		if (current_entry->fault_source.utlb_id == utlb_id &&
				current_entry->va_space == va_space && current_entry->fault_address == addr) {
			// We have found the page
			if (current_entry->is_fatal)
				++(*fatal_faults);
			else
				++(*non_fatal_faults);
		}
	}
}

static bool no_fatal_pages_in_utlb(uvm_fault_service_batch_context_t *batch_context,
		NvU32 start_index,
		NvU32 utlb_id)
{
	NvU32 i;

	// Fault filtering is not allowed in the TLB-based fault cancel path
	UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

	for (i = start_index; i < batch_context->num_cached_faults; ++i) {
		uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

		if (current_entry->fault_source.utlb_id == utlb_id) {
			// We have found a fault for the uTLB
			NvU32 fatal_faults;
			NvU32 non_fatal_faults;

			faults_for_page_in_utlb(batch_context,
					current_entry->va_space,
					current_entry->fault_address,
					utlb_id,
					&fatal_faults,
					&non_fatal_faults);

			if (non_fatal_faults > 0 && fatal_faults == 0)
				return true;
		}
	}

	return false;
}


static void record_fatal_fault_helper(uvm_gpu_t *gpu, uvm_fault_buffer_entry_t *entry, UvmEventFatalReason reason)
{
	uvm_va_space_t *va_space;

	va_space = entry->va_space;
	UVM_ASSERT(va_space);
	uvm_va_space_down_read(va_space);

	// Record fatal fault event
	rhacuvm_uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, entry, reason);
	uvm_va_space_up_read(va_space);
}

static NV_STATUS try_to_cancel_utlbs(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
	NvU32 i;

	// Fault filtering is not allowed in the TLB-based fault cancel path
	UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

	for (i = 0; i < batch_context->num_cached_faults; ++i) {
		uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];
		uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];
		NvU32 gpc_id = current_entry->fault_source.gpc_id;
		NvU32 utlb_id = current_entry->fault_source.utlb_id;
		NvU32 client_id = current_entry->fault_source.client_id;

		// Only fatal faults are considered
		if (!current_entry->is_fatal)
			continue;

		// Only consider uTLBs in lock-down
		if (!utlb->in_lockdown)
			continue;

		// Issue a single cancel per uTLB
		if (utlb->cancelled)
			continue;

		if (is_first_fault_in_utlb(batch_context, i) &&
				!no_fatal_pages_in_utlb(batch_context, i + 1, utlb_id)) {
			NV_STATUS status;

			record_fatal_fault_helper(gpu, current_entry, current_entry->fatal_reason);

			status = rhacuvm_push_cancel_on_gpu(gpu, current_entry->instance_ptr, gpc_id, client_id, &batch_context->tracker);
			if (status != NV_OK)
				return status;

			utlb->cancelled = true;
		}
	}

	return NV_OK;
}

static NV_STATUS cancel_faults_precise_tlb(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
	NV_STATUS status;
	NV_STATUS tracker_status;
	uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->fault_buffer_info.replayable;
	bool first = true;

	UVM_ASSERT(gpu->replayable_faults_supported);

	// 1) Disable prefetching to avoid new requests keep coming and flooding
	//    the buffer
	if (gpu->fault_buffer_info.prefetch_faults_enabled)
		gpu->arch_hal->disable_prefetch_faults(gpu);

	while (1) {
		NvU32 utlb_id;

		// 2) Record one fatal fault per uTLB to check if it shows up after
		// the replay. This is used to handle the case in which the uTLB is
		// being cancelled from behind our backs by RM. See the comment in
		// step 6.
		for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
			uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[utlb_id];

			if (!first && utlb->has_fatal_faults) {
				NvU32 idx = find_fatal_fault_in_utlb(batch_context, utlb_id);
				UVM_ASSERT(idx < batch_context->num_cached_faults);

				utlb->prev_fatal_fault = batch_context->fault_cache[idx];
			}
			else {
				utlb->prev_fatal_fault.fault_address = (NvU64)-1;
			}
		}
		first = false;

		// 3) Flush fault buffer. After this call, all faults from any of the
		// faulting uTLBs are before PUT. New faults from other uTLBs can keep
		// arriving. Therefore, in each iteration we just try to cancel faults
		// from uTLBs that contained fatal faults in the previous iterations
		// and will cause the TLB to stop generating new page faults after the
		// following replay with type UVM_FAULT_REPLAY_TYPE_START_ACK_ALL
		status = rhacuvm_fault_buffer_flush_locked(gpu,
				UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
				UVM_FAULT_REPLAY_TYPE_START_ACK_ALL,
				batch_context);
		if (status != NV_OK)
			break;

		// 4) Wait for replay to finish
		status = rhacuvm_uvm_tracker_wait(&replayable_faults->replay_tracker);
		if (status != NV_OK)
			break;

		batch_context->num_invalid_prefetch_faults = 0;
		batch_context->num_replays                 = 0;
		batch_context->has_fatal_faults            = false;
		batch_context->has_throttled_faults        = false;

		// 5) Fetch all faults from buffer
		rhacuvm_fetch_fault_buffer_entries(gpu, batch_context, FAULT_FETCH_MODE_ALL);
		++batch_context->batch_id;

		UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

		// No more faults left, we are done
		if (batch_context->num_cached_faults == 0)
			break;

		// 6) Check what uTLBs are in lockdown mode and can be cancelled
		for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
			uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[utlb_id];

			utlb->in_lockdown = false;
			utlb->cancelled   = false;

			if (utlb->prev_fatal_fault.fault_address != (NvU64)-1) {
				// If a previously-reported fault shows up again we can "safely"
				// assume that the uTLB that contains it is in lockdown mode
				// and no new translations will show up before cancel.
				// A fatal fault could only be removed behind our backs by RM
				// issuing a cancel, which only happens when RM is resetting the
				// engine. That means the instance pointer can't generate any
				// new faults, so we won't have an ABA problem where a new
				// fault arrives with the same state.
				if (is_fatal_fault_in_buffer(batch_context, &utlb->prev_fatal_fault))
					utlb->in_lockdown = true;
			}
		}

		// 7) Preprocess faults
		status = rhacuvm_preprocess_fault_batch(gpu, batch_context);
		if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
			continue;
		else if (status != NV_OK)
			break;

		// 8) Service all non-fatal faults and mark all non-serviceable faults
		// as fatal
		RHAC_ASSERT(false);
		status = rhacuvm_service_fault_batch(gpu, FAULT_SERVICE_MODE_CANCEL, batch_context);
		if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
			continue;

		UVM_ASSERT(batch_context->num_replays == 0);
		if (status == NV_ERR_NO_MEMORY)
			continue;
		else if (status != NV_OK)
			break;

		// No more fatal faults left, we are done
		if (!batch_context->has_fatal_faults)
			break;

		// 9) Search for uTLBs that contain fatal faults and meet the
		// requirements to be cancelled
		try_to_cancel_utlbs(gpu, batch_context);
	}

	// 10) Re-enable prefetching
	if (gpu->fault_buffer_info.prefetch_faults_enabled)
		gpu->arch_hal->enable_prefetch_faults(gpu);

	if (status == NV_OK)
		status = rhacuvm_push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);

	tracker_status = rhacuvm_uvm_tracker_wait(&batch_context->tracker);

	return status == NV_OK? tracker_status: status;
}

static NV_STATUS cancel_faults_precise(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
	UVM_ASSERT(batch_context->has_fatal_faults);
	if (gpu->fault_cancel_va_supported) {
		return cancel_faults_precise_va(gpu,
				batch_context,
				FAULT_CANCEL_MODE_FATAL,
				UvmEventFatalReasonInvalid);
	}

	return cancel_faults_precise_tlb(gpu, batch_context);
}

static void cancel_fault_batch_tlb(uvm_gpu_t *gpu,
		uvm_fault_service_batch_context_t *batch_context,
		UvmEventFatalReason reason)
{
	NvU32 i;

	// Fault filtering is not allowed in the TLB-based fault cancel path
	UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

	for (i = 0; i < batch_context->num_cached_faults; ++i) {
		NV_STATUS status;
		uvm_fault_buffer_entry_t *current_entry;
		uvm_fault_utlb_info_t *utlb;

		current_entry = &batch_context->fault_cache[i];
		utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];

		// If this uTLB has been already cancelled, skip it
		if (utlb->cancelled)
			continue;

		record_fatal_fault_helper(gpu, current_entry, reason);

		status = rhacuvm_push_cancel_on_gpu(gpu,
				current_entry->instance_ptr,
				current_entry->fault_source.gpc_id,
				current_entry->fault_source.client_id,
				&batch_context->tracker);
		if (status != NV_OK)
			break;

		utlb->cancelled = true;
	}
}

static void cancel_fault_batch(uvm_gpu_t *gpu,
		uvm_fault_service_batch_context_t *batch_context,
		UvmEventFatalReason reason)
{
	if (gpu->fault_cancel_va_supported) {
		cancel_faults_precise_va(gpu, batch_context, FAULT_CANCEL_MODE_ALL, reason);
		return;
	}

	cancel_fault_batch_tlb(gpu, batch_context, reason);
}

static void enable_disable_prefetch_faults(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
	if (!gpu->prefetch_fault_supported)
		return;

	// If more than 66% of faults are invalid prefetch accesses, disable
	// prefetch faults for a while.
	// Some tests rely on this logic (and ratio) to correctly disable prefetch
	// fault reporting. If the logic changes, the tests will have to be changed.
	if (gpu->fault_buffer_info.prefetch_faults_enabled &&
			((batch_context->num_invalid_prefetch_faults * 3 > gpu->fault_buffer_info.max_batch_size * 2 &&
			  (*uvm_perf_reenable_prefetch_faults_lapse_msec_p) > 0) ||
			 (*uvm_enable_builtin_tests_p &&
			  gpu->rm_info.isSimulated &&
			  batch_context->num_invalid_prefetch_faults > 5))) {
		rhacuvm_uvm_gpu_disable_prefetch_faults(gpu);
	}
	else if (!gpu->fault_buffer_info.prefetch_faults_enabled) {
		NvU64 lapse = NV_GETTIME() - gpu->fault_buffer_info.disable_prefetch_faults_timestamp;
		// Reenable prefetch faults after some time
		if (lapse > ((NvU64)(*uvm_perf_reenable_prefetch_faults_lapse_msec_p) * (1000 * 1000)))
			rhacuvm_uvm_gpu_enable_prefetch_faults(gpu);
	}
}

static uvm_fault_access_type_t check_fault_access_permissions(uvm_gpu_t *gpu,
		uvm_va_block_t *va_block,
		uvm_fault_buffer_entry_t *fault_entry,
		bool allow_migration)
{
	NV_STATUS perm_status;

	perm_status = rhacuvm_uvm_va_range_check_logical_permissions(va_block->va_range,
			gpu->id,
			fault_entry->fault_access_type,
			allow_migration);
	if (perm_status == NV_OK)
		return fault_entry->fault_access_type;

	if (fault_entry->fault_access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH) {
		fault_entry->is_invalid_prefetch = true;
		return UVM_FAULT_ACCESS_TYPE_COUNT;
	}

	// At this point we know that some fault instances cannot be serviced
	fault_entry->is_fatal = true;
	fault_entry->fatal_reason = uvm_tools_status_to_fatal_fault_reason(perm_status);

	if (fault_entry->fault_access_type > UVM_FAULT_ACCESS_TYPE_READ) {
		fault_entry->replayable.cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_WRITE_AND_ATOMIC;

		// If there are pending read accesses on the same page, we have to
		// service them before we can cancel the write/atomic faults. So we
		// retry with read fault access type.
		if (uvm_fault_access_type_mask_test(fault_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ)) {
			perm_status = rhacuvm_uvm_va_range_check_logical_permissions(va_block->va_range,
					gpu->id,
					UVM_FAULT_ACCESS_TYPE_READ,
					allow_migration);
			if (perm_status == NV_OK)
				return UVM_FAULT_ACCESS_TYPE_READ;

			// If that didn't succeed, cancel all faults
			fault_entry->replayable.cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
			fault_entry->fatal_reason = uvm_tools_status_to_fatal_fault_reason(perm_status);
		}
	}
	else {
		fault_entry->replayable.cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
	}

	return UVM_FAULT_ACCESS_TYPE_COUNT;
}

static NV_STATUS service_non_managed_fault(uvm_fault_buffer_entry_t *current_entry,
		const uvm_fault_buffer_entry_t *previous_entry,
		NV_STATUS lookup_status,
		uvm_gpu_va_space_t *gpu_va_space,
		struct mm_struct *mm,
		uvm_fault_service_batch_context_t *batch_context,
		uvm_ats_fault_invalidate_t *ats_invalidate,
		uvm_fault_utlb_info_t *utlb)
{
	NV_STATUS status = lookup_status;
	bool is_duplicate = false;
	UVM_ASSERT(utlb->num_pending_faults > 0);
	UVM_ASSERT(lookup_status != NV_OK);

	if (previous_entry) {
		is_duplicate = (current_entry->va_space == previous_entry->va_space) &&
			(current_entry->fault_address == previous_entry->fault_address);

		if (is_duplicate) {
			// Propagate the is_invalid_prefetch flag across all prefetch faults
			// on the page
			if (previous_entry->is_invalid_prefetch)
				current_entry->is_invalid_prefetch = true;

			// If a page is throttled, all faults on the page must be skipped
			if (previous_entry->is_throttled)
				current_entry->is_throttled = true;
		}
	}

	// Generate fault events for all fault packets
	uvm_perf_event_notify_gpu_fault(&current_entry->va_space->perf_events,
			NULL,
			gpu_va_space->gpu->id,
			current_entry,
			batch_context->batch_id,
			is_duplicate);

	if (status != NV_ERR_INVALID_ADDRESS)
		return status;

	if (uvm_can_ats_service_faults(gpu_va_space, mm)) {
		// The VA isn't managed. See if ATS knows about it, unless it is a
		// duplicate and the previous fault was non-fatal so the page has
		// already been serviced
		if (!is_duplicate || previous_entry->is_fatal)
			status = rhacuvm_uvm_ats_service_fault_entry(gpu_va_space, current_entry, ats_invalidate);
		else
			status = NV_OK;
	}
	else {
		// If the VA block cannot be found, set the fatal fault flag,
		// unless it is a prefetch fault
		if (current_entry->fault_access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH) {
			current_entry->is_invalid_prefetch = true;
		}
		else {
			current_entry->is_fatal = true;
			current_entry->fatal_reason = uvm_tools_status_to_fatal_fault_reason(status);
			current_entry->replayable.cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
		}

		// Do not fail due to logical errors
		status = NV_OK;
	}

	if (is_duplicate)
		batch_context->num_duplicate_faults += current_entry->num_instances;
	else
		batch_context->num_duplicate_faults += current_entry->num_instances - 1;

	if (current_entry->is_invalid_prefetch)
		batch_context->num_invalid_prefetch_faults += current_entry->num_instances;

	if (current_entry->is_fatal) {
		utlb->has_fatal_faults = true;
		batch_context->has_fatal_faults = true;
	}

	if (current_entry->is_throttled)
		batch_context->has_throttled_faults = true;

	return status;
}

#endif //__RHAC_NVIDIA_DECL_GPU_H__
