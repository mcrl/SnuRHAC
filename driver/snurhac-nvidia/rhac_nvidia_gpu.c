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

#include <linux/sort.h>

#include "rhac_config.h"
#include "rhac_correlator.h"
#include "rhac_utils.h"

#include "rhac_nvidia_symbols.h"
#include "rhac_nvidia_gpu.h"
#include "rhac_nvidia_helpers.h"
#include "rhac_nvidia_pipeline.h"
#include "rhac_nvidia_common.h"

#include "rhac_comm.h"
#include "rhac_nvidia_decl.h"

#define RHAC_ISR_THRASHING_OFF
static unsigned uvm_perf_fault_coalesce = 1;

static void rhac_fetch_fault_buffer_entries(uvm_gpu_t *gpu,
    uvm_fault_service_batch_context_t *batch_context,
    fault_fetch_mode_t fetch_mode)
{
  NvU32 get;
  NvU32 put;
  NvU32 fault_index;
  NvU32 num_coalesced_faults;
  NvU32 utlb_id;
  uvm_fault_buffer_entry_t *fault_cache;
  uvm_spin_loop_t spin;
  uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->fault_buffer_info.replayable;
  const bool in_pascal_cancel_path = (!gpu->fault_cancel_va_supported && fetch_mode == FAULT_FETCH_MODE_ALL);
  const bool may_filter = uvm_perf_fault_coalesce && !in_pascal_cancel_path;

  // TODO: Bug 1766600: right now uvm locks do not support the synchronization
  //       method used by top and bottom ISR. Add uvm lock assert when it's
  //       supported. Use plain mutex kernel utilities for now.
  UVM_ASSERT(mutex_is_locked(&gpu->isr.replayable_faults.service_lock.m));
  UVM_ASSERT(gpu->replayable_faults_supported);

  fault_cache = batch_context->fault_cache;

  get = replayable_faults->cached_get;

  // Read put pointer from GPU and cache it
  if (get == replayable_faults->cached_put)
    replayable_faults->cached_put = gpu->fault_buffer_hal->read_put(gpu);

  put = replayable_faults->cached_put;

  batch_context->is_single_instance_ptr = true;
  batch_context->last_fault = NULL;

  fault_index = 0;
  num_coalesced_faults = 0;

  // Clear uTLB counters
  for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
    batch_context->utlbs[utlb_id].num_pending_faults = 0;
    batch_context->utlbs[utlb_id].has_fatal_faults = false;
  }
  batch_context->max_utlb_id = 0;

  if (get == put)
    goto done;

  // Parse until get != put and have enough space to cache.
  while ((get != put) && (fetch_mode == FAULT_FETCH_MODE_ALL || fault_index < gpu->fault_buffer_info.max_batch_size)) {
    bool is_same_instance_ptr = true;
    uvm_fault_buffer_entry_t *current_entry = &fault_cache[fault_index];
    uvm_fault_utlb_info_t *current_tlb;

    // We cannot just wait for the last entry (the one pointed by put) to
    // become valid, we have to do it individually since entries can be
    // written out of order
    UVM_SPIN_WHILE(!gpu->fault_buffer_hal->entry_is_valid(gpu, get), &spin) {
      // We have some entry to work on. Let's do the rest later.
      if (fetch_mode != FAULT_FETCH_MODE_ALL &&
          fetch_mode != FAULT_FETCH_MODE_BATCH_ALL &&
          fault_index > 0)
        goto done;
    }

    // Prevent later accesses being moved above the read of the valid bit
    smp_mb__after_atomic();

    // Got valid bit set. Let's cache.
    gpu->fault_buffer_hal->parse_entry(gpu, get, current_entry);

    // The GPU aligns the fault addresses to 4k, but all of our tracking is
    // done in PAGE_SIZE chunks which might be larger.
    current_entry->fault_address = UVM_PAGE_ALIGN_DOWN(current_entry->fault_address);

    // Make sure that all fields in the entry are properly initialized
    current_entry->is_fatal = (current_entry->fault_type >= UVM_FAULT_TYPE_FATAL);

    if (current_entry->is_fatal) {
      // Record the fatal fault event later as we need the va_space locked
      current_entry->fatal_reason = UvmEventFatalReasonInvalidFaultType;
    }
    else {
      current_entry->fatal_reason = UvmEventFatalReasonInvalid;
    }

    current_entry->va_space = NULL;
    current_entry->filtered = false;

    if (current_entry->fault_source.utlb_id > batch_context->max_utlb_id) {
      UVM_ASSERT(current_entry->fault_source.utlb_id < replayable_faults->utlb_count);
      batch_context->max_utlb_id = current_entry->fault_source.utlb_id;
    }

    current_tlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];

    if (fault_index > 0) {
      UVM_ASSERT(batch_context->last_fault);
      is_same_instance_ptr = cmp_fault_instance_ptr(current_entry, batch_context->last_fault) == 0;

      // Coalesce duplicate faults when possible
      if (may_filter && !current_entry->is_fatal) {
        bool merged = fetch_fault_buffer_try_merge_entry(current_entry,
            batch_context,
            current_tlb,
            is_same_instance_ptr);
        if (merged)
          goto next_fault;
      }
    }

    if (batch_context->is_single_instance_ptr && !is_same_instance_ptr)
      batch_context->is_single_instance_ptr = false;

    current_entry->num_instances = 1;
    current_entry->access_type_mask = uvm_fault_access_type_mask_bit(current_entry->fault_access_type);
    INIT_LIST_HEAD(&current_entry->merged_instances_list);

    ++current_tlb->num_pending_faults;
    current_tlb->last_fault = current_entry;
    batch_context->last_fault = current_entry;

    ++num_coalesced_faults;

next_fault:
    ++fault_index;
    ++get;
    if (get == replayable_faults->max_faults)
      get = 0;
  }

done:
  write_get(gpu, get);

  batch_context->num_cached_faults = fault_index;
  batch_context->num_coalesced_faults = num_coalesced_faults;

#ifdef RHAC_DYNAMIC_PREFETCH_NON_READONLY
  if (batch_context->num_cached_faults == 0)
    return;

  NvU32 i;
  NvU32 num_added_entry = 0;

  // traverse all coalesced faults and get correlators
  for (i = 0; i < batch_context->num_cached_faults; ++i) {
    if (fault_cache[i].filtered == true)
      continue;
    num_added_entry += rhac_get_correlators(gpu, &fault_cache[i],
        &fault_cache[fault_index + num_added_entry]);
  }

  // wakeup correlator thread to update correlation table
  rhac_wakeup_correlator_thread(gpu);

  // update final number of faults
  batch_context->num_cached_faults += num_added_entry;
  batch_context->num_coalesced_faults += num_added_entry;
#endif
}


static NV_STATUS rhac_service_batch_managed_faults_in_block_locked(
		struct rhac_comm *comm,
		uvm_gpu_t *gpu,
		uvm_va_block_t *va_block,
		uvm_va_block_retry_t *va_block_retry,
		NvU32 first_fault_index,
		uvm_service_block_context_t *block_context,
		uvm_fault_service_batch_context_t *batch_context)
{
	NV_STATUS status = NV_OK;
	NvU32 i;
	uvm_page_index_t first_page_index;
	uvm_page_index_t last_page_index;
	NvU32 page_fault_count = 0;
	uvm_range_group_range_iter_t iter;
	uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache;
	//uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->fault_buffer_info.replayable;
	uvm_va_space_t *va_space;
		

	// Check that all uvm_fault_access_type_t values can fit into an NvU8
	BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_COUNT > (int)(NvU8)-1);

	uvm_assert_mutex_locked(&va_block->lock);

	// Check that the va_block is still valid
	UVM_ASSERT(va_block->va_range);
	va_space = va_block->va_range->va_space;


	first_page_index = PAGES_PER_UVM_VA_BLOCK;
	last_page_index = 0;

	// Initialize fault service block context
	uvm_processor_mask_zero(&block_context->resident_processors);
	block_context->thrashing_pin_count = 0;
	block_context->read_duplicate_count = 0;

	rhacuvm_uvm_range_group_range_migratability_iter_first(va_space, va_block->start, va_block->end, &iter);

	// The first entry is guaranteed to fall within this block
	UVM_ASSERT(ordered_fault_cache[first_fault_index]->va_space == va_space);
	UVM_ASSERT(ordered_fault_cache[first_fault_index]->fault_address >= va_block->start);
	UVM_ASSERT(ordered_fault_cache[first_fault_index]->fault_address <= va_block->end);

	// Scan the sorted array and notify the fault event for all fault entries
	// in the block
	for (i = first_fault_index;
			i < batch_context->num_coalesced_faults &&
			ordered_fault_cache[i]->va_space == va_space &&
			ordered_fault_cache[i]->fault_address <= va_block->end;
			++i) {
		uvm_fault_buffer_entry_t *current_entry = ordered_fault_cache[i];
		const uvm_fault_buffer_entry_t *previous_entry = NULL;
		bool read_duplicate;
		uvm_processor_id_t new_residency;
		uvm_perf_thrashing_hint_t thrashing_hint;
		uvm_page_index_t page_index = uvm_va_block_cpu_page_index(va_block, current_entry->fault_address);
		bool is_duplicate = false;
		uvm_fault_access_type_t service_access_type;
		NvU32 service_access_type_mask;

		UVM_ASSERT(current_entry->fault_access_type ==
				uvm_fault_access_type_mask_highest(current_entry->access_type_mask));

		current_entry->is_fatal            = false;
		current_entry->is_throttled        = false;
		current_entry->is_invalid_prefetch = false;

		if (i > first_fault_index) {
			previous_entry = ordered_fault_cache[i - 1];
			is_duplicate = current_entry->fault_address == previous_entry->fault_address;
		}

		if (block_context->num_retries == 0) {
			uvm_perf_event_notify_gpu_fault(&va_space->perf_events,
					va_block,
					gpu->id,
					current_entry,
					batch_context->batch_id,
					is_duplicate);
		}

		// Service the most intrusive fault per page, only. Waive the rest
		if (is_duplicate) {
			// Propagate the is_invalid_prefetch flag across all prefetch
			// faults on the page
			current_entry->is_invalid_prefetch = previous_entry->is_invalid_prefetch;

			// If a page is throttled, all faults on the page must be skipped
			current_entry->is_throttled = previous_entry->is_throttled;

			// The previous fault was non-fatal so the page has been already
			// serviced
			if (!previous_entry->is_fatal) {
				goto next;
			}
		}

		// Ensure that the migratability iterator covers the current fault
		// address
		while (iter.end < current_entry->fault_address)
			rhacuvm_uvm_range_group_range_migratability_iter_next(va_space, &iter, va_block->end);

		UVM_ASSERT(iter.start <= current_entry->fault_address && iter.end >= current_entry->fault_address);

		service_access_type = check_fault_access_permissions(gpu, va_block, current_entry, iter.migratable);

		// Do not exit early due to logical errors such as access permission
		// violation.
		if (service_access_type == UVM_FAULT_ACCESS_TYPE_COUNT) {
			goto next;
		}

		if (service_access_type != current_entry->fault_access_type) {
			// Some of the fault instances cannot be serviced due to invalid
			// access permissions. Recompute the access type service mask to
			// service the rest.
			UVM_ASSERT(service_access_type < current_entry->fault_access_type);
			service_access_type_mask = uvm_fault_access_type_mask_bit(service_access_type);
		}
		else {
			service_access_type_mask = current_entry->access_type_mask;
		}

		// If the GPU already has the necessary access permission, the fault
		// does not need to be serviced
		if (rhacuvm_uvm_va_block_page_is_gpu_authorized(va_block,
					page_index,
					gpu->id,
					uvm_fault_access_type_to_prot(service_access_type))) {
			goto next;
		}

		thrashing_hint = rhacuvm_uvm_perf_thrashing_get_hint(va_block, current_entry->fault_address, gpu->id);
#ifdef RHAC_ISR_THRASHING_OFF
		thrashing_hint.type = UVM_PERF_THRASHING_HINT_TYPE_NONE;
#endif
		if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_THROTTLE) {
			// Throttling is implemented by sleeping in the fault handler on
			// the CPU and by continuing to process faults on other pages on
			// the GPU
			current_entry->is_throttled = true;
			goto next;
		}
		else if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_PIN) {
			if (block_context->thrashing_pin_count++ == 0)
				uvm_page_mask_zero(&block_context->thrashing_pin_mask);

			uvm_page_mask_set(&block_context->thrashing_pin_mask, page_index);
		}

		RHAC_ASSERT(thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_NONE);

		// Compute new residency and update the masks
		new_residency = rhacuvm_uvm_va_block_select_residency(va_block,
				page_index,
				gpu->id,
				service_access_type_mask,
				&thrashing_hint,
				UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS,
				&read_duplicate);

		RHAC_ASSERT(uvm_id_equal(new_residency, gpu->id));

//		// RELAXED: read_duplicate is set depending on access type
//		if (service_access_type <= UVM_FAULT_ACCESS_TYPE_READ) {
//			read_duplicate = true;
//		}

		if (!uvm_processor_mask_test_and_set(&block_context->resident_processors, new_residency))
			uvm_page_mask_zero(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency);

		uvm_page_mask_set(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency, page_index);

		if (read_duplicate) {
			if (block_context->read_duplicate_count++ == 0)
				uvm_page_mask_zero(&block_context->read_duplicate_mask);

			uvm_page_mask_set(&block_context->read_duplicate_mask, page_index);
		}

		++page_fault_count;

		block_context->access_type[page_index] = service_access_type;

		if (page_index < first_page_index)
			first_page_index = page_index;
		if (page_index > last_page_index)
			last_page_index = page_index;

next:
		// Only update counters the first time since logical permissions cannot
		// change while we hold the VA space lock
		// TODO: Bug 1750144: That might not be true with HMM.
		if (block_context->num_retries == 0) {
			uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];

			if (current_entry->is_invalid_prefetch)
				batch_context->num_invalid_prefetch_faults += current_entry->num_instances;

			if (is_duplicate)
				batch_context->num_duplicate_faults += current_entry->num_instances;
			else
				batch_context->num_duplicate_faults += current_entry->num_instances - 1;

			if (current_entry->is_throttled)
				batch_context->has_throttled_faults = true;

			if (current_entry->is_fatal) {
				utlb->has_fatal_faults = true;
				batch_context->has_fatal_faults = true;
			}
		}
	}

	// Apply the changes computed in the fault service block context, if there
	// are pages to be serviced
	if (page_fault_count > 0) {
		block_context->region = uvm_va_block_region(first_page_index, last_page_index + 1);
		status = rhac_uvm_va_block_service_locked_global(comm, gpu->id, va_block, block_context);
	}

	++block_context->num_retries;

	if (status == NV_OK && batch_context->has_fatal_faults)
		status = rhacuvm_uvm_va_block_set_cancel(va_block, gpu);

	return status;
}

int rhac_uvm_gpu_fault_start(
		struct rhac_comm *comm,
		uvm_gpu_t *gpu,
		struct mm_struct *mm,
		uvm_va_block_t *va_block,
		NvU32 first_fault_index,
		uvm_service_block_context_t *service_block_context,
		uvm_fault_service_batch_context_t *batch_context)
{
	NV_STATUS status;
	uvm_va_block_retry_t va_block_retry;

	service_block_context->operation = UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS;
	service_block_context->num_retries = 0;
	service_block_context->block_context.mm = mm;

	RHAC_ASSERT((va_block->start & (0x1fffff)) == 0);

	status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &va_block_retry,
			rhac_service_batch_managed_faults_in_block_locked(
				comm,
				gpu,
				va_block,
				&va_block_retry,
				first_fault_index,
				service_block_context,
				batch_context));

	return status != NV_OK;
}

int rhac_uvm_gpu_fault_done(
		struct rhac_comm *comm,
		struct mm_struct *mm,
		uvm_va_block_t *va_block,
		uvm_fault_service_batch_context_t *batch_context,
		uvm_mutex_t *lock
		)
{
	NV_STATUS tracker_status = NV_OK;

	uvm_mutex_lock(lock);
	tracker_status = rhacuvm_uvm_tracker_add_tracker_safe(&batch_context->tracker, &va_block->tracker);
	uvm_mutex_unlock(lock);


	return tracker_status != NV_OK;
}

static NV_STATUS rhac_service_fault_batch(
		struct rhac_comm *pa,
		uvm_gpu_t *gpu,
		fault_service_mode_t service_mode,
		uvm_fault_service_batch_context_t *batch_context,
		uvm_mutex_t *lock
		)
{
	NV_STATUS status = NV_OK;
	NvU32 i;
	uvm_va_space_t *va_space = NULL;
	uvm_gpu_va_space_t *gpu_va_space = NULL;
	uvm_ats_fault_invalidate_t *ats_invalidate = &gpu->fault_buffer_info.replayable.ats_invalidate;
	const bool replay_per_va_block = service_mode != FAULT_SERVICE_MODE_CANCEL &&
		gpu->fault_buffer_info.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK;
	struct mm_struct *mm = NULL;

	UVM_ASSERT(gpu->replayable_faults_supported);

	ats_invalidate->write_faults_in_batch = false;

	for (i = 0; i < batch_context->num_coalesced_faults;) {
		uvm_va_block_t *va_block;
		NvU32 block_faults;
		uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
		uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];

		UVM_ASSERT(current_entry->va_space);

		if (current_entry->va_space != va_space) {
			if (va_space != NULL) {
				status = rhacuvm_uvm_ats_invalidate_tlbs(gpu_va_space, ats_invalidate, &batch_context->tracker);
				if (status != NV_OK)
					goto fail;

				uvm_va_space_up_read(va_space);
				if (mm) {
					uvm_up_read_mmap_sem(&mm->mmap_sem);
					rhacuvm_uvm_va_space_mm_release(va_space);
					mm = NULL;
				}
			}

			va_space = current_entry->va_space;
			mm = rhacuvm_uvm_va_space_mm_retain(va_space);
			if (mm) uvm_down_read_mmap_sem(&mm->mmap_sem);

			uvm_va_space_down_read(va_space);

			gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);
			if (gpu_va_space && gpu_va_space->needs_fault_buffer_flush) {
				status = rhacuvm_fault_buffer_flush_locked(gpu,
						UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
						UVM_FAULT_REPLAY_TYPE_START,
						batch_context);
				gpu_va_space->needs_fault_buffer_flush = false;

				if (status == NV_OK)
					status = NV_WARN_MORE_PROCESSING_REQUIRED;

				break;
			}
		}

		if (current_entry->is_fatal) {
			++i;
			batch_context->has_fatal_faults = true;
			utlb->has_fatal_faults = true;
			UVM_ASSERT(utlb->num_pending_faults > 0);
			RHAC_ASSERT(false);
			continue;
		}

		if (!uvm_processor_mask_test(&va_space->registered_gpu_va_spaces, gpu->id)) {
			++i;
			RHAC_ASSERT(false);
			continue;
		}

		status = rhacuvm_uvm_va_block_find_create(current_entry->va_space, current_entry->fault_address, &va_block);
		if (status == NV_OK) {

			status = rhac_nvidia_pipeline_gpu_fault(pa, gpu, mm, va_block, i, batch_context, &block_faults, lock);
			if (status != NV_OK) {
				goto fail;
			}

			i += block_faults;
		}
		else {
			RHAC_ASSERT(false);
			const uvm_fault_buffer_entry_t *previous_entry = i == 0 ? NULL : batch_context->ordered_fault_cache[i - 1];

			status = service_non_managed_fault(current_entry,
					previous_entry,
					status,
					gpu_va_space,
					mm,
					batch_context,
					ats_invalidate,
					utlb);

			if (status != NV_OK)
				goto fail;

			++i;
			continue;
		}

		if (replay_per_va_block) {
			RHAC_ASSERT(false);
			status = rhacuvm_push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
			if (status != NV_OK)
				goto fail;

			++batch_context->batch_id;
		}
	}

	if (va_space != NULL) {
		NV_STATUS invalidate_status = rhacuvm_uvm_ats_invalidate_tlbs(gpu_va_space, ats_invalidate, &batch_context->tracker);
		if (invalidate_status != NV_OK)
			status = invalidate_status;
	} 

fail:
	if (va_space != NULL) {
		uvm_va_space_up_read(va_space);
		if (mm) {
			uvm_up_read_mmap_sem(&mm->mmap_sem);
			rhacuvm_uvm_va_space_mm_release(va_space);
		}
	}

	return status;
}

static void rhac_gpu_service_replayable_faults(uvm_gpu_t *gpu)
{
	NvU32 num_replays = 0;
	NvU32 num_batches = 0;
	NvU32 num_throttled = 0;
	NV_STATUS status = NV_OK;
	uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->fault_buffer_info.replayable;
	uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

	UVM_ASSERT(gpu->replayable_faults_supported);

	uvm_tracker_init(&batch_context->tracker);

	struct rhac_comm *comm = rhac_comm_alloc();
	RHAC_ASSERT(comm);
  comm->type = 1;

	while (1) {
		if (num_throttled >= *uvm_perf_fault_max_throttle_per_service_p ||
				num_batches >= *uvm_perf_fault_max_batches_per_service_p) {
			break;
		}

		batch_context->num_invalid_prefetch_faults = 0;
		batch_context->num_duplicate_faults        = 0;
		batch_context->num_replays                 = 0;
		batch_context->has_fatal_faults            = false;
		batch_context->has_throttled_faults        = false;

		//rhacuvm_fetch_fault_buffer_entries(gpu, batch_context, FAULT_FETCH_MODE_BATCH_READY);
		rhac_fetch_fault_buffer_entries(gpu, batch_context, FAULT_FETCH_MODE_BATCH_READY);
		if (batch_context->num_cached_faults == 0)
			break;

		++batch_context->batch_id;

		status = rhacuvm_preprocess_fault_batch(gpu, batch_context);

		num_replays += batch_context->num_replays;

		if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
			continue;
		else if (status != NV_OK)
			break;

		uvm_mutex_t lock;
		uvm_mutex_init(&lock, UVM_LOCK_ORDER_VA_BLOCK);
		status = rhac_service_fault_batch(comm, gpu, FAULT_SERVICE_MODE_REGULAR, batch_context, &lock);
		if (rhac_comm_wait(comm)) {
			RHAC_ASSERT(false);
			status = NV_ERR_TIMEOUT;
			batch_context->has_fatal_faults = true;
			break;
		}

		num_replays += batch_context->num_replays;

		if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
			continue;

		enable_disable_prefetch_faults(gpu, batch_context);

		if (status != NV_OK) {
			cancel_fault_batch(gpu, batch_context, uvm_tools_status_to_fatal_fault_reason(status));
			break;
		}

		if (batch_context->has_fatal_faults) {
			status = rhacuvm_uvm_tracker_wait(&batch_context->tracker);
			if (status == NV_OK)
				status = cancel_faults_precise(gpu, batch_context);

			break;
		}

		if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH) {
			status = rhacuvm_push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
			if (status != NV_OK)
				break;
			++num_replays;
		}
		else if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH) {
			uvm_gpu_buffer_flush_mode_t flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT;

			if (batch_context->num_duplicate_faults * 100 >
					batch_context->num_cached_faults * replayable_faults->replay_update_put_ratio) {
				flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
			}

			status = rhacuvm_fault_buffer_flush_locked(gpu,
					flush_mode,
					UVM_FAULT_REPLAY_TYPE_START,
					batch_context);
			if (status != NV_OK)
				break;
			++num_replays;
			status = rhacuvm_uvm_tracker_wait(&replayable_faults->replay_tracker);
			if (status != NV_OK)
				break;
		}

		if (batch_context->has_throttled_faults)
			++num_throttled;

		++num_batches;
	}
	rhac_comm_free(comm);

	if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
		status = NV_OK;

	if ((status == NV_OK && replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_ONCE) ||
			num_replays == 0) {
		status = rhacuvm_push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
	}

	rhacuvm_uvm_tracker_deinit(&batch_context->tracker);

	if (status != NV_OK)
		UVM_DBG_PRINT("Error servicing replayable faults on GPU: %s\n", gpu->name);
}

static void rhac_replayable_faults_isr_bottom_half(void *args)
{
	uvm_gpu_t *gpu = (uvm_gpu_t *)args;
	unsigned int cpu;

	UVM_ASSERT(gpu->replayable_faults_supported);

	cpu = get_cpu();
	++gpu->isr.replayable_faults.stats.bottom_half_count;
	cpumask_set_cpu(cpu, &gpu->isr.replayable_faults.stats.cpus_used_mask);
	++gpu->isr.replayable_faults.stats.cpu_exec_count[cpu];
	put_cpu();

	//RHAC_LOG("GPU START");
	rhac_gpu_service_replayable_faults(gpu);
	//RHAC_LOG("GPU END");

	rhacuvm_uvm_gpu_replayable_faults_isr_unlock(gpu);
	rhacuvm_uvm_gpu_kref_put(gpu);

}

static void rhac_replayable_faults_isr_bottom_half_entry(void *args)
{
	UVM_ENTRY_VOID(rhac_replayable_faults_isr_bottom_half(args));
}

static bool gpu_initialized;
static nv_q_func_t gpu_bh;

int rhac_nvidia_gpu_init(void)
{
	uvm_gpu_t *gpu;
	nv_kthread_q_item_t *bh;


	uvm_mutex_lock(&g_uvm_global_p->global_lock);
	uvm_spin_lock_irqsave(&g_uvm_global_p->gpu_table_lock);
	for_each_global_gpu(gpu) {
		nv_kref_get(&gpu->gpu_kref);
		rhacuvm_uvm_gpu_replayable_faults_isr_lock(gpu);
		bh = &gpu->isr.replayable_faults.bottom_half_q_item;
		if (!gpu_bh) gpu_bh = bh->function_to_run;
		bh->function_to_run = rhac_replayable_faults_isr_bottom_half_entry;
#ifdef RHAC_DYNAMIC_PREFETCH_NON_READONLY
    uvm_replayable_fault_buffer_info_t *replayable_faults =
      &gpu->fault_buffer_info.replayable;
    uvm_fault_service_batch_context_t *batch_context =
      &replayable_faults->batch_service_context;

    if (batch_context->fault_cache)
      rhacuvm_uvm_kvfree(batch_context->fault_cache);
    batch_context->fault_cache = uvm_kvmalloc_zero(
        replayable_faults->max_faults *
        sizeof(*batch_context->fault_cache) *
        RHAC_CORRELATOR_LEVEL * RHAC_CORRELATOR_SUCCS);
    if (!batch_context->fault_cache)
      return NV_ERR_NO_MEMORY;

    if (batch_context->ordered_fault_cache)
      rhacuvm_uvm_kvfree(batch_context->ordered_fault_cache);
    batch_context->ordered_fault_cache = uvm_kvmalloc_zero(
        replayable_faults->max_faults *
        sizeof(*batch_context->ordered_fault_cache) *
        RHAC_CORRELATOR_LEVEL * RHAC_CORRELATOR_SUCCS);
    if (!batch_context->ordered_fault_cache)
      return NV_ERR_NO_MEMORY;
#endif
		rhacuvm_uvm_gpu_replayable_faults_isr_unlock(gpu);
	}
	uvm_spin_unlock_irqrestore(&g_uvm_global_p->gpu_table_lock);
	uvm_mutex_unlock(&g_uvm_global_p->global_lock);

	gpu_initialized = true;
	return 0;
}

void rhac_nvidia_gpu_deinit(void)
{
	unsigned int i;
	uvm_gpu_t *gpu;

	if (gpu_initialized) {
		uvm_mutex_lock(&g_uvm_global_p->global_lock);
		uvm_spin_lock_irqsave(&g_uvm_global_p->gpu_table_lock);
		for_each_global_gpu(gpu) {
			rhacuvm_uvm_gpu_replayable_faults_isr_lock(gpu);
			i = uvm_id_gpu_index(gpu->id);

#ifdef RHAC_DYNAMIC_PREFETCH_NON_READONLY
// FIXME
      /*
      uvm_replayable_fault_buffer_info_t *replayable_faults =
        &gpu->fault_buffer_info.replayable;
      uvm_fault_service_batch_context_t *batch_context =
        &replayable_faults->batch_service_context;

      if (batch_context->fault_cache)
        rhacuvm_uvm_kvfree(batch_context->fault_cache);
      batch_context->fault_cache = uvm_kvmalloc_zero(
          replayable_faults->max_faults *
          sizeof(*batch_context->fault_cache));

      if (batch_context->ordered_fault_cache)
        rhacuvm_uvm_kvfree(batch_context->ordered_fault_cache);
      batch_context->ordered_fault_cache = uvm_kvmalloc_zero(
          replayable_faults->max_faults *
          sizeof(*batch_context->ordered_fault_cache));
      */
#endif
			gpu->isr.replayable_faults.bottom_half_q_item.function_to_run = gpu_bh;
			rhacuvm_uvm_gpu_replayable_faults_isr_unlock(gpu);

			rhacuvm_uvm_gpu_kref_put(gpu);
		}
		uvm_spin_unlock_irqrestore(&g_uvm_global_p->gpu_table_lock);
		uvm_mutex_unlock(&g_uvm_global_p->global_lock);
		gpu_bh = NULL;
		gpu_initialized = false;
	}
}
