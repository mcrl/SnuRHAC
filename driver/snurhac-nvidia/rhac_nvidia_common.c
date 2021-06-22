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

#include <linux/swap.h>
#include <linux/mm.h>

#include "rhac_config.h"
#include "rhac_ctx.h"
#include "rhac_nvidia_pipeline.h"
#include "rhac_nvidia_prefetch.h"

#include "rhac_nvidia_symbols.h"
#include "rhac_utils.h"
#include "rhac_nvidia_common.h"
#include "rhac_nvidia_helpers.h"
#include "rhac_comm.h"
#include "rhac_pdsc.h"

#include "rhac_nvidia_decl.h"
#include "rhac_protocol.h"

#define RHAC_ISR_PREFETCH_OFF

static uvm_prot_t rhac_compute_new_permission(uvm_va_block_t *va_block,
		uvm_page_index_t page_index,
		uvm_processor_id_t fault_processor_id,
		uvm_processor_id_t new_residency,
		uvm_fault_access_type_t access_type)
{
	uvm_va_range_t *va_range;
	uvm_va_space_t *va_space;
	uvm_prot_t logical_prot, new_prot;

	// TODO: Bug 1766432: Refactor into policies. Current policy is
	//       query_promote: upgrade access privileges to avoid future faults IF
	//       they don't trigger further revocations.
	va_range = va_block->va_range;
	UVM_ASSERT(va_range);
	va_space = va_range->va_space;
	UVM_ASSERT(va_space);

	new_prot = uvm_fault_access_type_to_prot(access_type);
	logical_prot = rhacuvm_uvm_va_range_logical_prot(va_range);


	UVM_ASSERT(logical_prot >= new_prot);

	if (logical_prot > UVM_PROT_READ_ONLY && new_prot == UVM_PROT_READ_ONLY &&
			!block_region_might_read_duplicate(va_block, uvm_va_block_region_for_page(page_index))) {
		uvm_processor_mask_t processors_with_atomic_mapping;
		uvm_processor_mask_t revoke_processors;

		rhacuvm_uvm_va_block_page_authorized_processors(va_block,
				page_index,
				UVM_PROT_READ_WRITE_ATOMIC,
				&processors_with_atomic_mapping);

		uvm_processor_mask_andnot(&revoke_processors,
				&processors_with_atomic_mapping,
				&va_space->has_native_atomics[uvm_id_value(new_residency)]);

		// Only check if there are no faultable processors in the revoke processors mask
		uvm_processor_mask_and(&revoke_processors, &revoke_processors, &va_space->faultable_processors);

		if (uvm_processor_mask_empty(&revoke_processors))
			new_prot = UVM_PROT_READ_WRITE;
	}

	if (logical_prot == UVM_PROT_READ_WRITE_ATOMIC && new_prot == UVM_PROT_READ_WRITE) {
		if (uvm_processor_mask_test(&va_space->has_native_atomics[uvm_id_value(new_residency)], fault_processor_id))
			new_prot = UVM_PROT_READ_WRITE_ATOMIC;
	}

	return new_prot;
}

static NV_STATUS rhac_uvm_post_block_fault(
		struct rhac_comm *comm,
		uvm_processor_id_t processor_id,
		uvm_va_block_t *block,
		uvm_va_block_context_t *block_context,
		uvm_service_block_context_t *service_context,
		uvm_processor_id_t dst_id, 
		const uvm_page_mask_t *fault_page_mask
		)
{
	int err;

	uint64_t blk_vaddr = block->start;

	uvm_page_mask_t prot_mask, atom_mask;
	uvm_page_mask_zero(&prot_mask);
	uvm_page_mask_zero(&atom_mask);

	uvm_page_index_t page_idx;
	uvm_prot_t access_type;
	for_each_va_block_page_in_mask(page_idx, fault_page_mask, block) {
		access_type = rhac_compute_new_permission(block,
				page_idx,
				processor_id,
				dst_id,
				service_context->access_type[page_idx]);

		RHAC_ASSERT(access_type < UVM_PROT_MAX && access_type > UVM_PROT_NONE);
		if (access_type > UVM_PROT_READ_ONLY)
			uvm_page_mask_set(&prot_mask, page_idx);
		if (access_type == UVM_PROT_READ_WRITE_ATOMIC)
			uvm_page_mask_set(&atom_mask, page_idx);
	}

  comm->processing = true;
  if (rhac_ctx_get_global()->num_nodes <= 1) {
      rhac_nvidia_pipeline_enqueue(comm, comm->next);
  } else {
    err = rhac_protocol_post_update(comm, blk_vaddr, fault_page_mask->bitmap, prot_mask.bitmap, atom_mask.bitmap);
    if (err) {
      return NV_ERR_GENERIC;
    }
  }

	return NV_OK;
}

static void rhac_set_prefetch_pages(
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		uvm_va_range_t *va_range,
		const uvm_va_space_t *va_space,
		uvm_service_block_context_t *service_context,
		uvm_perf_prefetch_hint_t *prefetch_hint
		)
{
  uvm_processor_id_t new_residency;

  // Performance heuristics policy: we only consider prefetching when there
  // are migrations to a single processor, only.
  if (uvm_processor_mask_get_count(&service_context->resident_processors) == 1) {
    uvm_page_mask_t *new_residency_mask;

    new_residency = uvm_processor_mask_find_first_id(&service_context->resident_processors);
    new_residency_mask = &service_context->per_processor_masks[uvm_id_value(new_residency)].new_residency;

    // Update prefetch tracking structure with the pages that will migrate
    // due to faults
    rhacuvm_uvm_perf_prefetch_prenotify_fault_migrations(va_block,
        &service_context->block_context,
        new_residency,
        new_residency_mask,
        service_context->region);

#ifdef RHAC_DYNAMIC_PREFETCH_READONLY
    *prefetch_hint = rhacuvm_uvm_perf_prefetch_get_hint(va_block, new_residency_mask);
    // Obtain the prefetch hint and give a fake fault access type to the
    // prefetched pages
    if (UVM_ID_IS_VALID(prefetch_hint->residency)) {
      // we do dynamic prefetching for GPU only
      if (!UVM_ID_IS_CPU(new_residency) &&
          va_range->read_duplication == UVM_READ_DUPLICATION_ENABLED) {
        UVM_ASSERT(prefetch_hint->prefetch_pages_mask != NULL);
        uvm_page_index_t page_index;
        for_each_va_block_page_in_mask(page_index, prefetch_hint->prefetch_pages_mask, va_block) {
          UVM_ASSERT(!uvm_page_mask_test(new_residency_mask, page_index));
          service_context->access_type[page_index] = UVM_FAULT_ACCESS_TYPE_PREFETCH;
          if (uvm_va_range_is_read_duplicate(va_range) ||
              (va_range->read_duplication != UVM_READ_DUPLICATION_DISABLED &&
               uvm_page_mask_test(&va_block->read_duplicated_pages, page_index))) {
            if (service_context->read_duplicate_count++ == 0)
              uvm_page_mask_zero(&service_context->read_duplicate_mask);
            uvm_page_mask_set(&service_context->read_duplicate_mask, page_index);
          }
        }
      }
      else {
        prefetch_hint->residency = UVM_ID_INVALID;
        uvm_page_mask_zero((uvm_page_mask_t*)prefetch_hint->prefetch_pages_mask);
      }
      service_context->region = uvm_va_block_region_from_block(va_block);
    }
#else
    *prefetch_hint = rhacuvm_uvm_perf_prefetch_get_hint(va_block, new_residency_mask);
    // Obtain the prefetch hint and give a fake fault access type to the
    // prefetched pages
    if (UVM_ID_IS_VALID(prefetch_hint->residency)) {
      UVM_ASSERT(prefetch_hint->prefetch_pages_mask != NULL);
#ifdef RHAC_ISR_PREFETCH_OFF
      prefetch_hint->residency = UVM_ID_INVALID;
      uvm_page_mask_zero((uvm_page_mask_t*)prefetch_hint->prefetch_pages_mask);
#else
      uvm_page_index_t page_index;
      for_each_va_block_page_in_mask(page_index, prefetch_hint->prefetch_pages_mask, va_block) {
        UVM_ASSERT(!uvm_page_mask_test(new_residency_mask, page_index));
        service_context->access_type[page_index] = UVM_FAULT_ACCESS_TYPE_PREFETCH;
        if (uvm_va_range_is_read_duplicate(va_range) ||
            (va_range->read_duplication != UVM_READ_DUPLICATION_DISABLED &&
             uvm_page_mask_test(&va_block->read_duplicated_pages, page_index))) {
          if (service_context->read_duplicate_count++ == 0)
            uvm_page_mask_zero(&service_context->read_duplicate_mask);
          uvm_page_mask_set(&service_context->read_duplicate_mask, page_index);
        }
      }
#endif
      service_context->region = uvm_va_block_region_from_block(va_block);
    }
#endif
  }
}

int block_copy_resident_pages_local(
		uvm_va_block_t *block,
		uvm_va_block_context_t *block_context,
		uvm_processor_id_t dst_id,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *page_mask,
		const uvm_page_mask_t *prefetch_page_mask,
		uvm_va_block_transfer_mode_t transfer_mode)
{
	NV_STATUS status = NV_OK;
	NV_STATUS tracker_status;
	uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
	uvm_page_mask_t *resident_mask = rhacuvm_uvm_va_block_resident_mask_get(block, dst_id);
	NvU32 missing_pages_count;
	NvU32 pages_copied;
	NvU32 pages_copied_to_cpu;
	uvm_processor_mask_t src_processor_mask;
	uvm_page_mask_t *copy_page_mask = &block_context->make_resident.page_mask;
	uvm_page_mask_t *migrated_pages = &block_context->make_resident.pages_migrated;
	uvm_page_mask_t *staged_pages = &block_context->make_resident.pages_staged;
	block_transfer_mode_internal_t transfer_mode_internal;

	uvm_page_mask_zero(migrated_pages);

	if (page_mask) {
		uvm_page_mask_andnot(copy_page_mask, page_mask, resident_mask);
	}
	else {
		uvm_page_mask_complement(copy_page_mask, resident_mask);
	}

	missing_pages_count = uvm_page_mask_region_weight(copy_page_mask, region);

	// If nothing needs to be copied, just check if we need to break
	// read-duplication (i.e. transfer_mode is UVM_VA_BLOCK_TRANSFER_MODE_MOVE)
	if (missing_pages_count == 0) {
		goto out;
	}

	// TODO: Bug 1753731: Add P2P2P copies staged through a GPU
	// TODO: Bug 1753731: When a page is resident in multiple locations due to
	//       read-duplication, spread out the source of the copy so we don't
	//       bottleneck on a single location.

	uvm_processor_mask_zero(&src_processor_mask);

	if (!uvm_id_equal(dst_id, UVM_ID_CPU)) {
		// If the destination is a GPU, first move everything from processors
		// with copy access supported. Notably this will move pages from the CPU
		// as well even if later some extra copies from CPU are required for
		// staged copies.
		uvm_processor_mask_and(&src_processor_mask, block_get_can_copy_from_mask(block, dst_id), &block->resident);
		uvm_processor_mask_clear(&src_processor_mask, dst_id);

		status = block_copy_resident_pages_mask(block,
				block_context,
				dst_id,
				&src_processor_mask,
				region,
				copy_page_mask,
				prefetch_page_mask,
				transfer_mode == UVM_VA_BLOCK_TRANSFER_MODE_COPY?
				BLOCK_TRANSFER_MODE_INTERNAL_COPY:
				BLOCK_TRANSFER_MODE_INTERNAL_MOVE,
				missing_pages_count,
				migrated_pages,
				&pages_copied,
				&local_tracker);

		RHAC_ASSERT(missing_pages_count >= pages_copied);
		UVM_ASSERT(missing_pages_count >= pages_copied);
		missing_pages_count -= pages_copied;

		if (status != NV_OK)
			goto out;

		if (missing_pages_count == 0)
			goto out;

		if (pages_copied) {
			uvm_page_mask_andnot(copy_page_mask, copy_page_mask, migrated_pages);
		}
	}

	// Now copy from everywhere else to the CPU. This is both for when the
	// destination is the CPU (src_processor_mask empty) and for a staged copy
	// (src_processor_mask containing processors with copy access to dst_id).
	uvm_processor_mask_andnot(&src_processor_mask, &block->resident, &src_processor_mask);
	uvm_processor_mask_clear(&src_processor_mask, dst_id);
	uvm_processor_mask_clear(&src_processor_mask, UVM_ID_CPU);

	uvm_page_mask_zero(staged_pages);

	if (UVM_ID_IS_CPU(dst_id)) {
		transfer_mode_internal = transfer_mode == UVM_VA_BLOCK_TRANSFER_MODE_COPY?
			BLOCK_TRANSFER_MODE_INTERNAL_COPY:
			BLOCK_TRANSFER_MODE_INTERNAL_MOVE;
	}
	else {
		transfer_mode_internal = transfer_mode == UVM_VA_BLOCK_TRANSFER_MODE_COPY?
			BLOCK_TRANSFER_MODE_INTERNAL_COPY_TO_STAGE:
			BLOCK_TRANSFER_MODE_INTERNAL_MOVE_TO_STAGE;
	}

	status = block_copy_resident_pages_mask(block,
			block_context,
			UVM_ID_CPU,
			&src_processor_mask,
			region,
			copy_page_mask,
			prefetch_page_mask,
			transfer_mode_internal,
			missing_pages_count,
			staged_pages,
			&pages_copied_to_cpu,
			&local_tracker);
	if (status != NV_OK)
		goto out;

	// If destination is the CPU then we copied everything there above
	if (UVM_ID_IS_CPU(dst_id)) {
		uvm_page_mask_or(migrated_pages, migrated_pages, staged_pages);

		goto out;
	}

	// Add everything to the block's tracker so that the
	// block_copy_resident_pages_between() call below will acquire it.
	status = rhacuvm_uvm_tracker_add_tracker_safe(&block->tracker, &local_tracker);
	if (status != NV_OK)
		goto out;
	uvm_tracker_clear(&local_tracker);

	// Now copy staged pages from the CPU to the destination.
	status = block_copy_resident_pages_between(block,
			block_context,
			dst_id,
			UVM_ID_CPU,
			region,
			staged_pages,
			prefetch_page_mask,
			transfer_mode == UVM_VA_BLOCK_TRANSFER_MODE_COPY?
			BLOCK_TRANSFER_MODE_INTERNAL_COPY_FROM_STAGE:
			BLOCK_TRANSFER_MODE_INTERNAL_MOVE_FROM_STAGE,
			migrated_pages,
			&pages_copied,
			&local_tracker);

	UVM_ASSERT(missing_pages_count >= pages_copied);
	missing_pages_count -= pages_copied;

	if (status != NV_OK)
		goto out;

	// If we get here, that means we were staging the copy through the CPU and
	// we should copy as many pages from the CPU as we copied to the CPU.
	UVM_ASSERT(pages_copied == pages_copied_to_cpu);

out:
	// Pages that weren't resident anywhere else were populated at the
	// destination directly. Mark them as resident now. We only do it if there
	// have been no errors because we cannot identify which pages failed.
	if (status == NV_OK && missing_pages_count > 0)
		block_copy_set_first_touch_residency(block, block_context, dst_id, region, page_mask);

	// Break read duplication
	if (transfer_mode == UVM_VA_BLOCK_TRANSFER_MODE_MOVE) {
		const uvm_page_mask_t *break_read_duplication_mask;

		if (status == NV_OK) {
			break_read_duplication_mask = page_mask;
		}
		else {
			// We reuse this mask since copy_page_mask is no longer used in the
			// function

			if (page_mask)
				uvm_page_mask_and(&block_context->make_resident.page_mask, resident_mask, page_mask);
			else
				uvm_page_mask_copy(&block_context->make_resident.page_mask, resident_mask);

			break_read_duplication_mask = &block_context->make_resident.page_mask;
		}
		break_read_duplication_in_region(block, block_context, dst_id, region, break_read_duplication_mask);
	}

	// Accumulate the pages that migrated into the output mask
	uvm_page_mask_or(&block_context->make_resident.pages_changed_residency,
			&block_context->make_resident.pages_changed_residency,
			migrated_pages);

	// Add everything from the local tracker to the block's tracker.
	// Notably this is also needed for handling block_copy_resident_pages_between()
	// failures in the first loop.
	tracker_status = rhacuvm_uvm_tracker_add_tracker_safe(&block->tracker, &local_tracker);
	rhacuvm_uvm_tracker_deinit(&local_tracker);

	return status == NV_OK ? tracker_status : status;
}

int rhac_uvm_va_block_make_resident_global(
		struct rhac_comm *comm,
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		uvm_va_block_retry_t *va_block_retry,
		uvm_va_block_context_t *va_block_context,
		uvm_processor_id_t dest_id,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *page_mask,
		const uvm_page_mask_t *prefetch_page_mask,
		uvm_make_resident_cause_t cause,
		uvm_service_block_context_t *service_context
		)
{
	NV_STATUS status;
	uvm_va_range_t *va_range = va_block->va_range;
	uvm_processor_mask_t unmap_processor_mask;
	uvm_page_mask_t *unmap_page_mask = &va_block_context->make_resident.page_mask;
	uvm_page_mask_t *resident_mask;

	va_block_context->make_resident.dest_id = dest_id;
	va_block_context->make_resident.cause = cause;

	if (prefetch_page_mask) {
		UVM_ASSERT(cause == UVM_MAKE_RESIDENT_CAUSE_REPLAYABLE_FAULT ||
				cause == UVM_MAKE_RESIDENT_CAUSE_NON_REPLAYABLE_FAULT ||
				cause == UVM_MAKE_RESIDENT_CAUSE_ACCESS_COUNTER);
	}

	uvm_assert_mutex_locked(&va_block->lock);
	UVM_ASSERT(va_block->va_range);
	UVM_ASSERT(va_block->va_range->type == UVM_VA_RANGE_TYPE_MANAGED);

	resident_mask = block_resident_mask_get_alloc(va_block, dest_id);
	if (!resident_mask)
		return NV_ERR_NO_MEMORY;

	// Unmap all mapped processors except for UVM-Lite GPUs as their mappings
	// are largely persistent.
	uvm_processor_mask_andnot(&unmap_processor_mask, &va_block->mapped, &va_range->uvm_lite_gpus);

	if (page_mask)
		uvm_page_mask_andnot(unmap_page_mask, page_mask, resident_mask);
	else
		uvm_page_mask_complement(unmap_page_mask, resident_mask);

	// Unmap all pages not resident on the destination
	status = rhacuvm_uvm_va_block_unmap_mask(va_block, va_block_context, &unmap_processor_mask, region, unmap_page_mask);
	if (status != NV_OK)
		return status;

	if (page_mask)
		uvm_page_mask_and(unmap_page_mask, page_mask, &va_block->read_duplicated_pages);
	else
		uvm_page_mask_init_from_region(unmap_page_mask, region, &va_block->read_duplicated_pages);

	// Also unmap read-duplicated pages excluding dest_id
	uvm_processor_mask_clear(&unmap_processor_mask, dest_id);
	status = rhacuvm_uvm_va_block_unmap_mask(va_block, va_block_context, &unmap_processor_mask, region, unmap_page_mask);
	if (status != NV_OK)
		return status;

	rhacuvm_uvm_tools_record_read_duplicate_invalidate(va_block,
			dest_id,
			region,
			unmap_page_mask);

	// Note that block_populate_pages and block_move_resident_pages also use
	// va_block_context->make_resident.page_mask.
	unmap_page_mask = NULL;

	status = block_populate_pages(va_block, va_block_retry, va_block_context, dest_id, region, page_mask);
	if (status != NV_OK)
		return status;

	status = rhac_uvm_post_block_fault(
			comm,
			processor_id,
			va_block,
			va_block_context,
			service_context,
			dest_id,
			page_mask);
	if (status != NV_OK) {
		RHAC_ASSERT(false);
		return status;
	}

	return NV_OK;
}

int rhac_uvm_va_block_make_resident_read_duplicate_global(
		struct rhac_comm *comm,
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		uvm_va_block_retry_t *va_block_retry,
		uvm_va_block_context_t *va_block_context,
		uvm_processor_id_t dest_id,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *page_mask,
		const uvm_page_mask_t *prefetch_page_mask,
		uvm_make_resident_cause_t cause,
		uvm_service_block_context_t *service_context
		)
{
	NV_STATUS status = NV_OK;
	uvm_processor_id_t src_id;

	va_block_context->make_resident.dest_id = dest_id;
	va_block_context->make_resident.cause = cause;

	if (prefetch_page_mask) {
		// TODO: Bug 1877578: investigate automatic read-duplicate policies
		UVM_ASSERT(cause == UVM_MAKE_RESIDENT_CAUSE_REPLAYABLE_FAULT ||
				cause == UVM_MAKE_RESIDENT_CAUSE_NON_REPLAYABLE_FAULT ||
				cause == UVM_MAKE_RESIDENT_CAUSE_ACCESS_COUNTER);
	}

	uvm_assert_mutex_locked(&va_block->lock);
	UVM_ASSERT(va_block->va_range);
	UVM_ASSERT(va_block->va_range->type == UVM_VA_RANGE_TYPE_MANAGED);

	// For pages that are entering read-duplication we need to unmap remote
	// mappings and revoke RW and higher access permissions.
	//
	// The current implementation:
	// - Unmaps pages from all processors but the one with the resident copy
	// - Revokes write access from the processor with the resident copy
	for_each_id_in_mask(src_id, &va_block->resident) {
		// Note that the below calls to block_populate_pages and
		// block_move_resident_pages also use
		// va_block_context->make_resident.page_mask.
		uvm_page_mask_t *preprocess_page_mask = &va_block_context->make_resident.page_mask;
		const uvm_page_mask_t *resident_mask = rhacuvm_uvm_va_block_resident_mask_get(va_block, src_id);
		UVM_ASSERT(!uvm_page_mask_empty(resident_mask));

		if (page_mask)
			uvm_page_mask_andnot(preprocess_page_mask, page_mask, &va_block->read_duplicated_pages);
		else
			uvm_page_mask_complement(preprocess_page_mask, &va_block->read_duplicated_pages);

		// If there are no pages that need to be unmapped/revoked, skip to the
		// next processor
		if (!uvm_page_mask_and(preprocess_page_mask, preprocess_page_mask, resident_mask))
			continue;

		status = block_prep_read_duplicate_mapping(va_block, va_block_context, src_id, region, preprocess_page_mask);
		if (status != NV_OK)
			return status;
	}

	status = block_populate_pages(va_block, va_block_retry, va_block_context, dest_id, region, page_mask);
	if (status != NV_OK)
		return status;

	status = rhac_uvm_post_block_fault(
			comm, 
			processor_id,
			va_block,
			va_block_context,
			service_context,
			dest_id,
			page_mask);
	if (status != NV_OK)
		return status;

	return NV_OK;
}

int rhac_uvm_migrate_pages_global(
		struct rhac_comm *comm,
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		const uvm_va_range_t *va_range,
		const uvm_va_space_t *va_space,
		uvm_va_block_retry_t *block_retry,
		uvm_service_block_context_t *service_context,
		const uvm_perf_prefetch_hint_t *prefetch_hint
		)
{
	uvm_processor_id_t new_residency;
	NV_STATUS status = NV_OK;

	// 1- Migrate pages and compute mapping protections
	for_each_id_in_mask(new_residency, &service_context->resident_processors) {

		uvm_processor_mask_t *all_involved_processors = &service_context->block_context.make_resident.all_involved_processors;
		uvm_page_mask_t *new_residency_mask = &service_context->per_processor_masks[uvm_id_value(new_residency)].new_residency;
		uvm_page_mask_t *did_migrate_mask = &service_context->block_context.make_resident.pages_changed_residency;
		uvm_make_resident_cause_t cause;

		UVM_ASSERT_MSG(service_context->operation == UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS ||
				service_context->operation == UVM_SERVICE_OPERATION_NON_REPLAYABLE_FAULTS ||
				service_context->operation == UVM_SERVICE_OPERATION_ACCESS_COUNTERS,
				"Invalid operation value %u\n", service_context->operation);

		if (service_context->operation == UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS)
			cause = UVM_MAKE_RESIDENT_CAUSE_REPLAYABLE_FAULT;
		else if (service_context->operation == UVM_SERVICE_OPERATION_NON_REPLAYABLE_FAULTS)
			cause = UVM_MAKE_RESIDENT_CAUSE_NON_REPLAYABLE_FAULT;
		else
			cause = UVM_MAKE_RESIDENT_CAUSE_ACCESS_COUNTER;

		// 1.1- Migrate pages

		// Reset masks before all of the make_resident calls
		uvm_page_mask_zero(did_migrate_mask);
		uvm_processor_mask_zero(all_involved_processors);

		if (UVM_ID_IS_VALID(prefetch_hint->residency)) {
			UVM_ASSERT(uvm_id_equal(prefetch_hint->residency, new_residency));
			UVM_ASSERT(prefetch_hint->prefetch_pages_mask != NULL);

			uvm_page_mask_or(new_residency_mask, new_residency_mask, prefetch_hint->prefetch_pages_mask);
		}

		if (service_context->read_duplicate_count == 0 ||
				uvm_page_mask_andnot(&service_context->block_context.caller_page_mask,
					new_residency_mask,
					&service_context->read_duplicate_mask)) {
			status = rhac_uvm_va_block_make_resident_global(
					comm,
					processor_id,
					va_block,
					block_retry,
					&service_context->block_context,
					new_residency,
					service_context->region,
					service_context->read_duplicate_count == 0?
					new_residency_mask:
					&service_context->block_context.caller_page_mask,
					prefetch_hint->prefetch_pages_mask,
					cause,
					service_context
					);
			if (status != NV_OK)
				return status;
				
		}

		if (service_context->read_duplicate_count != 0 &&
				uvm_page_mask_and(&service_context->block_context.caller_page_mask,
					new_residency_mask,
					&service_context->read_duplicate_mask)) {
			status = rhac_uvm_va_block_make_resident_read_duplicate_global(
					comm,
					processor_id,
					va_block,
					block_retry,
					&service_context->block_context,
					new_residency,
					service_context->region,
					&service_context->block_context.caller_page_mask,
					prefetch_hint->prefetch_pages_mask,
					cause,
					service_context
					);
			if (status != NV_OK)
				return status;

		}
	}

	return NV_OK;
}

int rhac_uvm_migrate_pages_local(
		struct rhac_comm *comm,
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		const uvm_va_range_t *va_range,
		const uvm_va_space_t *va_space,
		uvm_va_block_retry_t *block_retry,
		uvm_service_block_context_t *service_context,
		const uvm_perf_prefetch_hint_t *prefetch_hint,
		uvm_processor_mask_t *processors_involved_in_cpu_migration
		)
{
	uvm_processor_id_t new_residency;
	NV_STATUS status = NV_OK;
	uvm_prot_t new_prot; 

	uvm_va_block_context_t *va_block_context = &service_context->block_context;

	// 1- Migrate pages and compute mapping protections
	for_each_id_in_mask(new_residency, &service_context->resident_processors) {

		uvm_processor_mask_t *all_involved_processors = &service_context->block_context.make_resident.all_involved_processors;
		uvm_page_mask_t *new_residency_mask = &service_context->per_processor_masks[uvm_id_value(new_residency)].new_residency;
		uvm_page_mask_t *did_migrate_mask = &service_context->block_context.make_resident.pages_changed_residency;
		uvm_page_index_t page_index;

		UVM_ASSERT_MSG(service_context->operation == UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS ||
				service_context->operation == UVM_SERVICE_OPERATION_NON_REPLAYABLE_FAULTS ||
				service_context->operation == UVM_SERVICE_OPERATION_ACCESS_COUNTERS,
				"Invalid operation value %u\n", service_context->operation);

		if (service_context->read_duplicate_count == 0 ||
				uvm_page_mask_andnot(&service_context->block_context.caller_page_mask,
					new_residency_mask,
					&service_context->read_duplicate_mask)) {
			status = block_copy_resident_pages_local(va_block,
					va_block_context,
					new_residency,
					service_context->region,
					service_context->read_duplicate_count == 0?
					new_residency_mask:
					&service_context->block_context.caller_page_mask,
					prefetch_hint->prefetch_pages_mask,
					UVM_VA_BLOCK_TRANSFER_MODE_MOVE);
			if (status != NV_OK)
				return status;

			// Update eviction heuristics, if needed. Notably this could repeat the call
			// done in block_set_resident_processor(), but that doesn't do anything bad
			// and it's simpler to keep it in both places.
			//
			// Skip this if we didn't do anything (the input region and/or page mask was
			// empty).
			if (uvm_processor_mask_test(&va_block->resident, new_residency))
				block_mark_memory_used(va_block, new_residency);

		}

		if (service_context->read_duplicate_count != 0 &&
				uvm_page_mask_and(&service_context->block_context.caller_page_mask,
					new_residency_mask,
					&service_context->read_duplicate_mask)) {
			status = block_copy_resident_pages_local(va_block,
					va_block_context,
					new_residency,
					service_context->region,
					&service_context->block_context.caller_page_mask,
					prefetch_hint->prefetch_pages_mask,
					UVM_VA_BLOCK_TRANSFER_MODE_COPY);
			if (status != NV_OK)
				return status;

			// Update eviction heuristics, if needed. Notably this could repeat the call
			// done in block_set_resident_processor(), but that doesn't do anything bad
			// and it's simpler to keep it in both places.
			//
			// Skip this if we didn't do anything (the input region and/or page mask was
			// empty).
			if (uvm_processor_mask_test(&va_block->resident, new_residency))
				block_mark_memory_used(va_block, new_residency);

		}

		if (UVM_ID_IS_CPU(new_residency)) {
			// Save all the processors involved in migrations to the CPU for
			// an ECC check before establishing the CPU mappings.
			uvm_processor_mask_copy(processors_involved_in_cpu_migration, all_involved_processors);
		}

		if (UVM_ID_IS_CPU(processor_id) && !uvm_processor_mask_empty(all_involved_processors))
			service_context->cpu_fault.did_migrate = true;

		uvm_page_mask_andnot(&service_context->did_not_migrate_mask, new_residency_mask, did_migrate_mask);

		// 1.2 - Compute mapping protections for the requesting processor on
		// the new residency
		for_each_va_block_page_in_region_mask(page_index, new_residency_mask, service_context->region) {
			new_prot = rhac_compute_new_permission(va_block,
					page_index,
					processor_id,
					new_residency,
					service_context->access_type[page_index]);

			if (service_context->mappings_by_prot[new_prot-1].count++ == 0)
				uvm_page_mask_zero(&service_context->mappings_by_prot[new_prot-1].page_mask);

			uvm_page_mask_set(&service_context->mappings_by_prot[new_prot-1].page_mask, page_index);
		}

		// 1.3- Revoke permissions
		//
		// NOTE: uvm_va_block_make_resident destroys mappings to old locations.
		//       Thus, we need to revoke only if residency did not change and we
		//       are mapping higher than READ ONLY.
		for (new_prot = UVM_PROT_READ_WRITE; new_prot <= UVM_PROT_READ_WRITE_ATOMIC; ++new_prot) {
			bool pages_need_revocation;
			uvm_processor_mask_t revoke_processors;
			uvm_prot_t revoke_prot;
			bool this_processor_has_enabled_atomics;

			if (service_context->mappings_by_prot[new_prot-1].count == 0)
				continue;

			pages_need_revocation = uvm_page_mask_and(&service_context->revocation_mask,
					&service_context->did_not_migrate_mask,
					&service_context->mappings_by_prot[new_prot-1].page_mask);
			if (!pages_need_revocation)
				continue;

			uvm_processor_mask_and(&revoke_processors, &va_block->mapped, &va_space->faultable_processors);

			// Do not revoke the processor that took the fault
			uvm_processor_mask_clear(&revoke_processors, processor_id);

			this_processor_has_enabled_atomics = uvm_processor_mask_test(&va_space->system_wide_atomics_enabled_processors,
					processor_id);

			// Atomic operations on processors with system-wide atomics
			// disabled or with native atomics access to new_residency
			// behave like writes.
			if (new_prot == UVM_PROT_READ_WRITE ||
					!this_processor_has_enabled_atomics ||
					uvm_processor_mask_test(&va_space->has_native_atomics[uvm_id_value(new_residency)], processor_id)) {

				// Exclude processors with native atomics on the resident copy
				uvm_processor_mask_andnot(&revoke_processors,
						&revoke_processors,
						&va_space->has_native_atomics[uvm_id_value(new_residency)]);

				// Exclude processors with disabled system-wide atomics
				uvm_processor_mask_and(&revoke_processors,
						&revoke_processors,
						&va_space->system_wide_atomics_enabled_processors);
			}

			if (UVM_ID_IS_CPU(processor_id)) {
				revoke_prot = UVM_PROT_READ_WRITE_ATOMIC;
			}
			else {
				revoke_prot = (new_prot == UVM_PROT_READ_WRITE_ATOMIC)? UVM_PROT_READ_WRITE:
					UVM_PROT_READ_WRITE_ATOMIC;
			}

			// UVM-Lite processors must always have RWA mappings
			if (uvm_processor_mask_andnot(&revoke_processors, &revoke_processors, &va_range->uvm_lite_gpus)) {
				// Access counters should never trigger revocations apart from
				// read-duplication, which are performed in the calls to
				// uvm_va_block_make_resident_read_duplicate, above.
				if (service_context->operation == UVM_SERVICE_OPERATION_ACCESS_COUNTERS) {
					UVM_ASSERT(check_access_counters_dont_revoke(va_block,
								&service_context->block_context,
								service_context->region,
								&revoke_processors,
								&service_context->revocation_mask,
								revoke_prot));
				}

				// Downgrade other processors' mappings
				status = rhacuvm_uvm_va_block_revoke_prot_mask(va_block,
						&service_context->block_context,
						&revoke_processors,
						service_context->region,
						&service_context->revocation_mask,
						revoke_prot);
				if (status != NV_OK)
					return status;
			}
		}
	}

	return NV_OK;
}

static NV_STATUS rhac_check_ecc(
		uvm_processor_id_t processor_id,
		uvm_va_space_t *va_space,
		uvm_va_block_t *va_block,
		uvm_service_block_context_t *service_context,
		const uvm_processor_mask_t *processors_involved_in_cpu_migration,
		struct rhac_comm *ch
		)
{
	NV_STATUS status = NV_OK;

	// 2- Check for ECC errors on all GPUs involved in the migration if CPU is
	//    the destination. Migrations in response to CPU faults are special
	//    because they're on the only path (apart from tools) where CUDA is not
	//    involved and wouldn't have a chance to do its own ECC checking.
	if (service_context->operation == UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS &&
			UVM_ID_IS_CPU(processor_id) &&
			!uvm_processor_mask_empty(processors_involved_in_cpu_migration)) {
		uvm_gpu_id_t gpu_id;

		// Before checking for ECC errors, make sure all of the GPU work
		// is finished. Creating mappings on the CPU would have to wait
		// for the tracker anyway so this shouldn't hurt performance.
		status = rhacuvm_uvm_tracker_wait(&va_block->tracker);
		if (status != NV_OK)
			return status;

		for_each_gpu_id_in_mask(gpu_id, processors_involved_in_cpu_migration) {
			// We cannot call into RM here so use the no RM ECC check.
			status = rhacuvm_uvm_gpu_check_ecc_error_no_rm(uvm_va_space_get_gpu(va_space, gpu_id));
			if (status == NV_WARN_MORE_PROCESSING_REQUIRED) {
				// In case we need to call into RM to be sure whether
				// there is an ECC error or not, signal that to the
				// caller by adding the GPU to the mask.
				//
				// In that case the ECC error might be noticed only after
				// the CPU mappings have been already created below,
				// exposing different CPU threads to the possibly corrupt
				// data, but this thread will fault eventually and that's
				// considered to be an acceptable trade-off between
				// performance and ECC error containment.
				uvm_processor_mask_set(&service_context->cpu_fault.gpus_to_check_for_ecc, gpu_id);
				status = NV_OK;
			}
			if (status != NV_OK)
				return status;
		}
	}

	return NV_OK;
}

static NV_STATUS rhac_remap_pages(
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		const uvm_va_space_t *va_space,
		uvm_service_block_context_t *service_context,
		struct rhac_comm *ch
		)
{
	NV_STATUS status = NV_OK;
	uvm_prot_t new_prot;

	// 3- Map requesting processor with the necessary privileges
	for (new_prot = UVM_PROT_READ_ONLY; new_prot <= UVM_PROT_READ_WRITE_ATOMIC; ++new_prot) {
		const uvm_page_mask_t *map_prot_mask = &service_context->mappings_by_prot[new_prot-1].page_mask;

		if (service_context->mappings_by_prot[new_prot-1].count == 0)
			continue;

		// 3.1 - Unmap CPU pages
		if (service_context->operation != UVM_SERVICE_OPERATION_ACCESS_COUNTERS && UVM_ID_IS_CPU(processor_id)) {
			// The kernel can downgrade our CPU mappings at any time without
			// notifying us, which means our PTE state could be stale. We
			// handle this by unmapping the CPU PTE and re-mapping it again.
			//
			// A CPU fault is unexpected if:
			// curr_prot == RW || (!is_write && curr_prot == RO)
			status = rhacuvm_uvm_va_block_unmap(va_block,
					&service_context->block_context,
					UVM_ID_CPU,
					service_context->region,
					map_prot_mask,
					NULL);
			if (status != NV_OK)
				return status;
		}

		// 3.2 - Add new mappings

		// The faulting processor can be mapped remotely due to user hints or
		// the thrashing mitigation heuristics. Therefore, we set the cause
		// accordingly in each case.

		// Map pages that are thrashing first
		if (service_context->thrashing_pin_count > 0 && va_space->tools.enabled) {
			RHAC_ASSERT(false);
			uvm_page_mask_t *helper_page_mask = &service_context->block_context.caller_page_mask;
			bool pages_need_mapping = uvm_page_mask_and(helper_page_mask,
					map_prot_mask,
					&service_context->thrashing_pin_mask);
			if (pages_need_mapping) {
				status = rhac_uvm_va_block_map(va_block,
						&service_context->block_context,
						processor_id,
						service_context->region,
						helper_page_mask,
						new_prot,
						UvmEventMapRemoteCauseThrashing,
						&va_block->tracker);
				if (status != NV_OK)
					return status;

				// Remove thrashing pages from the map mask
				pages_need_mapping = uvm_page_mask_andnot(helper_page_mask,
						map_prot_mask,
						&service_context->thrashing_pin_mask);
				if (!pages_need_mapping)
					continue;

				map_prot_mask = helper_page_mask;
			}
		}

		status = rhac_uvm_va_block_map(va_block,
				&service_context->block_context,
				processor_id,
				service_context->region,
				map_prot_mask,
				new_prot,
				UvmEventMapRemoteCausePolicy,
				&va_block->tracker);
		if (status != NV_OK)
			return status;
	}

	return NV_OK;
}

static NV_STATUS rhac_set_accessed_by(
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		uvm_service_block_context_t *service_context,
		struct rhac_comm *ch
		)
{
	uvm_processor_id_t new_residency;
	NV_STATUS status = NV_OK;
	uvm_prot_t new_prot;

	// 4- If pages did migrate, map SetAccessedBy processors, except for UVM-Lite
	for_each_id_in_mask(new_residency, &service_context->resident_processors) {
		const uvm_page_mask_t *new_residency_mask;
		new_residency_mask = &service_context->per_processor_masks[uvm_id_value(new_residency)].new_residency;

		for (new_prot = UVM_PROT_READ_ONLY; new_prot <= UVM_PROT_READ_WRITE_ATOMIC; ++new_prot) {
			uvm_page_mask_t *map_prot_mask = &service_context->block_context.caller_page_mask;
			bool pages_need_mapping;

			if (service_context->mappings_by_prot[new_prot-1].count == 0)
				continue;

			pages_need_mapping = uvm_page_mask_and(map_prot_mask,
					new_residency_mask,
					&service_context->mappings_by_prot[new_prot-1].page_mask);
			if (!pages_need_mapping)
				continue;

			// Map pages that are thrashing
			if (service_context->thrashing_pin_count > 0) {
				RHAC_ASSERT(false);
				uvm_page_index_t page_index;

				for_each_va_block_page_in_region_mask(page_index,
						&service_context->thrashing_pin_mask,
						service_context->region) {
					uvm_processor_mask_t *map_thrashing_processors = NULL;
					NvU64 page_addr = uvm_va_block_cpu_page_address(va_block, page_index);

					// Check protection type
					if (!uvm_page_mask_test(map_prot_mask, page_index))
						continue;

					map_thrashing_processors = rhacuvm_uvm_perf_thrashing_get_thrashing_processors(va_block, page_addr);

					status = rhacuvm_uvm_va_block_add_mappings_after_migration(va_block,
							&service_context->block_context,
							new_residency,
							processor_id,
							uvm_va_block_region_for_page(page_index),
							map_prot_mask,
							new_prot,
							map_thrashing_processors);
					if (status != NV_OK)
						return status;
				}

				pages_need_mapping = uvm_page_mask_andnot(map_prot_mask,
						map_prot_mask,
						&service_context->thrashing_pin_mask);
				if (!pages_need_mapping)
					continue;
			}

			// Map the the rest of pages in a single shot
			
			status = rhacuvm_uvm_va_block_add_mappings_after_migration(va_block,
					&service_context->block_context,
					new_residency,
					processor_id,
					service_context->region,
					map_prot_mask,
					new_prot,
					NULL);
			if (status != NV_OK)
				return status;
		}
	}

	return NV_OK;
}

int rhac_uvm_va_block_service_locked_local(
		struct rhac_comm *comm,
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		uvm_service_block_context_t *service_context
		)
{
	NV_STATUS status = NV_OK;
	uvm_va_range_t *va_range = va_block->va_range;
	uvm_va_space_t *va_space = va_range->va_space;
	uvm_perf_prefetch_hint_t prefetch_hint = UVM_PERF_PREFETCH_HINT_NONE();
	uvm_processor_mask_t processors_involved_in_cpu_migration;

	uvm_assert_mutex_locked(&va_block->lock);
	UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_MANAGED);

	uvm_va_block_retry_t block_retry;

	uvm_processor_mask_zero(&processors_involved_in_cpu_migration);

	status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &block_retry,
			rhac_uvm_migrate_pages_local(comm, processor_id, va_block, va_range, va_space, &block_retry, service_context,
				&prefetch_hint, &processors_involved_in_cpu_migration));
	if (status != NV_OK) {
		RHAC_ASSERT(false);
		return status;
	}

	status = rhac_check_ecc(processor_id, va_space, va_block, service_context, &processors_involved_in_cpu_migration, comm);
	if (status != NV_OK) {
		RHAC_ASSERT(false);
		return status;
	}

	status = rhac_remap_pages(processor_id, va_block, va_space, service_context, comm);
	if (status != NV_OK) {
		RHAC_ASSERT(false);
		return status;
	}

	status = rhac_set_accessed_by(processor_id, va_block, service_context, comm);
	if (status != NV_OK) {
		RHAC_ASSERT(false);
		return status;
	}

	return NV_OK;
}

NV_STATUS uvm_va_block_service_locked(uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		uvm_va_block_retry_t *block_retry,
		uvm_service_block_context_t *service_context)
{
	NV_STATUS status = NV_OK;
	uvm_processor_id_t new_residency;
	uvm_prot_t new_prot;
	uvm_va_range_t *va_range = va_block->va_range;
	uvm_va_space_t *va_space = va_range->va_space;
	uvm_perf_prefetch_hint_t prefetch_hint = UVM_PERF_PREFETCH_HINT_NONE();
	uvm_processor_mask_t processors_involved_in_cpu_migration;

	uvm_assert_mutex_locked(&va_block->lock);
	UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_MANAGED);

	// GPU fault servicing must be done under the VA space read lock. GPU fault
	// servicing is required for RM to make forward progress, and we allow other
	// threads to call into RM while holding the VA space lock in read mode. If
	// we took the VA space lock in write mode on the GPU fault service path,
	// we could deadlock because the thread in RM which holds the VA space lock
	// for read wouldn't be able to complete until fault servicing completes.
	if (service_context->operation != UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS || UVM_ID_IS_CPU(processor_id))
		uvm_assert_rwsem_locked(&va_space->lock);
	else
		uvm_assert_rwsem_locked_read(&va_space->lock);

	// Performance heuristics policy: we only consider prefetching when there
	// are migrations to a single processor, only.
	if (uvm_processor_mask_get_count(&service_context->resident_processors) == 1) {
		uvm_page_index_t page_index;
		uvm_page_mask_t *new_residency_mask;

		new_residency = uvm_processor_mask_find_first_id(&service_context->resident_processors);
		new_residency_mask = &service_context->per_processor_masks[uvm_id_value(new_residency)].new_residency;

		// Update prefetch tracking structure with the pages that will migrate
		// due to faults
		rhacuvm_uvm_perf_prefetch_prenotify_fault_migrations(va_block,
				&service_context->block_context,
				new_residency,
				new_residency_mask,
				service_context->region);

		prefetch_hint = rhacuvm_uvm_perf_prefetch_get_hint(va_block, new_residency_mask);

		// Obtain the prefetch hint and give a fake fault access type to the
		// prefetched pages
		if (UVM_ID_IS_VALID(prefetch_hint.residency)) {
			UVM_ASSERT(prefetch_hint.prefetch_pages_mask != NULL);

			for_each_va_block_page_in_mask(page_index, prefetch_hint.prefetch_pages_mask, va_block) {
				UVM_ASSERT(!uvm_page_mask_test(new_residency_mask, page_index));

				service_context->access_type[page_index] = UVM_FAULT_ACCESS_TYPE_PREFETCH;

				if (uvm_va_range_is_read_duplicate(va_range) ||
						(va_range->read_duplication != UVM_READ_DUPLICATION_DISABLED &&
						 uvm_page_mask_test(&va_block->read_duplicated_pages, page_index))) {
					if (service_context->read_duplicate_count++ == 0)
						uvm_page_mask_zero(&service_context->read_duplicate_mask);

					uvm_page_mask_set(&service_context->read_duplicate_mask, page_index);
				}
			}

			service_context->region = uvm_va_block_region_from_block(va_block);
		}
	}

	for (new_prot = UVM_PROT_READ_ONLY; new_prot < UVM_PROT_MAX; ++new_prot)
		service_context->mappings_by_prot[new_prot-1].count = 0;

	uvm_processor_mask_zero(&processors_involved_in_cpu_migration);

	// 1- Migrate pages and compute mapping protections
	for_each_id_in_mask(new_residency, &service_context->resident_processors) {
		uvm_processor_mask_t *all_involved_processors = &service_context->block_context.make_resident.all_involved_processors;
		uvm_page_mask_t *new_residency_mask = &service_context->per_processor_masks[uvm_id_value(new_residency)].new_residency;
		uvm_page_mask_t *did_migrate_mask = &service_context->block_context.make_resident.pages_changed_residency;
		uvm_page_index_t page_index;
		uvm_make_resident_cause_t cause;

		UVM_ASSERT_MSG(service_context->operation == UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS ||
				service_context->operation == UVM_SERVICE_OPERATION_NON_REPLAYABLE_FAULTS ||
				service_context->operation == UVM_SERVICE_OPERATION_ACCESS_COUNTERS,
				"Invalid operation value %u\n", service_context->operation);

		if (service_context->operation == UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS)
			cause = UVM_MAKE_RESIDENT_CAUSE_REPLAYABLE_FAULT;
		else if (service_context->operation == UVM_SERVICE_OPERATION_NON_REPLAYABLE_FAULTS)
			cause = UVM_MAKE_RESIDENT_CAUSE_NON_REPLAYABLE_FAULT;
		else
			cause = UVM_MAKE_RESIDENT_CAUSE_ACCESS_COUNTER;

		// 1.1- Migrate pages

		// Reset masks before all of the make_resident calls
		uvm_page_mask_zero(did_migrate_mask);
		uvm_processor_mask_zero(all_involved_processors);

		if (UVM_ID_IS_VALID(prefetch_hint.residency)) {
			UVM_ASSERT(uvm_id_equal(prefetch_hint.residency, new_residency));
			UVM_ASSERT(prefetch_hint.prefetch_pages_mask != NULL);

			uvm_page_mask_or(new_residency_mask, new_residency_mask, prefetch_hint.prefetch_pages_mask);
		}

		if (service_context->read_duplicate_count == 0 ||
				uvm_page_mask_andnot(&service_context->block_context.caller_page_mask,
					new_residency_mask,
					&service_context->read_duplicate_mask)) {
			status = rhacuvm_uvm_va_block_make_resident(va_block,
					block_retry,
					&service_context->block_context,
					new_residency,
					service_context->region,
					service_context->read_duplicate_count == 0?
					new_residency_mask:
					&service_context->block_context.caller_page_mask,
					prefetch_hint.prefetch_pages_mask,
					cause);
			if (status != NV_OK)
				return status;
		}

		if (service_context->read_duplicate_count != 0 &&
				uvm_page_mask_and(&service_context->block_context.caller_page_mask,
					new_residency_mask,
					&service_context->read_duplicate_mask)) {
			status = rhacuvm_uvm_va_block_make_resident_read_duplicate(va_block,
					block_retry,
					&service_context->block_context,
					new_residency,
					service_context->region,
					&service_context->block_context.caller_page_mask,
					prefetch_hint.prefetch_pages_mask,
					cause);
			if (status != NV_OK)
				return status;
		}

		if (UVM_ID_IS_CPU(new_residency)) {
			// Save all the processors involved in migrations to the CPU for
			// an ECC check before establishing the CPU mappings.
			uvm_processor_mask_copy(&processors_involved_in_cpu_migration, all_involved_processors);
		}

		if (UVM_ID_IS_CPU(processor_id) && !uvm_processor_mask_empty(all_involved_processors))
			service_context->cpu_fault.did_migrate = true;

		uvm_page_mask_andnot(&service_context->did_not_migrate_mask, new_residency_mask, did_migrate_mask);

		// 1.2 - Compute mapping protections for the requesting processor on
		// the new residency
		for_each_va_block_page_in_region_mask(page_index, new_residency_mask, service_context->region) {
			new_prot = rhac_compute_new_permission(va_block,
					page_index,
					processor_id,
					new_residency,
					service_context->access_type[page_index]);

			if (service_context->mappings_by_prot[new_prot-1].count++ == 0)
				uvm_page_mask_zero(&service_context->mappings_by_prot[new_prot-1].page_mask);

			uvm_page_mask_set(&service_context->mappings_by_prot[new_prot-1].page_mask, page_index);
		}

		// 1.3- Revoke permissions
		//
		// NOTE: uvm_va_block_make_resident destroys mappings to old locations.
		//       Thus, we need to revoke only if residency did not change and we
		//       are mapping higher than READ ONLY.
		for (new_prot = UVM_PROT_READ_WRITE; new_prot <= UVM_PROT_READ_WRITE_ATOMIC; ++new_prot) {
			bool pages_need_revocation;
			uvm_processor_mask_t revoke_processors;
			uvm_prot_t revoke_prot;
			bool this_processor_has_enabled_atomics;

			if (service_context->mappings_by_prot[new_prot-1].count == 0)
				continue;

			pages_need_revocation = uvm_page_mask_and(&service_context->revocation_mask,
					&service_context->did_not_migrate_mask,
					&service_context->mappings_by_prot[new_prot-1].page_mask);
			if (!pages_need_revocation)
				continue;

			uvm_processor_mask_and(&revoke_processors, &va_block->mapped, &va_space->faultable_processors);

			// Do not revoke the processor that took the fault
			uvm_processor_mask_clear(&revoke_processors, processor_id);

			this_processor_has_enabled_atomics = uvm_processor_mask_test(&va_space->system_wide_atomics_enabled_processors,
					processor_id);

			// Atomic operations on processors with system-wide atomics
			// disabled or with native atomics access to new_residency
			// behave like writes.
			if (new_prot == UVM_PROT_READ_WRITE ||
					!this_processor_has_enabled_atomics ||
					uvm_processor_mask_test(&va_space->has_native_atomics[uvm_id_value(new_residency)], processor_id)) {

				// Exclude processors with native atomics on the resident copy
				uvm_processor_mask_andnot(&revoke_processors,
						&revoke_processors,
						&va_space->has_native_atomics[uvm_id_value(new_residency)]);

				// Exclude processors with disabled system-wide atomics
				uvm_processor_mask_and(&revoke_processors,
						&revoke_processors,
						&va_space->system_wide_atomics_enabled_processors);
			}

			if (UVM_ID_IS_CPU(processor_id)) {
				revoke_prot = UVM_PROT_READ_WRITE_ATOMIC;
			}
			else {
				revoke_prot = (new_prot == UVM_PROT_READ_WRITE_ATOMIC)? UVM_PROT_READ_WRITE:
					UVM_PROT_READ_WRITE_ATOMIC;
			}

			// UVM-Lite processors must always have RWA mappings
			if (uvm_processor_mask_andnot(&revoke_processors, &revoke_processors, &va_range->uvm_lite_gpus)) {
				// Access counters should never trigger revocations apart from
				// read-duplication, which are performed in the calls to
				// uvm_va_block_make_resident_read_duplicate, above.
				if (service_context->operation == UVM_SERVICE_OPERATION_ACCESS_COUNTERS) {
					UVM_ASSERT(check_access_counters_dont_revoke(va_block,
								&service_context->block_context,
								service_context->region,
								&revoke_processors,
								&service_context->revocation_mask,
								revoke_prot));
				}

				// Downgrade other processors' mappings
				status = rhacuvm_uvm_va_block_revoke_prot_mask(va_block,
						&service_context->block_context,
						&revoke_processors,
						service_context->region,
						&service_context->revocation_mask,
						revoke_prot);
				if (status != NV_OK)
					return status;
			}
		}
	}

	// 2- Check for ECC errors on all GPUs involved in the migration if CPU is
	//    the destination. Migrations in response to CPU faults are special
	//    because they're on the only path (apart from tools) where CUDA is not
	//    involved and wouldn't have a chance to do its own ECC checking.
	if (service_context->operation == UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS &&
			UVM_ID_IS_CPU(processor_id) &&
			!uvm_processor_mask_empty(&processors_involved_in_cpu_migration)) {
		uvm_gpu_id_t gpu_id;

		// Before checking for ECC errors, make sure all of the GPU work
		// is finished. Creating mappings on the CPU would have to wait
		// for the tracker anyway so this shouldn't hurt performance.
		status = rhacuvm_uvm_tracker_wait(&va_block->tracker);
		if (status != NV_OK)
			return status;

		for_each_gpu_id_in_mask(gpu_id, &processors_involved_in_cpu_migration) {
			// We cannot call into RM here so use the no RM ECC check.
			status = rhacuvm_uvm_gpu_check_ecc_error_no_rm(uvm_va_space_get_gpu(va_space, gpu_id));
			if (status == NV_WARN_MORE_PROCESSING_REQUIRED) {
				// In case we need to call into RM to be sure whether
				// there is an ECC error or not, signal that to the
				// caller by adding the GPU to the mask.
				//
				// In that case the ECC error might be noticed only after
				// the CPU mappings have been already created below,
				// exposing different CPU threads to the possibly corrupt
				// data, but this thread will fault eventually and that's
				// considered to be an acceptable trade-off between
				// performance and ECC error containment.
				uvm_processor_mask_set(&service_context->cpu_fault.gpus_to_check_for_ecc, gpu_id);
				status = NV_OK;
			}
			if (status != NV_OK)
				return status;
		}
	}

	// 3- Map requesting processor with the necessary privileges
	for (new_prot = UVM_PROT_READ_ONLY; new_prot <= UVM_PROT_READ_WRITE_ATOMIC; ++new_prot) {
		const uvm_page_mask_t *map_prot_mask = &service_context->mappings_by_prot[new_prot-1].page_mask;

		if (service_context->mappings_by_prot[new_prot-1].count == 0)
			continue;

		// 3.1 - Unmap CPU pages
		if (service_context->operation != UVM_SERVICE_OPERATION_ACCESS_COUNTERS && UVM_ID_IS_CPU(processor_id)) {
			// The kernel can downgrade our CPU mappings at any time without
			// notifying us, which means our PTE state could be stale. We
			// handle this by unmapping the CPU PTE and re-mapping it again.
			//
			// A CPU fault is unexpected if:
			// curr_prot == RW || (!is_write && curr_prot == RO)
			status = rhacuvm_uvm_va_block_unmap(va_block,
					&service_context->block_context,
					UVM_ID_CPU,
					service_context->region,
					map_prot_mask,
					NULL);
			if (status != NV_OK)
				return status;
		}

		// 3.2 - Add new mappings

		// The faulting processor can be mapped remotely due to user hints or
		// the thrashing mitigation heuristics. Therefore, we set the cause
		// accordingly in each case.

		// Map pages that are thrashing first
		if (service_context->thrashing_pin_count > 0 && va_space->tools.enabled) {
			RHAC_ASSERT(false);
			uvm_page_mask_t *helper_page_mask = &service_context->block_context.caller_page_mask;
			bool pages_need_mapping = uvm_page_mask_and(helper_page_mask,
					map_prot_mask,
					&service_context->thrashing_pin_mask);
			if (pages_need_mapping) {
				status = rhac_uvm_va_block_map(va_block,
						&service_context->block_context,
						processor_id,
						service_context->region,
						helper_page_mask,
						new_prot,
						UvmEventMapRemoteCauseThrashing,
						&va_block->tracker);
				if (status != NV_OK)
					return status;

				// Remove thrashing pages from the map mask
				pages_need_mapping = uvm_page_mask_andnot(helper_page_mask,
						map_prot_mask,
						&service_context->thrashing_pin_mask);
				if (!pages_need_mapping)
					continue;

				map_prot_mask = helper_page_mask;
			}
		}

		status = rhac_uvm_va_block_map(va_block,
				&service_context->block_context,
				processor_id,
				service_context->region,
				map_prot_mask,
				new_prot,
				UvmEventMapRemoteCausePolicy,
				&va_block->tracker);
		if (status != NV_OK)
			return status;
	}

	// 4- If pages did migrate, map SetAccessedBy processors, except for UVM-Lite
	for_each_id_in_mask(new_residency, &service_context->resident_processors) {
		const uvm_page_mask_t *new_residency_mask;
		new_residency_mask = &service_context->per_processor_masks[uvm_id_value(new_residency)].new_residency;

		for (new_prot = UVM_PROT_READ_ONLY; new_prot <= UVM_PROT_READ_WRITE_ATOMIC; ++new_prot) {
			uvm_page_mask_t *map_prot_mask = &service_context->block_context.caller_page_mask;
			bool pages_need_mapping;

			if (service_context->mappings_by_prot[new_prot-1].count == 0)
				continue;

			pages_need_mapping = uvm_page_mask_and(map_prot_mask,
					new_residency_mask,
					&service_context->mappings_by_prot[new_prot-1].page_mask);
			if (!pages_need_mapping)
				continue;

			// Map pages that are thrashing
			if (service_context->thrashing_pin_count > 0) {
				uvm_page_index_t page_index;

				for_each_va_block_page_in_region_mask(page_index,
						&service_context->thrashing_pin_mask,
						service_context->region) {
					uvm_processor_mask_t *map_thrashing_processors = NULL;
					NvU64 page_addr = uvm_va_block_cpu_page_address(va_block, page_index);

					// Check protection type
					if (!uvm_page_mask_test(map_prot_mask, page_index))
						continue;

					map_thrashing_processors = rhacuvm_uvm_perf_thrashing_get_thrashing_processors(va_block, page_addr);

					status = rhacuvm_uvm_va_block_add_mappings_after_migration(va_block,
							&service_context->block_context,
							new_residency,
							processor_id,
							uvm_va_block_region_for_page(page_index),
							map_prot_mask,
							new_prot,
							map_thrashing_processors);
					if (status != NV_OK)
						return status;
				}

				pages_need_mapping = uvm_page_mask_andnot(map_prot_mask,
						map_prot_mask,
						&service_context->thrashing_pin_mask);
				if (!pages_need_mapping)
					continue;
			}

			// Map the the rest of pages in a single shot
			status = rhacuvm_uvm_va_block_add_mappings_after_migration(va_block,
					&service_context->block_context,
					new_residency,
					processor_id,
					service_context->region,
					map_prot_mask,
					new_prot,
					NULL);
			if (status != NV_OK)
				return status;
		}
	}

	return NV_OK;
}

int rhac_uvm_va_block_service_locked_global(
		struct rhac_comm *comm,
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		uvm_service_block_context_t *service_context
		)
{
	NV_STATUS status = NV_OK;
	uvm_va_block_retry_t block_retry;

	uvm_prot_t new_prot;
	uvm_va_range_t *va_range = va_block->va_range;
	uvm_va_space_t *va_space = va_range->va_space;
	uvm_perf_prefetch_hint_t prefetch_hint = UVM_PERF_PREFETCH_HINT_NONE();

	uvm_assert_mutex_locked(&va_block->lock);
	UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_MANAGED);

	// GPU fault servicing must be done under the VA space read lock. GPU fault
	// servicing is required for RM to make forward progress, and we allow other
	// threads to call into RM while holding the VA space lock in read mode. If
	// we took the VA space lock in write mode on the GPU fault service path,
	// we could deadlock because the thread in RM which holds the VA space lock
	// for read wouldn't be able to complete until fault servicing completes.
	if (service_context->operation != UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS || UVM_ID_IS_CPU(processor_id))
		uvm_assert_rwsem_locked(&va_space->lock);
	else
		uvm_assert_rwsem_locked_read(&va_space->lock);

	rhac_set_prefetch_pages(processor_id, va_block, va_range, va_space, service_context, &prefetch_hint);

	for (new_prot = UVM_PROT_READ_ONLY; new_prot < UVM_PROT_MAX; ++new_prot)
		service_context->mappings_by_prot[new_prot-1].count = 0;

	status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &block_retry,
			rhac_uvm_migrate_pages_global(comm, processor_id, va_block, va_range, va_space,
				&block_retry, service_context, &prefetch_hint));
	if (status != NV_OK) {
		RHAC_ASSERT(false);
		return status;
	}

	return NV_OK;
}

int rhac_nvidia_make_resident_from_cpu(
		uvm_va_block_t *va_block,
		uvm_va_block_context_t *block_context,
		uvm_processor_id_t dst_id,
		uvm_va_block_region_t region,
    const uvm_page_mask_t *page_mask,
		const uvm_page_mask_t *copy_mask,
		uvm_va_block_transfer_mode_t mode
		)
{
	NV_STATUS status;

	//uvm_tracker_t local_tracker = UVM_TRACKER_INIT();

	uvm_page_mask_t *cpu_resident_mask = block_resident_mask_get_alloc(va_block, UVM_ID_CPU);
	RHAC_ASSERT(cpu_resident_mask);

	uvm_page_mask_or(cpu_resident_mask, cpu_resident_mask, copy_mask);
	block_set_resident_processor(va_block, UVM_ID_CPU);


	if (UVM_ID_IS_GPU(dst_id)) {
		// Call first for allocation
		uvm_va_block_gpu_state_t *state = block_gpu_state_get_alloc(va_block, block_get_gpu(va_block, dst_id));
		RHAC_ASSERT(state);

//		RHAC_ASSERT(!uvm_page_mask_intersects(&state->pte_bits[UVM_PTE_BITS_GPU_WRITE], copy_mask));
//		RHAC_ASSERT(!uvm_page_mask_intersects(&state->pte_bits[UVM_PTE_BITS_GPU_WRITE], copy_mask));

		// FIXME
		uvm_page_mask_andnot(&state->resident, &state->resident, copy_mask);
		uvm_page_mask_andnot(&state->pte_bits[UVM_PTE_BITS_GPU_READ], &state->pte_bits[UVM_PTE_BITS_GPU_READ], copy_mask);
	}

	status = block_copy_resident_pages(va_block,
		block_context,
		dst_id,
		region,
		page_mask,
		NULL,
		mode);
	RHAC_ASSERT(status == NV_OK);

	if (UVM_ID_IS_GPU(dst_id)) {
		if (!uvm_page_mask_andnot(cpu_resident_mask, cpu_resident_mask, copy_mask))
			block_clear_resident_processor(va_block, UVM_ID_CPU);
	}

		
	return 0;
}
