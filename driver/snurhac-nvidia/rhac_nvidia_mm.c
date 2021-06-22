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

#include <asm/page.h>

#include "rhac_ctx.h"
#include "rhac_utils.h"
#include "rhac_nvidia_decl.h"
#include "rhac_nvidia_helpers.h"
#include "rhac_nvidia_mm.h"
#include "rhac_nvidia_symbols.h"
#include "rhac_nvidia_pipeline.h"
#include "rhac_nvidia_prefetch.h"
#include "rhac_comm.h"

#include "rhac_nvidia_prefetch.h"

#include "nvidia-uvm/uvm8_forward_decl.h"
#include "nvidia-uvm/uvm8_gpu.h"
#include "nvidia-uvm/uvm8_hal.h"
#include "nvidia-uvm/uvm8_hal_types.h"
#include "nvidia-uvm/uvm8_va_space.h"
#include "nvidia-uvm/uvm8_va_space_mm.h"
#include "nvidia-uvm/uvm8_va_range.h"
#include "nvidia-uvm/uvm8_va_block.h"
#include "nvidia-uvm/uvm8_processors.h"

static void down_write_va_space(uvm_va_space_t *va_space)
{
	struct mm_struct *mm = rhacuvm_uvm_va_space_mm_retain(va_space);
	if (mm) {
		uvm_down_write_mmap_sem(&mm->mmap_sem);
	} else {
		mm = rhac_ctx_get_global()->mm;
		uvm_down_write_mmap_sem(&mm->mmap_sem);
	}
	uvm_va_space_down_write(va_space);
}

static void up_write_va_space(uvm_va_space_t *va_space)
{
	struct mm_struct *mm = rhacuvm_uvm_va_space_mm_retain(va_space);
	if (mm) {
		uvm_up_write_mmap_sem(&mm->mmap_sem);
	} else {
		mm = rhac_ctx_get_global()->mm;
		uvm_up_write_mmap_sem(&mm->mmap_sem);
	}
	uvm_va_space_up_write(va_space);
}

static void down_read_va_space(uvm_va_space_t *va_space)
{
	//FIXME
	rhacuvm_uvm_va_space_mm_retain(va_space);
	//struct mm_struct *mm = rhacuvm_uvm_va_space_mm_retain(va_space);
	/*
	if (mm) {
		uvm_down_read_mmap_sem(&mm->mmap_sem);
	} else {
		mm = rhac_ctx_get_global()->mm;
		uvm_down_read_mmap_sem(&mm->mmap_sem);
	}
	*/
	uvm_va_space_down_read(va_space);
}

static void up_read_va_space(uvm_va_space_t *va_space)
{
	uvm_va_space_up_read(va_space);
	//FIXME
	/*
	struct mm_struct *mm = rhacuvm_uvm_va_space_mm_retain(va_space);
	if (mm) {
		uvm_up_read_mmap_sem(&mm->mmap_sem);
	} else {
		mm = rhac_ctx_get_global()->mm;
		uvm_up_read_mmap_sem(&mm->mmap_sem);
	}
	*/
}

static uvm_va_space_t* mm_get_va_space(uint64_t addr)
{
	struct vm_area_struct *vma = find_vma(rhac_ctx_get_global()->mm, addr);
	RHAC_ASSERT(!IS_ERR_OR_NULL(vma));
	if (IS_ERR_OR_NULL(vma)) {
		RHAC_LOG("cannot find vma: addr %llx", addr);
		return NULL;
	}

	uvm_va_space_t *va_space = uvm_va_space_get(vma->vm_file);
	RHAC_ASSERT(va_space);

	return va_space;
}

static uvm_va_block_t* mm_get_va_block(uvm_va_space_t *va_space, uint64_t addr)
{
  NV_STATUS status = NV_OK;

  if (!va_space) {
    va_space = mm_get_va_space(addr);
  }

  uvm_va_block_t *va_block;
  status = rhacuvm_uvm_va_block_find_create(va_space, addr, &va_block);
  RHAC_ASSERT(status == NV_OK);

  return va_block;
}

void rhac_nvidia_mm_lock_blk(uint64_t blk_vaddr)
{
	uvm_va_block_t *va_block;
	uvm_va_space_t *va_space;

	va_space = mm_get_va_space(blk_vaddr);
	RHAC_ASSERT(va_space != NULL);
	down_read_va_space(va_space);

	va_block = mm_get_va_block(va_space, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);
	uvm_mutex_lock(&va_block->lock);

}

void rhac_nvidia_mm_unlock_blk(uint64_t blk_vaddr)
{
	uvm_va_block_t *va_block;
	uvm_va_space_t *va_space;

	va_space = mm_get_va_space(blk_vaddr);
	RHAC_ASSERT(va_space != NULL);

	va_block = mm_get_va_block(va_space, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);
	
	uvm_mutex_unlock(&va_block->lock);
	up_read_va_space(va_space);
}

int rhac_nvidia_mm_disable_write_async(uint64_t blk_vaddr, const unsigned long *prot_mask)
{
	NV_STATUS status = NV_OK;
	uvm_va_block_t *va_block;
	uvm_va_block_context_t *block_context;

	uvm_page_mask_t *revocation_mask = (uvm_page_mask_t*) prot_mask;


	struct rhac_isr_block *blk = 
		rhac_isr_table_blk_find(&rhac_ctx_get_global()->isr_table,
				blk_vaddr);


	//
	// 0 .setup
	//
	va_block = mm_get_va_block(NULL, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);

	block_context = &blk->block_context;

	// 
	// 1. disable write permissios
	// 
	if (!uvm_page_mask_empty(revocation_mask)) {
		uvm_prot_t prot;
		uvm_page_mask_t remap;
		uvm_processor_mask_t mapped;
		uvm_processor_mask_copy(&mapped, &va_block->mapped);
		bool cpu_remap = uvm_processor_mask_test(&va_block->mapped, UVM_ID_CPU);

		/*
		uvm_processor_id_t id;
		for_each_id_in_mask(id, &va_block->mapped) {
			status = rhacuvm_uvm_va_block_unmap(
					va_block,
					block_context,
					id,
					uvm_va_block_region_from_block(va_block),
					revocation_mask,
					&va_block->tracker);
			if (status != NV_OK) {
				RHAC_ASSERT(false);
			}
		}
		*/

		if (cpu_remap) {
			// In the case of CPU, uvm_va_block_reokve_prot_mask unamp than revoke protection,
			// so we remap the pages
			uvm_page_mask_and(&remap,
					block_map_with_prot_mask_get(va_block, UVM_ID_CPU, UVM_PROT_READ_ONLY),
					revocation_mask
					);
		}

		for (prot = UVM_PROT_READ_WRITE; prot < UVM_PROT_MAX; prot++) {
			status = rhacuvm_uvm_va_block_revoke_prot_mask(
					va_block,
					block_context,
					&va_block->mapped,
					uvm_va_block_region_from_block(va_block),
					revocation_mask,
					prot);
			RHAC_ASSERT(status == NV_OK);
		}

		//RHAC_ASSERT(!(cpu_remap && uvm_processor_mask_test(&va_block->mapped, UVM_ID_CPU)));

		if (cpu_remap) {
			block_context->mm = uvm_va_range_vma(va_block->va_range)->vm_mm;
			status = rhac_uvm_va_block_map(va_block,
					block_context,
					UVM_ID_CPU,
					uvm_va_block_region_from_block(va_block),
					&remap,
					UVM_PROT_READ_ONLY,
					UvmEventMapRemoteCausePolicy,
					&va_block->tracker);
			RHAC_ASSERT(status == NV_OK);
		}
	}

	RHAC_ASSERT(status == NV_OK);
	if (status != NV_OK) return -EINVAL;

	return 0;
}

int rhac_nvidia_mm_populate(uint64_t blk_vaddr, const unsigned long *page_mask)
{
	NV_STATUS status = NV_OK;
	uvm_va_block_t *va_block;
	uvm_va_block_context_t *block_context;

	va_block = mm_get_va_block(NULL, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);

	struct rhac_isr_block *blk = 
		rhac_isr_table_blk_find(&rhac_ctx_get_global()->isr_table,
				blk_vaddr);
	block_context = &blk->block_context;

	status = block_populate_pages(va_block,
			NULL,
			block_context,
			UVM_ID_CPU,
			uvm_va_block_region_from_block(va_block),
			(uvm_page_mask_t*)page_mask);
	RHAC_ASSERT(status == NV_OK);

	return status == NV_OK;
}

int rhac_nvidia_mm_stage_to_cpu_async(uint64_t blk_vaddr, const unsigned long *page_mask)
{
	NV_STATUS status = NV_OK;
	uvm_va_block_t *va_block;
	uvm_va_block_context_t *block_context;

	uvm_processor_id_t id;

	uvm_page_mask_t *copy_page_mask = (uvm_page_mask_t*)page_mask;

	va_block = mm_get_va_block(NULL, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);

	struct rhac_isr_block *blk = 
		rhac_isr_table_blk_find(&rhac_ctx_get_global()->isr_table,
				blk_vaddr);
	block_context = &blk->block_context;


	// 
	// stage to cpu 
	// if a page is in CPU, it is either RD or not
	// if it is RD in CPU, we doesn't have to read the pages from GPUs.
	// else if it is non RD in CPU, it is the page
	status = block_populate_pages(va_block,
			NULL,
			block_context,
			UVM_ID_CPU,
			uvm_va_block_region_from_block(va_block),
			copy_page_mask);
	RHAC_ASSERT(status == NV_OK);

	NvU32 copied_pages = 0;

	uvm_processor_mask_t resident_processor_mask;
	uvm_processor_mask_copy(&resident_processor_mask, &va_block->resident);
	uvm_processor_mask_clear(&resident_processor_mask, UVM_ID_CPU);
	if (uvm_processor_mask_empty(&resident_processor_mask)) {
		goto OUT; // Early exit
	}

	uvm_page_mask_t *cpu_resident_mask = block_resident_mask_get_alloc(va_block, UVM_ID_CPU);
	uvm_page_mask_t to_migrate_pages, migrated_pages, may_rd;

	uvm_page_mask_zero(&migrated_pages);
	uvm_page_mask_andnot(&to_migrate_pages, copy_page_mask, cpu_resident_mask);
	uvm_page_mask_andnot(&may_rd, &to_migrate_pages, &va_block->read_duplicated_pages);

	for_each_id_in_mask(id, &resident_processor_mask) {
		status = block_copy_resident_pages_between(
				va_block, 
				block_context,
				UVM_ID_CPU,
				id,
				uvm_va_block_region_from_block(va_block),
				&to_migrate_pages,
				NULL,
				BLOCK_TRANSFER_MODE_INTERNAL_COPY_TO_STAGE,
				&migrated_pages,
				&copied_pages,
				&va_block->tracker);
		RHAC_ASSERT(status == NV_OK);


	}
	uvm_page_mask_andnot(&va_block->read_duplicated_pages, &va_block->read_duplicated_pages, &may_rd);
	if (!uvm_page_mask_andnot(cpu_resident_mask, cpu_resident_mask, &to_migrate_pages)) {
		block_clear_resident_processor(va_block, UVM_ID_CPU);
	}

OUT: 
	RHAC_ASSERT(status == NV_OK);
	if (status != NV_OK) return -EINVAL;

	return 0;
}

int rhac_nvidia_mm_stage_to_cpu(uint64_t blk_vaddr, const unsigned long *page_mask)
{
	NV_STATUS status = NV_OK;
	uvm_va_block_t *va_block;
	uvm_va_block_context_t *block_context;

	uvm_processor_id_t id;

	uvm_page_mask_t *copy_page_mask = (uvm_page_mask_t*)page_mask;

	va_block = mm_get_va_block(NULL, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);

	//	block_context = rhacuvm_uvm_va_block_context_alloc();
	//	RHAC_ASSERT(block_context);

	struct rhac_isr_block *blk = 
		rhac_isr_table_blk_find(&rhac_ctx_get_global()->isr_table,
				blk_vaddr);
	block_context = &blk->block_context;


	// 
	// stage to cpu 
	// if a page is in CPU, it is either RD or not
	// if it is RD in CPU, we doesn't have to read the pages from GPUs.
	// else if it is non RD in CPU, it is the page
	status = block_populate_pages(va_block,
			NULL,
			block_context,
			UVM_ID_CPU,
			uvm_va_block_region_from_block(va_block),
			copy_page_mask);
	RHAC_ASSERT(status == NV_OK);

	NvU32 copied_pages = 0;

	uvm_processor_mask_t resident_processor_mask;
	uvm_processor_mask_copy(&resident_processor_mask, &va_block->resident);
	uvm_processor_mask_clear(&resident_processor_mask, UVM_ID_CPU);
	if (uvm_processor_mask_empty(&resident_processor_mask)) {
		goto OUT; // Early exit
	}

	uvm_page_mask_t *cpu_resident_mask = block_resident_mask_get_alloc(va_block, UVM_ID_CPU);
	uvm_page_mask_t to_migrate_pages, migrated_pages, may_rd;

	uvm_page_mask_zero(&migrated_pages);
	uvm_page_mask_andnot(&to_migrate_pages, copy_page_mask, cpu_resident_mask);

	uvm_page_mask_andnot(&may_rd, &to_migrate_pages, &va_block->read_duplicated_pages);

	for_each_id_in_mask(id, &resident_processor_mask) {
		status = block_copy_resident_pages_between(
				va_block, 
				block_context,
				UVM_ID_CPU,
				id,
				uvm_va_block_region_from_block(va_block),
				&to_migrate_pages,
				NULL,
				BLOCK_TRANSFER_MODE_INTERNAL_COPY_TO_STAGE,
				&migrated_pages,
				&copied_pages,
				&va_block->tracker);
		RHAC_ASSERT(status == NV_OK);
	}
	uvm_page_mask_andnot(&va_block->read_duplicated_pages, &va_block->read_duplicated_pages, &may_rd);
	if (!uvm_page_mask_andnot(cpu_resident_mask, cpu_resident_mask, &may_rd)) {
		block_clear_resident_processor(va_block, UVM_ID_CPU);
	}

OUT: 
	status = rhacuvm_uvm_tracker_wait(&va_block->tracker);
	RHAC_ASSERT(status == NV_OK);

	if (status != NV_OK) return -EINVAL;

	return 0;
}

int rhac_nvidia_mm_inv(uint64_t blk_vaddr, const unsigned long *invmask)
{
	NV_STATUS status = NV_OK;

	uvm_va_block_t *va_block;
	uvm_processor_id_t id;

	va_block = mm_get_va_block(NULL, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);

	struct rhac_isr_block *blk = 
		rhac_isr_table_blk_find(&rhac_ctx_get_global()->isr_table,
				blk_vaddr);

	uvm_va_block_context_t *block_context = &blk->block_context;

	block_context->mm = uvm_va_range_vma(va_block->va_range)->vm_mm;
	uvm_page_mask_t *unmap_page_mask = (uvm_page_mask_t*) invmask;
	for_each_id_in_mask(id, &va_block->mapped) {
		status = rhacuvm_uvm_va_block_unmap(
				va_block,
				block_context,
				id,
				uvm_va_block_region_from_block(va_block),
				unmap_page_mask,
				&va_block->tracker);
		if (status != NV_OK) {
			RHAC_ASSERT(false);
			goto OUT;
		}
	}

	for_each_id_in_mask(id, &va_block->resident) {
		uvm_page_mask_t *resident_mask = rhacuvm_uvm_va_block_resident_mask_get(va_block, id);
		if (!uvm_page_mask_andnot(resident_mask, resident_mask, unmap_page_mask)) 
			block_clear_resident_processor(va_block, id);

	}

OUT:

	status = rhacuvm_uvm_tracker_wait(&va_block->tracker);
	RHAC_ASSERT(status == NV_OK);
	if (status != NV_OK) return -EINVAL;
	return 0;
}

static bool always_split(uvm_va_range_t *va_range, void *data) 
{
	return true;
}

int rhac_nvidia_mm_split_va_range(uint64_t blk_vaddr, uint64_t len)
{
	uvm_va_space_t *va_space;
	NV_STATUS status;

	va_space = mm_get_va_space(blk_vaddr);
	RHAC_ASSERT(va_space != NULL);

	down_write_va_space(va_space);

	status = rhacuvm_uvm_va_space_split_span_as_needed(
			va_space,
			blk_vaddr,
			blk_vaddr + len,
			always_split,
			NULL);
	RHAC_ASSERT(status == NV_OK);

	up_write_va_space(va_space);

	return 0;
}

int rhac_nvidia_mm_toggle_dup_flag(uint64_t blk_vaddr, uint64_t size, uint32_t on)
{
	uvm_va_space_t *va_space;
	uvm_va_range_t *va_range;

	va_space = mm_get_va_space(blk_vaddr);
	RHAC_ASSERT(va_space != NULL);

	down_write_va_space(va_space);

	if (on) {
		va_range = rhacuvm_uvm_va_range_find(va_space, (NvU64)blk_vaddr);
		if (va_range == NULL) {
			RHAC_LOG("Failed to get va_range from vaddr=%lx",
					(long unsigned int)blk_vaddr);
			return -EINVAL;
		}

		if (va_range->type == UVM_VA_RANGE_TYPE_MANAGED)
			va_range->read_duplication = UVM_READ_DUPLICATION_ENABLED;
	} else {
		va_range = rhacuvm_uvm_va_range_find(va_space, (NvU64)blk_vaddr);
		if (va_range == NULL) {
			RHAC_LOG("Failed to get va_range from vaddr=%lx",
					(long unsigned int)blk_vaddr);
			return -EINVAL;
		}

		if (va_range->type == UVM_VA_RANGE_TYPE_MANAGED)
			va_range->read_duplication = UVM_READ_DUPLICATION_DISABLED;
	}

	up_write_va_space(va_space);

	return 0;
}

int rhac_nvidia_mm_prefetch_to_cpu(uint64_t blk_vaddr, uint64_t size, uint32_t device_id, bool is_async)
{
	int err;
	NV_STATUS status;
	uvm_va_space_t *va_space;
	uvm_tracker_t tracker = UVM_TRACKER_INIT();

	va_space = mm_get_va_space(blk_vaddr);
	RHAC_ASSERT(va_space != NULL);

	down_read_va_space(va_space);

	struct rhac_comm *pa = NULL;

	if (!is_async) {
		pa = rhac_comm_alloc();
		RHAC_ASSERT(pa);
		pa->type = 2;
	}

	uvm_mutex_t lock;
	uvm_mutex_init(&lock, UVM_LOCK_ORDER_VA_BLOCK);
	status = rhac_uvm_migrate(
			pa,
			va_space,
			blk_vaddr,
			size,
			uvm_id(device_id + 1),
			UVM_MIGRATE_FLAG_ASYNC,
			is_async ? NULL : &tracker,
			&lock
			);
	RHAC_ASSERT(status == NV_OK);

	if (!is_async) {
		err = rhac_comm_wait(pa);
		if (err) {
			RHAC_ASSERT(!err);
			return err;
		}
		rhac_comm_free(pa);

		rhacuvm_uvm_tracker_wait(&tracker);
		rhacuvm_uvm_tracker_deinit(&tracker);
	}

	up_read_va_space(va_space);


	return 0;
}

int rhac_nvidia_mm_prefetch_to_gpu(uint64_t blk_vaddr, uvm_page_mask_t *page_mask, uint32_t device_id)
{
	NV_STATUS status;
	uvm_va_space_t *va_space;
	uvm_va_range_t *va_range;
  uvm_va_block_t *va_block;
  uvm_va_block_region_t region;

	va_space = mm_get_va_space(blk_vaddr);
	RHAC_ASSERT(va_space != NULL);

	down_read_va_space(va_space);

	va_range = rhacuvm_uvm_va_space_iter_first(va_space, blk_vaddr, blk_vaddr);
	RHAC_ASSERT(va_range);

  const size_t block_index = rhacuvm_uvm_va_range_block_index(va_range, blk_vaddr);
  status = rhacuvm_uvm_va_range_block_create(va_range, block_index, &va_block);
  RHAC_ASSERT(va_block);
  if (status != NV_OK)
    return status;

  // calculate region based on page_mask
  uvm_page_index_t first_index =
    find_next_bit(page_mask->bitmap, PAGES_PER_UVM_VA_BLOCK, 0);
  uvm_page_index_t last_index = first_index;
  uvm_page_index_t candidate_index;
  while (1) {
    candidate_index =
      find_next_bit(page_mask->bitmap, PAGES_PER_UVM_VA_BLOCK, last_index + 1);
    if (candidate_index == PAGES_PER_UVM_VA_BLOCK)
      break;
    last_index = candidate_index;
  }
  region = uvm_va_block_region(first_index, last_index + 1);

  status = rhac_nvidia_pipeline_prefetch(
      NULL,
      va_block,
      region,
      page_mask,
      uvm_id(device_id + 1),
      UVM_MIGRATE_MODE_MAKE_RESIDENT_AND_MAP,
      NULL,
      NULL);
  RHAC_ASSERT(status == NV_OK);

	up_read_va_space(va_space);

	return 0;
}

int rhac_nvidia_mm_make_resident_cpu(
		uint64_t blk_vaddr,
		uvm_processor_id_t dst_id,
		const unsigned long *mask)
{
	uvm_va_block_t *va_block;

	va_block = mm_get_va_block(NULL, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);

	uvm_page_mask_t *cpu_resident_mask = block_resident_mask_get_alloc(va_block, UVM_ID_CPU);
	uvm_page_mask_t *stage_mask = (uvm_page_mask_t*) mask;

	if (!uvm_page_mask_empty(stage_mask)) {
		uvm_page_mask_or(cpu_resident_mask, cpu_resident_mask, stage_mask);
		if (UVM_ID_IS_GPU(dst_id)) {
			uvm_va_block_gpu_state_t *state = block_gpu_state_get(va_block, uvm_gpu_id(uvm_id_value(dst_id)));

      // FIXME
			uvm_page_mask_andnot(&state->resident, &state->resident, stage_mask);
			uvm_page_mask_andnot(&state->pte_bits[UVM_PTE_BITS_GPU_READ], &state->pte_bits[UVM_PTE_BITS_GPU_READ], stage_mask);
			uvm_page_mask_andnot(&state->pte_bits[UVM_PTE_BITS_GPU_WRITE], &state->pte_bits[UVM_PTE_BITS_GPU_WRITE], stage_mask);
			uvm_page_mask_andnot(&state->pte_bits[UVM_PTE_BITS_GPU_ATOMIC], &state->pte_bits[UVM_PTE_BITS_GPU_ATOMIC], stage_mask);
		}
		block_set_resident_processor(va_block, UVM_ID_CPU);
	}

	return NV_OK;
}


struct page** rhac_nvidia_mm_get_pages(uint64_t blk_vaddr, const unsigned long *mask)
{
	uvm_va_block_t *va_block;

	va_block = mm_get_va_block(NULL, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);

	return &va_block->cpu.pages[0];
}

int rhac_nvidia_mm_copy_to_buf(uint64_t blk_vaddr, const unsigned long *mask, void *buf)
{
	NV_STATUS status;
	uvm_va_block_t *va_block;

	va_block = mm_get_va_block(NULL, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);

	status = rhacuvm_uvm_tracker_wait(&va_block->tracker);
	RHAC_ASSERT(status == NV_OK);

	int i;
	for_each_set_bit(i, mask, RHAC_PDSC_PER_PBLK) {
		RHAC_ASSERT(!IS_ERR_OR_NULL(va_block->cpu.pages[i]));
		void *v = kmap_atomic(va_block->cpu.pages[i]);
    RHAC_ASSERT(!IS_ERR_OR_NULL(v));

    memcpy(buf, v, PAGE_SIZE);
    __kunmap_atomic(v);

		buf += PAGE_SIZE;
	}
	return 0;
}

static NV_STATUS rhac_block_populate_pages(
    uvm_va_block_t *block,
		uvm_va_block_retry_t *retry,
		uvm_page_mask_t *populate_page_mask,
		uvm_processor_id_t dest_id,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *page_mask)
{
	NV_STATUS status;
	const uvm_page_mask_t *resident_mask = block_resident_mask_get_alloc(block, dest_id);
	uvm_page_index_t page_index;

	if (!resident_mask)
		return NV_ERR_NO_MEMORY;

	if (page_mask)
		uvm_page_mask_andnot(populate_page_mask, page_mask, resident_mask);
	else
		uvm_page_mask_complement(populate_page_mask, resident_mask);

	if (UVM_ID_IS_GPU(dest_id))
		return block_populate_pages_gpu(block, retry, block_get_gpu(block, dest_id), region, populate_page_mask);

	for_each_va_block_page_in_region_mask(page_index, populate_page_mask, region) {
		uvm_processor_mask_t resident_on;
		bool resident_somewhere;
		rhacuvm_uvm_va_block_page_resident_processors(block, page_index, &resident_on);
		resident_somewhere = !uvm_processor_mask_empty(&resident_on);

		// For pages not resident anywhere, need to populate with zeroed memory
		status = block_populate_page_cpu(block, page_index, !resident_somewhere);
		if (status != NV_OK)
			return status;
	}

	return NV_OK;
}

int rhac_nvidia_mm_copy_from_buf(uint64_t blk_vaddr, const unsigned long *mask, void *buf)
{
	NV_STATUS status;
	uvm_va_block_t *va_block;

	va_block = mm_get_va_block(NULL, blk_vaddr);
	RHAC_ASSERT(va_block != NULL);

	//struct rhac_isr_block *blk = rhac_isr_table_blk_find(&rhac_ctx_get_global()->isr_table, blk_vaddr);
	//uvm_va_block_context_t *block_context = &blk->block_context;

	uvm_page_mask_t populate_page_mask;

	status = rhac_block_populate_pages(va_block,
			NULL,
			&populate_page_mask,
			UVM_ID_CPU,
			uvm_va_block_region_from_block(va_block),
			(uvm_page_mask_t*)mask);
	RHAC_ASSERT(status == NV_OK);

	int i;
  for_each_set_bit(i, mask, RHAC_PDSC_PER_PBLK) {
    RHAC_ASSERT(!IS_ERR_OR_NULL(va_block->cpu.pages[i]));

    if (!IS_ERR_OR_NULL(va_block->cpu.pages[i])) {

      void *v = kmap_atomic(va_block->cpu.pages[i]);

      RHAC_ASSERT(!IS_ERR_OR_NULL(v));

      memcpy(v, buf, PAGE_SIZE);
      __kunmap_atomic(v);
      buf += PAGE_SIZE;
    }
  }

	return 0;
}
