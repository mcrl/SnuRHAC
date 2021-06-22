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

#ifndef __RHAC_NVIDIA_DECL_H__
#define __RHAC_NVIDIA_DECL_H__

#include "nvidia-uvm/uvm8_forward_decl.h"
#include "nvidia-uvm/uvm8_hal.h"
#include "nvidia-uvm/uvm8_kvmalloc.h"
#include "rhac_nvidia_symbols.h"
#include "rhac_nvidia_helpers.h"
#include "rhac_utils.h"

/*
#ifdef UVM_ASSERT
#undef UVM_ASSERT
#define UVM_ASSERT(e) RHAC_ASSERT(e)
#endif
*/


extern struct kmem_cache *gg_uvm_va_block_gpu_state_cache;
extern struct kmem_cache *gg_uvm_page_mask_cache;

typedef enum
{
	BLOCK_TRANSFER_MODE_INTERNAL_MOVE            = 1,
	BLOCK_TRANSFER_MODE_INTERNAL_COPY            = 2,
	BLOCK_TRANSFER_MODE_INTERNAL_MOVE_TO_STAGE   = 3,
	BLOCK_TRANSFER_MODE_INTERNAL_MOVE_FROM_STAGE = 4,
	BLOCK_TRANSFER_MODE_INTERNAL_COPY_TO_STAGE   = 5,
	BLOCK_TRANSFER_MODE_INTERNAL_COPY_FROM_STAGE = 6
} block_transfer_mode_internal_t;


typedef struct
{
	// Processor the page is on
	uvm_processor_id_t processor;

	// The page index
	uvm_page_index_t page_index;
} block_phys_page_t;


static uvm_va_block_gpu_state_t *block_gpu_state_get(uvm_va_block_t *block, uvm_gpu_id_t gpu_id)
{
	return block->gpus[uvm_id_gpu_index(gpu_id)];
}

static size_t block_gpu_chunk_index(uvm_va_block_t *block,
		uvm_gpu_t *gpu,
		uvm_page_index_t page_index,
		uvm_chunk_size_t *out_chunk_size)
{
	uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, gpu->id);
	uvm_chunk_size_t size;
	uvm_gpu_chunk_t *chunk;

	size_t index = rhacuvm_uvm_va_block_gpu_chunk_index_range(block->start, uvm_va_block_size(block), gpu, page_index, &size);

	UVM_ASSERT(size >= PAGE_SIZE);

	if (gpu_state) {
		UVM_ASSERT(gpu_state->chunks);
		chunk = gpu_state->chunks[index];
		if (chunk) {
			UVM_ASSERT(uvm_gpu_chunk_get_size(chunk) == size);
			UVM_ASSERT(chunk->state != UVM_PMM_GPU_CHUNK_STATE_PMA_OWNED);
			UVM_ASSERT(chunk->state != UVM_PMM_GPU_CHUNK_STATE_FREE);
		}
	}

	if (out_chunk_size)
		*out_chunk_size = size;

	return index;
}

static size_t block_num_gpu_chunks(uvm_va_block_t *block, uvm_gpu_t *gpu)
{
	return block_gpu_chunk_index(block, gpu, uvm_va_block_cpu_page_index(block, block->end), NULL) + 1;
}


static uvm_gpu_t *block_get_gpu(uvm_va_block_t *block, uvm_gpu_id_t gpu_id)
{
	UVM_ASSERT(block->va_range);
	UVM_ASSERT(block->va_range->va_space);

	return uvm_va_space_get_gpu(block->va_range->va_space, gpu_id);
}


static void block_gpu_unmap_phys_all_cpu_pages(uvm_va_block_t *block, uvm_gpu_t *gpu)
{
	uvm_page_index_t page_index;
	uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, gpu->id);

	for_each_va_block_page(page_index, block) {
		if (gpu_state->cpu_pages_dma_addrs[page_index] == 0)
			continue;

		rhacuvm_uvm_pmm_sysmem_mappings_remove_gpu_mapping(&gpu->pmm_sysmem_mappings,
				gpu_state->cpu_pages_dma_addrs[page_index]);

		uvm_gpu_unmap_cpu_page(gpu, gpu_state->cpu_pages_dma_addrs[page_index]);
		gpu_state->cpu_pages_dma_addrs[page_index] = 0;
	}
}

static NV_STATUS block_gpu_map_phys_all_cpu_pages(uvm_va_block_t *block, uvm_gpu_t *gpu)
{
	NV_STATUS status;
	uvm_page_index_t page_index;
	uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, gpu->id);

	for_each_va_block_page(page_index, block) {
		if (!block->cpu.pages[page_index])
			continue;

		status = uvm_gpu_map_cpu_page(gpu, block->cpu.pages[page_index], &gpu_state->cpu_pages_dma_addrs[page_index]);
		if (status != NV_OK)
			goto error;

		status = rhacuvm_uvm_pmm_sysmem_mappings_add_gpu_mapping(&gpu->pmm_sysmem_mappings,
				gpu_state->cpu_pages_dma_addrs[page_index],
				uvm_va_block_cpu_page_address(block, page_index),
				PAGE_SIZE,
				block,
				UVM_ID_CPU);
		if (status != NV_OK)
			goto error;
	}

	return NV_OK;

error:
	block_gpu_unmap_phys_all_cpu_pages(block, gpu);
	return status;
}

static uvm_va_block_gpu_state_t *block_gpu_state_get_alloc(uvm_va_block_t *block, uvm_gpu_t *gpu)
{
	NV_STATUS status;
	uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, gpu->id);

	if (gpu_state)
		return gpu_state;

	gpu_state = nv_kmem_cache_zalloc(gg_uvm_va_block_gpu_state_cache, NV_UVM_GFP_FLAGS);
	if (!gpu_state) {
		return NULL;
	}

	gpu_state->chunks = uvm_kvmalloc_zero(block_num_gpu_chunks(block, gpu) * sizeof(gpu_state->chunks[0]));
	if (!gpu_state->chunks)
		goto error;

	gpu_state->cpu_pages_dma_addrs = uvm_kvmalloc_zero(uvm_va_block_num_cpu_pages(block) * sizeof(gpu_state->cpu_pages_dma_addrs[0]));
	if (!gpu_state->cpu_pages_dma_addrs)
		goto error;

	block->gpus[uvm_id_gpu_index(gpu->id)] = gpu_state;

	status = block_gpu_map_phys_all_cpu_pages(block, gpu);
	if (status != NV_OK)
		goto error;

	return gpu_state;

error:
	if (gpu_state) {
		if (gpu_state->chunks)
			rhacuvm_uvm_kvfree(gpu_state->chunks);
		kmem_cache_free(gg_uvm_va_block_gpu_state_cache, gpu_state);
	}
	block->gpus[uvm_id_gpu_index(gpu->id)] = NULL;

	return NULL;
}


static uvm_page_mask_t *block_resident_mask_get_alloc(uvm_va_block_t *block, uvm_processor_id_t processor)
{
	uvm_va_block_gpu_state_t *gpu_state;

	if (UVM_ID_IS_CPU(processor))
		return &block->cpu.resident;

	gpu_state = block_gpu_state_get_alloc(block, block_get_gpu(block, processor));
	if (!gpu_state)
		return NULL;

	return &gpu_state->resident;
}

static void block_mark_memory_used(uvm_va_block_t *block, uvm_processor_id_t id)
{
	uvm_gpu_t *gpu;

	if (UVM_ID_IS_CPU(id))
		return;

	gpu = block_get_gpu(block, id);

	// If the block is of the max size and the GPU supports eviction, mark the
	// root chunk as used in PMM.
	if (uvm_va_block_size(block) == UVM_CHUNK_SIZE_MAX && uvm_gpu_supports_eviction(gpu)) {
		// The chunk has to be there if this GPU is resident
		UVM_ASSERT(uvm_processor_mask_test(&block->resident, id));
		rhacuvm_uvm_pmm_gpu_mark_root_chunk_used(&gpu->pmm, block_gpu_state_get(block, gpu->id)->chunks[0]);
	}
}

static void block_set_resident_processor(uvm_va_block_t *block, uvm_processor_id_t id)
{
	UVM_ASSERT(!uvm_page_mask_empty(rhacuvm_uvm_va_block_resident_mask_get(block, id)));

	if (uvm_processor_mask_test_and_set(&block->resident, id))
		return;

	block_mark_memory_used(block, id);
}

static void block_clear_resident_processor(uvm_va_block_t *block, uvm_processor_id_t id)
{
	uvm_gpu_t *gpu;

	UVM_ASSERT(uvm_page_mask_empty(rhacuvm_uvm_va_block_resident_mask_get(block, id)));

	if (!uvm_processor_mask_test_and_clear(&block->resident, id))
		return;

	if (UVM_ID_IS_CPU(id))
		return;

	gpu = block_get_gpu(block, id);

	// If the block is of the max size and the GPU supports eviction, mark the
	// root chunk as unused in PMM.
	if (uvm_va_block_size(block) == UVM_CHUNK_SIZE_MAX && uvm_gpu_supports_eviction(gpu)) {
		// The chunk may not be there any more when residency is cleared.
		uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, gpu->id);
		if (gpu_state && gpu_state->chunks[0])
			rhacuvm_uvm_pmm_gpu_mark_root_chunk_unused(&gpu->pmm, gpu_state->chunks[0]);
	}
}

static uvm_va_block_transfer_mode_t get_block_transfer_mode_from_internal(block_transfer_mode_internal_t transfer_mode)
{
	switch (transfer_mode) {
		case BLOCK_TRANSFER_MODE_INTERNAL_MOVE:
		case BLOCK_TRANSFER_MODE_INTERNAL_MOVE_TO_STAGE:
		case BLOCK_TRANSFER_MODE_INTERNAL_MOVE_FROM_STAGE:
			return UVM_VA_BLOCK_TRANSFER_MODE_MOVE;

		case BLOCK_TRANSFER_MODE_INTERNAL_COPY:
		case BLOCK_TRANSFER_MODE_INTERNAL_COPY_TO_STAGE:
		case BLOCK_TRANSFER_MODE_INTERNAL_COPY_FROM_STAGE:
			return UVM_VA_BLOCK_TRANSFER_MODE_COPY;
	}

	UVM_ASSERT_MSG(0, "Invalid transfer mode %u\n", transfer_mode);
	return 0;
}

static bool block_check_resident_proximity(uvm_va_block_t *block, uvm_page_index_t page_index, uvm_processor_id_t new_residency)
{
	uvm_processor_mask_t resident_procs, mapped_procs;
	uvm_processor_id_t mapped_id, closest_id;

	uvm_processor_mask_andnot(&mapped_procs, &block->mapped, &block->va_range->uvm_lite_gpus);

	for_each_id_in_mask(mapped_id, &mapped_procs) {
		if (!uvm_page_mask_test(rhacuvm_uvm_va_block_map_mask_get(block, mapped_id), page_index))
			continue;

		rhacuvm_uvm_va_block_page_resident_processors(block, page_index, &resident_procs);
		UVM_ASSERT(!uvm_processor_mask_empty(&resident_procs));
		UVM_ASSERT(!uvm_processor_mask_test(&resident_procs, new_residency));
		uvm_processor_mask_set(&resident_procs, new_residency);
		closest_id = rhacuvm_uvm_processor_mask_find_closest_id(block->va_range->va_space, &resident_procs, mapped_id);
		UVM_ASSERT(!uvm_id_equal(closest_id, new_residency));
	}

	return true;
}

static bool block_page_is_clean(uvm_va_block_t *block,
		uvm_processor_id_t dst_id,
		uvm_processor_id_t src_id,
		uvm_page_index_t page_index)
{
	return uvm_id_equal(dst_id, block->va_range->preferred_location) &&
		UVM_ID_IS_CPU(src_id) &&
		!block_get_gpu(block, dst_id)->isr.replayable_faults.handling &&
		!PageDirty(block->cpu.pages[page_index]);
}



static bool block_processor_page_is_populated(uvm_va_block_t *block, uvm_processor_id_t proc, uvm_page_index_t page_index)
{
	uvm_va_block_gpu_state_t *gpu_state;
	size_t chunk_index;

	if (UVM_ID_IS_CPU(proc))
		return block->cpu.pages[page_index] != NULL;

	gpu_state = block_gpu_state_get(block, proc);
	if (!gpu_state)
		return false;

	chunk_index = block_gpu_chunk_index(block, block_get_gpu(block, proc), page_index, NULL);
	return gpu_state->chunks[chunk_index] != NULL;
}

static uvm_chunk_size_t block_gpu_chunk_size(uvm_va_block_t *block, uvm_gpu_t *gpu, uvm_page_index_t start_page_index)
{
	uvm_chunk_sizes_mask_t chunk_sizes = gpu->mmu_user_chunk_sizes;
	uvm_chunk_sizes_mask_t start_alignments, pow2_leq_size, allowed_sizes;
	NvU64 start = uvm_va_block_cpu_page_address(block, start_page_index);
	NvU64 size = block->end - start + 1;

	// Create a mask of all sizes for which start is aligned. x ^ (x-1) yields a
	// mask of the rightmost 1 bit in x, as well as all trailing 0 bits in x.
	// Example: 1011000 -> 0001111
	start_alignments = (uvm_chunk_sizes_mask_t)(start ^ (start - 1));

	// Next, compute all sizes (powers of two) which are <= size.
	pow2_leq_size = (uvm_chunk_sizes_mask_t)rounddown_pow_of_two(size);
	pow2_leq_size |= pow2_leq_size - 1;

	// Now and them all together to get our list of GPU-supported chunk sizes
	// which are aligned to start and will fit within size.
	allowed_sizes = chunk_sizes & start_alignments & pow2_leq_size;

	// start and size must always be aligned to at least the smallest supported
	// chunk size (PAGE_SIZE).
	UVM_ASSERT(allowed_sizes >= PAGE_SIZE);

	// Take the largest allowed size
	return uvm_chunk_find_last_size(allowed_sizes);
}

static bool is_block_phys_contig(uvm_va_block_t *block, uvm_processor_id_t id)
{
	// Check if the VA block has a single physically-contiguous chunk of storage
	// on the GPU
	return (UVM_ID_IS_GPU(id)) &&
		(uvm_va_block_size(block) == block_gpu_chunk_size(block, block_get_gpu(block, id), 0));
}


static const uvm_processor_mask_t *block_get_can_copy_from_mask(uvm_va_block_t *block, uvm_processor_id_t from)
{
	return &block->va_range->va_space->can_copy_from[uvm_id_value(from)];
}

static bool block_can_copy_from(uvm_va_block_t *va_block, uvm_processor_id_t from, uvm_processor_id_t to)
{
	return uvm_processor_mask_test(block_get_can_copy_from_mask(va_block, to), from);
}

static const char *block_processor_name(uvm_va_block_t *block, uvm_processor_id_t id)
{
	UVM_ASSERT(block->va_range);
	UVM_ASSERT(block->va_range->va_space);

	return uvm_va_space_processor_name(block->va_range->va_space, id);
}

static uvm_va_block_region_t block_gpu_chunk_region(uvm_va_block_t *block,
		uvm_chunk_size_t chunk_size,
		uvm_page_index_t page_index)
{
	NvU64 page_addr = uvm_va_block_cpu_page_address(block, page_index);
	NvU64 chunk_start_addr = UVM_ALIGN_DOWN(page_addr, chunk_size);
	uvm_page_index_t first = (uvm_page_index_t)((chunk_start_addr - block->start) / PAGE_SIZE);
	return uvm_va_block_region(first, first + (chunk_size / PAGE_SIZE));
}


static uvm_gpu_chunk_t *block_phys_page_chunk(uvm_va_block_t *block, block_phys_page_t block_page, size_t *chunk_offset)
{
    uvm_gpu_t *gpu = block_get_gpu(block, block_page.processor);
    uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, block_page.processor);
    size_t chunk_index;
    uvm_gpu_chunk_t *chunk;
    uvm_chunk_size_t chunk_size;

    UVM_ASSERT(gpu_state);

    chunk_index = block_gpu_chunk_index(block, gpu, block_page.page_index, &chunk_size);
    chunk = gpu_state->chunks[chunk_index];
    UVM_ASSERT(chunk);

    if (chunk_offset) {
        size_t page_offset = block_page.page_index -
                             block_gpu_chunk_region(block,chunk_size, block_page.page_index).first;
        *chunk_offset = page_offset * PAGE_SIZE;
    }

    return chunk;
}

static bool block_gpu_page_is_swizzled(uvm_va_block_t *block, uvm_gpu_t *gpu, uvm_page_index_t page_index)
{
    NvU32 big_page_size;
    size_t big_page_index;
    uvm_va_block_gpu_state_t *gpu_state;

    if (!gpu->big_page.swizzling)
        return false;

    gpu_state = block_gpu_state_get(block, gpu->id);
    UVM_ASSERT(gpu_state);

    big_page_size = rhacuvm_uvm_va_block_gpu_big_page_size(block, gpu);
    big_page_index = rhacuvm_uvm_va_block_big_page_index(block, page_index, big_page_size);

    return big_page_index != MAX_BIG_PAGES_PER_UVM_VA_BLOCK && test_bit(big_page_index, gpu_state->big_pages_swizzled);
}

static uvm_gpu_phys_address_t block_phys_page_address(uvm_va_block_t *block,
                                                      block_phys_page_t block_page,
                                                      uvm_gpu_t *gpu)
{
    uvm_va_block_gpu_state_t *accessing_gpu_state = block_gpu_state_get(block, gpu->id);
    size_t chunk_offset;
    uvm_gpu_chunk_t *chunk;

    UVM_ASSERT(accessing_gpu_state);

    if (UVM_ID_IS_CPU(block_page.processor)) {
        NvU64 dma_addr = accessing_gpu_state->cpu_pages_dma_addrs[block_page.page_index];

        // The page should be mapped for physical access already as we do that
        // eagerly on CPU page population and GPU state alloc.
        UVM_ASSERT(dma_addr != 0);

        return uvm_gpu_phys_address(UVM_APERTURE_SYS, dma_addr);
    }

    chunk = block_phys_page_chunk(block, block_page, &chunk_offset);

    if (uvm_id_equal(block_page.processor, gpu->id)) {
        return uvm_gpu_phys_address(UVM_APERTURE_VID, chunk->address + chunk_offset);
    }
    else {
        uvm_gpu_phys_address_t phys_addr;
        uvm_gpu_t *owning_gpu = block_get_gpu(block, block_page.processor);
        UVM_ASSERT(rhacuvm_uvm_va_space_peer_enabled(block->va_range->va_space, gpu, owning_gpu));
        phys_addr = rhacuvm_uvm_pmm_gpu_peer_phys_address(&owning_gpu->pmm, chunk, gpu);
        phys_addr.address += chunk_offset;
        return phys_addr;
    }
}

static uvm_gpu_address_t block_phys_page_copy_address(uvm_va_block_t *block,
		block_phys_page_t block_page,
		uvm_gpu_t *gpu)
{
	uvm_gpu_t *owning_gpu;
	size_t chunk_offset;
	uvm_gpu_chunk_t *chunk;
	uvm_gpu_address_t copy_addr;

	UVM_ASSERT_MSG(block_can_copy_from(block, gpu->id, block_page.processor),
			"from %s to %s\n",
			block_processor_name(block, gpu->id),
			block_processor_name(block, block_page.processor));

	// CPU and local GPU accesses can use block_phys_page_address
	if (UVM_ID_IS_CPU(block_page.processor) || uvm_id_equal(block_page.processor, gpu->id)) {
		copy_addr = uvm_gpu_address_from_phys(block_phys_page_address(block, block_page, gpu));

		// If this page is currently in a swizzled big page format, we have to
		// copy using the big page identity mapping in order to deswizzle.
		if (uvm_id_equal(block_page.processor, gpu->id) &&
				block_gpu_page_is_swizzled(block, gpu, block_page.page_index)) {
			return rhacuvm_uvm_mmu_gpu_address_for_big_page_physical(copy_addr, gpu);
		}

		return copy_addr;
	}

	// See the comments on the peer_identity_mappings_supported assignments in
	// the HAL for why we disable direct copies between peers.
	owning_gpu = block_get_gpu(block, block_page.processor);

	// GPUs which swizzle in particular must never have direct copies because
	// then we'd need to create both big and 4k mappings.
	UVM_ASSERT(!gpu->big_page.swizzling);
	UVM_ASSERT(!owning_gpu->big_page.swizzling);

	UVM_ASSERT(rhacuvm_uvm_va_space_peer_enabled(block->va_range->va_space, gpu, owning_gpu));

	chunk = block_phys_page_chunk(block, block_page, &chunk_offset);
	copy_addr = rhacuvm_uvm_pmm_gpu_peer_copy_address(&owning_gpu->pmm, chunk, gpu);
	copy_addr.address += chunk_offset;
	return copy_addr;
}

static block_phys_page_t block_phys_page(uvm_processor_id_t processor, uvm_page_index_t page_index)
{
	return (block_phys_page_t){ processor, page_index };
}

static bool block_phys_copy_contig_check(uvm_va_block_t *block,
		uvm_page_index_t page_index,
		const uvm_gpu_address_t *base_address,
		uvm_processor_id_t proc_id,
		uvm_gpu_t *copying_gpu)
{
	uvm_gpu_address_t page_address;
	uvm_gpu_address_t contig_address = *base_address;

	contig_address.address += page_index * PAGE_SIZE;

	page_address = block_phys_page_copy_address(block, block_phys_page(proc_id, page_index), copying_gpu);

	return uvm_gpu_addr_cmp(page_address, contig_address) == 0;
}

static void block_update_page_dirty_state(uvm_va_block_t *block,
		uvm_processor_id_t dst_id,
		uvm_processor_id_t src_id,
		uvm_page_index_t page_index)
{
	if (UVM_ID_IS_GPU(dst_id))
		return;

	if (uvm_id_equal(src_id, block->va_range->preferred_location))
		ClearPageDirty(block->cpu.pages[page_index]);
	else
		SetPageDirty(block->cpu.pages[page_index]);
}

static NV_STATUS block_copy_begin_push(uvm_va_block_t *va_block,
		uvm_processor_id_t dst_id,
		uvm_processor_id_t src_id,
		uvm_tracker_t *tracker,
		uvm_push_t *push)
{
	uvm_channel_type_t channel_type;
	uvm_gpu_t *gpu;

	UVM_ASSERT_MSG(!uvm_id_equal(src_id, dst_id),
			"Unexpected copy to self, processor %s\n",
			block_processor_name(va_block, src_id));

	if (UVM_ID_IS_CPU(src_id)) {
		gpu = block_get_gpu(va_block, dst_id);
		channel_type = UVM_CHANNEL_TYPE_CPU_TO_GPU;
	}
	else if (UVM_ID_IS_CPU(dst_id)) {
		gpu = block_get_gpu(va_block, src_id);
		channel_type = UVM_CHANNEL_TYPE_GPU_TO_CPU;
	}
	else {
		// For GPU to GPU copies, prefer to "push" the data from the source as
		// that works better at least for P2P over PCI-E.
		gpu = block_get_gpu(va_block, src_id);

		channel_type = UVM_CHANNEL_TYPE_GPU_TO_GPU;
	}

	UVM_ASSERT_MSG(block_can_copy_from(va_block, gpu->id, dst_id),
			"GPU %s dst %s src %s\n",
			block_processor_name(va_block, gpu->id),
			block_processor_name(va_block, dst_id),
			block_processor_name(va_block, src_id));
	UVM_ASSERT_MSG(block_can_copy_from(va_block, gpu->id, src_id),
			"GPU %s dst %s src %s\n",
			block_processor_name(va_block, gpu->id),
			block_processor_name(va_block, dst_id),
			block_processor_name(va_block, src_id));

	if (channel_type == UVM_CHANNEL_TYPE_GPU_TO_GPU) {
		uvm_gpu_t *dst_gpu = block_get_gpu(va_block, dst_id);
		return uvm_push_begin_acquire_gpu_to_gpu(gpu->channel_manager,
				dst_gpu,
				tracker,
				push,
				"Copy from %s to %s for block [0x%llx, 0x%llx]",
				block_processor_name(va_block, src_id),
				block_processor_name(va_block, dst_id),
				va_block->start,
				va_block->end);
	}

	return uvm_push_begin_acquire(gpu->channel_manager,
			channel_type,
			tracker,
			push,
			"Copy from %s to %s for block [0x%llx, 0x%llx]",
			block_processor_name(va_block, src_id),
			block_processor_name(va_block, dst_id),
			va_block->start,
			va_block->end);
}

static void break_read_duplication_in_region(uvm_va_block_t *block,
		uvm_va_block_context_t *block_context,
		uvm_processor_id_t dst_id,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *page_mask)
{
	uvm_processor_id_t id;
	uvm_page_mask_t *break_pages_in_region = &block_context->scratch_page_mask;

	uvm_page_mask_init_from_region(break_pages_in_region, region, page_mask);

	UVM_ASSERT(uvm_page_mask_subset(break_pages_in_region, rhacuvm_uvm_va_block_resident_mask_get(block, dst_id)));

	// Clear read_duplicated bit for all pages in region
	uvm_page_mask_andnot(&block->read_duplicated_pages, &block->read_duplicated_pages, break_pages_in_region);

	// Clear residency bits for all processors other than dst_id
	for_each_id_in_mask(id, &block->resident) {
		uvm_page_mask_t *other_resident_mask;

		if (uvm_id_equal(id, dst_id))
			continue;

		other_resident_mask = rhacuvm_uvm_va_block_resident_mask_get(block, id);

		if (!uvm_page_mask_andnot(other_resident_mask, other_resident_mask, break_pages_in_region))
			block_clear_resident_processor(block, id);
	}
}

static bool block_is_page_resident_anywhere(uvm_va_block_t *block, uvm_page_index_t page_index)
{
	uvm_processor_id_t id;
	for_each_id_in_mask(id, &block->resident) {
		if (uvm_page_mask_test(rhacuvm_uvm_va_block_resident_mask_get(block, id), page_index))
			return true;
	}

	return false;
}

static void block_copy_set_first_touch_residency(uvm_va_block_t *block,
		uvm_va_block_context_t *block_context,
		uvm_processor_id_t dst_id,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *page_mask)
{
	uvm_page_index_t page_index;
	uvm_page_mask_t *resident_mask = rhacuvm_uvm_va_block_resident_mask_get(block, dst_id);
	uvm_page_mask_t *first_touch_mask = &block_context->make_resident.page_mask;

	if (page_mask)
		uvm_page_mask_andnot(first_touch_mask, page_mask, resident_mask);
	else
		uvm_page_mask_complement(first_touch_mask, resident_mask);

	uvm_page_mask_region_clear_outside(first_touch_mask, region);

	for_each_va_block_page_in_mask(page_index, first_touch_mask, block) {
		UVM_ASSERT(!block_is_page_resident_anywhere(block, page_index));
		UVM_ASSERT(block_processor_page_is_populated(block, dst_id, page_index));
		UVM_ASSERT(block_check_resident_proximity(block, page_index, dst_id));
	}

	uvm_page_mask_or(resident_mask, resident_mask, first_touch_mask);
	if (!uvm_page_mask_empty(resident_mask))
		block_set_resident_processor(block, dst_id);

	// Add them to the output mask, too
	uvm_page_mask_or(&block_context->make_resident.pages_changed_residency,
			&block_context->make_resident.pages_changed_residency,
			first_touch_mask);
}

static uvm_gpu_chunk_t *block_retry_get_free_chunk(uvm_va_block_retry_t *retry, uvm_gpu_t *gpu, uvm_chunk_size_t size)
{
	uvm_gpu_chunk_t *gpu_chunk;

	list_for_each_entry(gpu_chunk, &retry->free_chunks, list) {
		if (rhacuvm_uvm_gpu_chunk_get_gpu(gpu_chunk) == gpu && uvm_gpu_chunk_get_size(gpu_chunk) == size) {
			list_del_init(&gpu_chunk->list);
			return gpu_chunk;
		}
	}

	return NULL;
}

static void block_retry_add_free_chunk(uvm_va_block_retry_t *retry, uvm_gpu_chunk_t *gpu_chunk)
{
    list_add_tail(&gpu_chunk->list, &retry->free_chunks);
}


static NV_STATUS block_alloc_gpu_chunk(uvm_va_block_t *block,
		uvm_va_block_retry_t *retry,
		uvm_gpu_t *gpu,
		uvm_chunk_size_t size,
		uvm_gpu_chunk_t **out_gpu_chunk)
{
	NV_STATUS status = NV_OK;
	uvm_gpu_chunk_t *gpu_chunk;

	// First try getting a free chunk from previously-made allocations.
	gpu_chunk = block_retry_get_free_chunk(retry, gpu, size);
	if (!gpu_chunk) {
		uvm_va_block_test_t *block_test = uvm_va_block_get_test(block);
		if (block_test && block_test->user_pages_allocation_retry_force_count > 0) {
			// Force eviction by pretending the allocation failed with no memory
			--block_test->user_pages_allocation_retry_force_count;
			status = NV_ERR_NO_MEMORY;
		}
		else {
			// Try allocating a new one without eviction
			status = uvm_pmm_gpu_alloc_user(&gpu->pmm, 1, size, UVM_PMM_ALLOC_FLAGS_NONE, &gpu_chunk, &retry->tracker);
		}

		if (status == NV_ERR_NO_MEMORY) {
			// If that fails with no memory, try allocating with eviction and
			// return back to the caller immediately so that the operation can
			// be restarted.
			uvm_mutex_unlock(&block->lock);

			status = uvm_pmm_gpu_alloc_user(&gpu->pmm, 1, size, UVM_PMM_ALLOC_FLAGS_EVICT, &gpu_chunk, &retry->tracker);
			if (status == NV_OK) {
				block_retry_add_free_chunk(retry, gpu_chunk);
				status = NV_ERR_MORE_PROCESSING_REQUIRED;
			}

			uvm_mutex_lock(&block->lock);
			return status;
		}
		else if (status != NV_OK) {
			return status;
		}
	}

	*out_gpu_chunk = gpu_chunk;
	return NV_OK;
}

static NV_STATUS block_zero_new_gpu_chunk(uvm_va_block_t *block,
		uvm_gpu_t *gpu,
		uvm_gpu_chunk_t *chunk,
		uvm_va_block_region_t chunk_region,
		uvm_tracker_t *tracker)
{
	uvm_va_block_gpu_state_t *gpu_state;
	NV_STATUS status;
	uvm_gpu_address_t memset_addr_base, memset_addr, phys_addr;
	uvm_push_t push;
	uvm_gpu_id_t id;
	uvm_va_block_region_t subregion, big_region_all;
	uvm_page_mask_t *zero_mask;
	bool big_page_swizzle = false;
	NvU32 big_page_size = 0;

	UVM_ASSERT(uvm_va_block_region_size(chunk_region) == uvm_gpu_chunk_get_size(chunk));

	if (chunk->is_zero)
		return NV_OK;

	gpu_state = block_gpu_state_get(block, gpu->id);
	zero_mask = kmem_cache_alloc(gg_uvm_page_mask_cache, NV_UVM_GFP_FLAGS);

	if (!zero_mask)
		return NV_ERR_NO_MEMORY;

	if (gpu->big_page.swizzling) {
		// When populating we don't yet know what the mapping will be, so we
		// don't know whether this will be initially mapped as a big page (which
		// must be swizzled) or as 4k pages (which must not). Our common case
		// for first populate on swizzled GPUs (UVM-Lite) is full migration and
		// mapping of an entire block, so assume that we will swizzle if the
		// block is large enough to fit a big page. If we're wrong, the big page
		// will be deswizzled at map time.
		//
		// Note also that this chunk might be able to fit more than one big
		// page.
		big_page_size = rhacuvm_uvm_va_block_gpu_big_page_size(block, gpu);
		big_region_all = rhacuvm_uvm_va_block_big_page_region_all(block, big_page_size);

		// Note that this condition also handles the case of having no big pages
		// in the block, in which case big_region_all is {0. 0}.
		if (uvm_va_block_region_contains_region(big_region_all, chunk_region)) {
			big_page_swizzle = true;
			UVM_ASSERT(uvm_gpu_chunk_get_size(chunk) >= big_page_size);
		}
	}

	phys_addr = uvm_gpu_address_physical(UVM_APERTURE_VID, chunk->address);
	if (big_page_swizzle)
		memset_addr_base = rhacuvm_uvm_mmu_gpu_address_for_big_page_physical(phys_addr, gpu);
	else
		memset_addr_base = phys_addr;

	memset_addr = memset_addr_base;

	// Tradeoff: zeroing entire chunk vs zeroing only the pages needed for the
	// operation.
	//
	// We may over-zero the page with this approach. For example, we might be
	// populating a 2MB chunk because only a single page within that chunk needs
	// to be made resident. If we also zero non-resident pages outside of the
	// strict region, we could waste the effort if those pages are populated on
	// another processor later and migrated here.
	//
	// We zero all non-resident pages in the chunk anyway for two reasons:
	//
	// 1) Efficiency. It's better to do all zeros as pipelined transfers once
	//    rather than scatter them around for each populate operation.
	//
	// 2) Optimizing the common case of block_populate_gpu_chunk being called
	//    for already-populated chunks. If we zero once at initial populate, we
	//    can simply check whether the chunk is present in the array. Otherwise
	//    we'd have to recompute the "is any page resident" mask every time.

	// Roll up all pages in chunk_region which are resident somewhere
	uvm_page_mask_zero(zero_mask);
	for_each_id_in_mask(id, &block->resident)
		uvm_page_mask_or(zero_mask, zero_mask, rhacuvm_uvm_va_block_resident_mask_get(block, id));

	// If all pages in the chunk are resident somewhere, we don't need to clear
	// anything. Just make sure the chunk is tracked properly.
	if (uvm_page_mask_region_full(zero_mask, chunk_region)) {
		status = rhacuvm_uvm_tracker_add_tracker_safe(&block->tracker, tracker);
		goto out;
	}

	// Complement to get the pages which are not resident anywhere. These
	// are the pages which must be zeroed.
	uvm_page_mask_complement(zero_mask, zero_mask);

	status = uvm_push_begin_acquire(gpu->channel_manager, UVM_CHANNEL_TYPE_GPU_INTERNAL, tracker, &push,
			"Zero out chunk [0x%llx, 0x%llx) for region [0x%llx, 0x%llx) in va block [0x%llx, 0x%llx)",
			chunk->address,
			chunk->address + uvm_gpu_chunk_get_size(chunk),
			uvm_va_block_region_start(block, chunk_region),
			uvm_va_block_region_end(block, chunk_region) + 1,
			block->start, block->end + 1);
	if (status != NV_OK)
		goto out;

	for_each_va_block_subregion_in_mask(subregion, zero_mask, chunk_region) {
		// Pipeline the memsets since they never overlap with each other
		uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_PIPELINED);

		// We'll push one membar later for all memsets in this loop
		uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);

		memset_addr.address = memset_addr_base.address + (subregion.first - chunk_region.first) * PAGE_SIZE;
		gpu->ce_hal->memset_8(&push, memset_addr, 0, uvm_va_block_region_size(subregion));
	}

	// A membar from this GPU is required between this memset and any PTE write
	// pointing this or another GPU to this chunk. Otherwise an engine could
	// read the PTE then access the page before the memset write is visible to
	// that engine.
	//
	// This memset writes GPU memory, so local mappings need only a GPU-local
	// membar. We can't easily determine here whether a peer GPU will ever map
	// this page in the future, so always use a sysmembar. uvm_push_end provides
	// one by default.
	//
	// TODO: Bug 1766424: Use GPU-local membars if no peer can currently map
	//       this page. When peer access gets enabled, do a MEMBAR_SYS at that
	//       point.
	rhacuvm_uvm_push_end(&push);
	status = rhacuvm_uvm_tracker_add_push_safe(&block->tracker, &push);

out:
	if (big_page_swizzle && status == NV_OK) {
		// Set big_pages_swizzled for each big page region covered by the new
		// chunk. We do this regardless of whether we actually wrote anything,
		// since this controls how the initial data will be copied into the
		// page later. See the above comment on big_page_swizzle.
		bitmap_set(gpu_state->big_pages_swizzled,
				rhacuvm_uvm_va_block_big_page_index(block, chunk_region.first, big_page_size),
				(size_t)uvm_div_pow2_64(uvm_va_block_region_size(chunk_region), big_page_size));
	}

	if (zero_mask)
		kmem_cache_free(gg_uvm_page_mask_cache, zero_mask);

	return status;
}

static void block_sysmem_mappings_remove_gpu_chunk(uvm_gpu_t *local_gpu,
                                                   uvm_gpu_chunk_t *chunk,
                                                   uvm_gpu_t *accessing_gpu)
{
    NvU64 peer_addr = rhacuvm_uvm_pmm_gpu_indirect_peer_addr(&local_gpu->pmm, chunk, accessing_gpu);
    uvm_pmm_sysmem_mappings_remove_gpu_chunk_mapping(&accessing_gpu->pmm_sysmem_mappings, peer_addr);
}

static void block_unmap_indirect_peers_from_gpu_chunk(uvm_va_block_t *block, uvm_gpu_t *gpu, uvm_gpu_chunk_t *chunk)
{
    uvm_va_space_t *va_space = block->va_range->va_space;
    uvm_gpu_t *peer_gpu;

    uvm_assert_rwsem_locked(&va_space->lock);
    uvm_assert_mutex_locked(&block->lock);

    // Indirect peer mappings are removed lazily by PMM, so we only need to
    // remove the sysmem reverse mappings.
    for_each_va_space_gpu_in_mask(peer_gpu, va_space, &va_space->indirect_peers[uvm_id_value(gpu->id)])
        block_sysmem_mappings_remove_gpu_chunk(gpu, chunk, peer_gpu);
}

static void block_retry_add_used_chunk(uvm_va_block_retry_t *retry, uvm_gpu_chunk_t *gpu_chunk)
{
    list_add_tail(&gpu_chunk->list, &retry->used_chunks);
}

static NV_STATUS block_sysmem_mappings_add_gpu_chunk(uvm_va_block_t *block,
                                                     uvm_gpu_t *local_gpu,
                                                     uvm_gpu_chunk_t *chunk,
                                                     uvm_gpu_t *accessing_gpu)
{
    NvU64 peer_addr = rhacuvm_uvm_pmm_gpu_indirect_peer_addr(&local_gpu->pmm, chunk, accessing_gpu);
    return uvm_pmm_sysmem_mappings_add_gpu_chunk_mapping(&accessing_gpu->pmm_sysmem_mappings,
                                                         peer_addr,
                                                         block->start + chunk->va_block_page_index * PAGE_SIZE,
                                                         uvm_gpu_chunk_get_size(chunk),
                                                         block,
                                                         local_gpu->id);
}

static NV_STATUS block_map_indirect_peers_to_gpu_chunk(uvm_va_block_t *block, uvm_gpu_t *gpu, uvm_gpu_chunk_t *chunk)
{
    uvm_va_space_t *va_space = block->va_range->va_space;
    uvm_gpu_t *accessing_gpu, *remove_gpu;
    NV_STATUS status;

    // Unlike block_map_phys_cpu_page_on_gpus, this function isn't called on the
    // eviction path, so we can assume that the VA space is locked.
    //
    // TODO: Bug 2007346: In the future we may want to enable eviction to peers,
    //       meaning we may need to allocate peer memory and map it on the
    //       eviction path. That will require making sure that peers can't be
    //       enabled or disabled either in the VA space or globally within this
    //       function.
    uvm_assert_rwsem_locked(&va_space->lock);
    uvm_assert_mutex_locked(&block->lock);

    for_each_va_space_gpu_in_mask(accessing_gpu, va_space, &va_space->indirect_peers[uvm_id_value(gpu->id)]) {
        status = rhacuvm_uvm_pmm_gpu_indirect_peer_map(&gpu->pmm, chunk, accessing_gpu);
        if (status != NV_OK)
            goto error;

        status = block_sysmem_mappings_add_gpu_chunk(block, gpu, chunk, accessing_gpu);
        if (status != NV_OK)
            goto error;
    }

    return NV_OK;

error:
    for_each_va_space_gpu_in_mask(remove_gpu, va_space, &va_space->indirect_peers[uvm_id_value(gpu->id)]) {
        if (remove_gpu == accessing_gpu)
            break;

        // Indirect peer mappings are removed lazily by PMM, so if an error
        // occurs the mappings established above will be removed when the
        // chunk is freed later on. We only need to remove the sysmem
        // reverse mappings.
        block_sysmem_mappings_remove_gpu_chunk(gpu, chunk, remove_gpu);
    }

    return status;
}

static NV_STATUS block_populate_gpu_chunk(uvm_va_block_t *block,
		uvm_va_block_retry_t *retry,
		uvm_gpu_t *gpu,
		size_t chunk_index,
		uvm_va_block_region_t chunk_region)
{
	uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get_alloc(block, gpu);
	uvm_gpu_chunk_t *chunk = NULL;
	uvm_chunk_size_t chunk_size = uvm_va_block_region_size(chunk_region);
	uvm_va_block_test_t *block_test = uvm_va_block_get_test(block);
	NV_STATUS status;
	bool mapped_indirect_peers = false;

	if (!gpu_state)
		return NV_ERR_NO_MEMORY;

	UVM_ASSERT(chunk_index < block_num_gpu_chunks(block, gpu));
	UVM_ASSERT(chunk_size & gpu->mmu_user_chunk_sizes);

	// We zero chunks as necessary at initial population, so if the chunk is
	// already populated we're done. See the comment in
	// block_zero_new_gpu_chunk.
	if (gpu_state->chunks[chunk_index])
		return NV_OK;

	UVM_ASSERT(uvm_page_mask_region_empty(&gpu_state->resident, chunk_region));

	status = block_alloc_gpu_chunk(block, retry, gpu, chunk_size, &chunk);
	if (status != NV_OK)
		return status;

	status = block_zero_new_gpu_chunk(block, gpu, chunk, chunk_region, &retry->tracker);
	if (status != NV_OK)
		goto error;

	// It is safe to modify the page index field without holding any PMM locks
	// because the chunk is pinned, which means that none of the other fields in
	// the bitmap can change.
	chunk->va_block_page_index = chunk_region.first;

	// va_block_page_index is a bitfield of size PAGE_SHIFT. Make sure at
	// compile-time that it can store VA Block page indexes.
	BUILD_BUG_ON(PAGES_PER_UVM_VA_BLOCK >= PAGE_SIZE);

	status = block_map_indirect_peers_to_gpu_chunk(block, gpu, chunk);
	if (status != NV_OK)
		goto error;

	mapped_indirect_peers = true;

	if (block_test && block_test->inject_populate_error) {
		block_test->inject_populate_error = false;

		// Use NV_ERR_MORE_PROCESSING_REQUIRED to force a retry rather than
		// causing a fatal OOM failure.
		status = NV_ERR_MORE_PROCESSING_REQUIRED;
		goto error;
	}

	// Record the used chunk so that it can be unpinned at the end of the whole
	// operation.
	block_retry_add_used_chunk(retry, chunk);
	gpu_state->chunks[chunk_index] = chunk;

	return NV_OK;

error:
	if (mapped_indirect_peers)
		block_unmap_indirect_peers_from_gpu_chunk(block, gpu, chunk);

	// block_zero_new_gpu_chunk may have pushed memsets on this chunk which it
	// placed in the block tracker.
	rhacuvm_uvm_pmm_gpu_free(&gpu->pmm, chunk, &block->tracker);
	return status;
}


static NV_STATUS block_populate_pages_gpu(uvm_va_block_t *block,
		uvm_va_block_retry_t *retry,
		uvm_gpu_t *gpu,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *populate_mask)
{
	uvm_va_block_region_t chunk_region, check_region;
	size_t chunk_index;
	uvm_page_index_t page_index;
	uvm_chunk_size_t chunk_size;
	NV_STATUS status;

	page_index = uvm_va_block_first_page_in_mask(region, populate_mask);
	if (page_index == region.outer)
		return NV_OK;

	chunk_index = block_gpu_chunk_index(block, gpu, page_index, &chunk_size);
	chunk_region = block_gpu_chunk_region(block, chunk_size, page_index);

	while (1) {
		check_region = uvm_va_block_region(max(chunk_region.first, region.first),
				min(chunk_region.outer, region.outer));
		page_index = uvm_va_block_first_page_in_mask(check_region, populate_mask);
		if (page_index != check_region.outer) {
			status = block_populate_gpu_chunk(block, retry, gpu, chunk_index, chunk_region);
			if (status != NV_OK)
				return status;
		}

		if (check_region.outer == region.outer)
			break;

		++chunk_index;
		chunk_size = block_gpu_chunk_size(block, gpu, chunk_region.outer);
		chunk_region = uvm_va_block_region(chunk_region.outer, chunk_region.outer + (chunk_size / PAGE_SIZE));
	}

	return NV_OK;
}

static void block_unmap_phys_cpu_page_on_gpus(uvm_va_block_t *block, uvm_page_index_t page_index)
{
    uvm_gpu_id_t id;

    for_each_gpu_id(id) {
        uvm_gpu_t *gpu;
        uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, id);
        if (!gpu_state)
            continue;

        if (gpu_state->cpu_pages_dma_addrs[page_index] == 0)
            continue;

        gpu = block_get_gpu(block, id);
        rhacuvm_uvm_pmm_sysmem_mappings_remove_gpu_mapping(&gpu->pmm_sysmem_mappings,
                                                   gpu_state->cpu_pages_dma_addrs[page_index]);
        uvm_gpu_unmap_cpu_page(gpu, gpu_state->cpu_pages_dma_addrs[page_index]);

        gpu_state->cpu_pages_dma_addrs[page_index] = 0;
    }
}



static NV_STATUS block_map_phys_cpu_page_on_gpus(uvm_va_block_t *block, uvm_page_index_t page_index, struct page *page)
{
    NV_STATUS status;
    uvm_gpu_id_t id;

    // We can't iterate over va_space->registered_gpus because we might be
    // on the eviction path, which does not have the VA space lock held. We have
    // the VA block lock held however, so the gpu_states can't change.
    uvm_assert_mutex_locked(&block->lock);

    for_each_gpu_id(id) {
        uvm_gpu_t *gpu;
        uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, id);
        if (!gpu_state)
            continue;

        UVM_ASSERT(gpu_state->cpu_pages_dma_addrs[page_index] == 0);

        gpu = block_get_gpu(block, id);

        status = uvm_gpu_map_cpu_page(gpu, page, &gpu_state->cpu_pages_dma_addrs[page_index]);
        if (status != NV_OK)
            goto error;

        status = rhacuvm_uvm_pmm_sysmem_mappings_add_gpu_mapping(&gpu->pmm_sysmem_mappings,
                                                         gpu_state->cpu_pages_dma_addrs[page_index],
                                                         uvm_va_block_cpu_page_address(block, page_index),
                                                         PAGE_SIZE,
                                                         block,
                                                         UVM_ID_CPU);
        if (status != NV_OK)
            goto error;
    }

    return NV_OK;

error:
    block_unmap_phys_cpu_page_on_gpus(block, page_index);
    return status;
}

static NV_STATUS block_populate_page_cpu(uvm_va_block_t *block, uvm_page_index_t page_index, bool zero)
{
	NV_STATUS status;
	struct page *page;
	gfp_t gfp_flags;
	uvm_va_block_test_t *block_test = uvm_va_block_get_test(block);

	if (block->cpu.pages[page_index])
		return NV_OK;

	UVM_ASSERT(!uvm_page_mask_test(&block->cpu.resident, page_index));

	// Return out of memory error if the tests have requested it. As opposed to
	// other error injection settings, this one is persistent.
	if (block_test && block_test->inject_cpu_pages_allocation_error)
		return NV_ERR_NO_MEMORY;

	gfp_flags = NV_UVM_GFP_FLAGS | GFP_HIGHUSER;
	if (zero)
		gfp_flags |= __GFP_ZERO;

	page = alloc_pages(gfp_flags, 0);
	if (!page)
		return NV_ERR_NO_MEMORY;

	// the kernel has 'written' zeros to this page, so it is dirty
	if (zero)
		SetPageDirty(page);

	status = block_map_phys_cpu_page_on_gpus(block, page_index, page);
	if (status != NV_OK)
		goto error;

	block->cpu.pages[page_index] = page;
	return NV_OK;

error:
	__free_page(page);
	return status;
}

static NV_STATUS block_populate_pages(uvm_va_block_t *block,
		uvm_va_block_retry_t *retry,
		uvm_va_block_context_t *block_context,
		uvm_processor_id_t dest_id,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *page_mask)
{
	NV_STATUS status;
	const uvm_page_mask_t *resident_mask = block_resident_mask_get_alloc(block, dest_id);
	uvm_page_index_t page_index;
	uvm_page_mask_t *populate_page_mask = &block_context->make_resident.page_mask;

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



static NV_STATUS block_copy_resident_pages_between(uvm_va_block_t *block,
		uvm_va_block_context_t *block_context,
		uvm_processor_id_t dst_id,
		uvm_processor_id_t src_id,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *page_mask,
		const uvm_page_mask_t *prefetch_page_mask,
		block_transfer_mode_internal_t transfer_mode,
		uvm_page_mask_t *migrated_pages,
		NvU32 *copied_pages,
		uvm_tracker_t *copy_tracker)
{
	NV_STATUS tracker_status, status = NV_OK;
	uvm_page_mask_t *src_resident_mask = rhacuvm_uvm_va_block_resident_mask_get(block, src_id);
	uvm_page_mask_t *dst_resident_mask = rhacuvm_uvm_va_block_resident_mask_get(block, dst_id);
	uvm_gpu_t *copying_gpu = NULL;
	uvm_push_t push;
	uvm_page_index_t page_index;
	uvm_page_index_t contig_start_index = region.outer;
	uvm_page_index_t last_index = region.outer;
	uvm_page_mask_t *copy_mask = &block_context->make_resident.copy_resident_pages_between_mask;
	uvm_range_group_range_t *rgr = NULL;
	bool rgr_has_changed = false;
	uvm_make_resident_cause_t cause = block_context->make_resident.cause;
	uvm_make_resident_cause_t contig_cause = cause;
	const bool may_prefetch = (cause == UVM_MAKE_RESIDENT_CAUSE_REPLAYABLE_FAULT ||
			cause == UVM_MAKE_RESIDENT_CAUSE_NON_REPLAYABLE_FAULT ||
			cause == UVM_MAKE_RESIDENT_CAUSE_ACCESS_COUNTER) && !!prefetch_page_mask;
	const bool is_src_phys_contig = is_block_phys_contig(block, src_id);
	const bool is_dst_phys_contig = is_block_phys_contig(block, dst_id);
	uvm_gpu_address_t contig_src_address = {0};
	uvm_gpu_address_t contig_dst_address = {0};
	uvm_va_range_t *va_range = block->va_range;
	uvm_va_space_t *va_space = va_range->va_space;
	const uvm_va_block_transfer_mode_t block_transfer_mode = get_block_transfer_mode_from_internal(transfer_mode);

	*copied_pages = 0;

	if (uvm_id_equal(dst_id, src_id))
		return NV_OK;

	uvm_page_mask_init_from_region(copy_mask, region, src_resident_mask);

	if (page_mask)
		uvm_page_mask_and(copy_mask, copy_mask, page_mask);

	// If there are not pages to be copied, exit early
	if (!uvm_page_mask_andnot(copy_mask, copy_mask, dst_resident_mask)) {
		return NV_OK;
	}

	// uvm_range_group_range_iter_first should only be called when the va_space
	// lock is held, which is always the case unless an eviction is taking
	// place.
	if (cause != UVM_MAKE_RESIDENT_CAUSE_EVICTION) {
		rgr = rhacuvm_uvm_range_group_range_iter_first(va_space,
				uvm_va_block_region_start(block, region),
				uvm_va_block_region_end(block, region));
		rgr_has_changed = true;
	}

	for_each_va_block_page_in_region_mask(page_index, copy_mask, region) {
		NvU64 page_start = uvm_va_block_cpu_page_address(block, page_index);
		uvm_make_resident_cause_t page_cause = (may_prefetch && uvm_page_mask_test(prefetch_page_mask, page_index))?
			UVM_MAKE_RESIDENT_CAUSE_PREFETCH:
			cause;

		UVM_ASSERT(block_check_resident_proximity(block, page_index, dst_id));

		if (UVM_ID_IS_CPU(dst_id)) {
			// To support staging through CPU, populate CPU pages on demand.
			// GPU destinations should have their pages populated already, but
			// that might change if we add staging through GPUs.
			status = block_populate_page_cpu(block, page_index, false);
			if (status != NV_OK)
				break;
		}

		UVM_ASSERT(block_processor_page_is_populated(block, dst_id, page_index));

		// If we're not evicting and we're migrating away from the preferred
		// location, then we should add the range group range to the list of
		// migrated ranges in the range group. It's safe to skip this because
		// the use of range_group's migrated_ranges list is a UVM-Lite
		// optimization - eviction is not supported on UVM-Lite GPUs.
		if (cause != UVM_MAKE_RESIDENT_CAUSE_EVICTION &&
				uvm_id_equal(src_id, va_range->preferred_location)) {
			RHAC_ASSERT(false);
			// rgr_has_changed is used to minimize the number of times the
			// migrated_ranges_lock is taken. It is set to false when the range
			// group range pointed by rgr is added to the migrated_ranges list,
			// and it is just set back to true when we move to a different
			// range group range.

			// The current page could be after the end of rgr. Iterate over the
			// range group ranges until rgr's end location is greater than or
			// equal to the current page.
			while (rgr && rgr->node.end < page_start) {
				rgr = rhacuvm_uvm_range_group_range_iter_next(va_space, rgr, uvm_va_block_region_end(block, region));
				rgr_has_changed = true;
			}

			// Check whether the current page lies within rgr. A single page
			// must entirely reside within a range group range. Since we've
			// incremented rgr until its end is higher than page_start, we now
			// check if page_start lies within rgr.
			if (rgr && rgr_has_changed && page_start >= rgr->node.start && page_start <= rgr->node.end) {
				uvm_spin_lock(&rgr->range_group->migrated_ranges_lock);
				if (list_empty(&rgr->range_group_migrated_list_node))
					list_move_tail(&rgr->range_group_migrated_list_node, &rgr->range_group->migrated_ranges);
				uvm_spin_unlock(&rgr->range_group->migrated_ranges_lock);

				rgr_has_changed = false;
			}
		}

		// No need to copy pages that haven't changed.  Just clear residency
		// information
		if (block_page_is_clean(block, dst_id, src_id, page_index)) {
			continue;
		}

		if (!copying_gpu) {
			status = block_copy_begin_push(block, dst_id, src_id, &block->tracker, &push);
			if (status != NV_OK)
				break;
			copying_gpu = uvm_push_get_gpu(&push);

			// Record all processors involved in the copy
			uvm_processor_mask_set(&block_context->make_resident.all_involved_processors, copying_gpu->id);
			uvm_processor_mask_set(&block_context->make_resident.all_involved_processors, dst_id);
			uvm_processor_mask_set(&block_context->make_resident.all_involved_processors, src_id);

			// This function is called just once per VA block and needs to
			// receive the "main" cause for the migration (it mainly checks if
			// we are in the eviction path). Therefore, we pass cause instead
			// of contig_cause
			rhacuvm_uvm_tools_record_block_migration_begin(block, &push, dst_id, src_id, page_start, cause);
		}
		else {
			uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_PIPELINED);
		}

		block_update_page_dirty_state(block, dst_id, src_id, page_index);

		if (last_index == region.outer) {
			contig_start_index = page_index;
			contig_cause = page_cause;

			// Computing the physical address is a non-trivial operation and
			// seems to be a performance limiter on systems with 2 or more
			// NVLINK links. Therefore, for physically-contiguous block
			// storage, we cache the start address and compute the page address
			// using the page index.
			if (is_src_phys_contig)
				contig_src_address = block_phys_page_copy_address(block, block_phys_page(src_id, 0), copying_gpu);
			if (is_dst_phys_contig)
				contig_dst_address = block_phys_page_copy_address(block, block_phys_page(dst_id, 0), copying_gpu);
		}
		else if ((page_index != last_index + 1) || contig_cause != page_cause) {
			uvm_va_block_region_t contig_region = uvm_va_block_region(contig_start_index, last_index + 1);
			size_t contig_region_size = uvm_va_block_region_size(contig_region);
			UVM_ASSERT(uvm_va_block_region_contains_region(region, contig_region));

			// If both src and dst are physically-contiguous, consolidate copies
			// of contiguous pages into a single method.
			if (is_src_phys_contig && is_dst_phys_contig) {
				uvm_gpu_address_t src_address = contig_src_address;
				uvm_gpu_address_t dst_address = contig_dst_address;

				src_address.address += contig_start_index * PAGE_SIZE;
				dst_address.address += contig_start_index * PAGE_SIZE;

				uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
				copying_gpu->ce_hal->memcopy(&push, dst_address, src_address, contig_region_size);
			}

			uvm_perf_event_notify_migration(&va_space->perf_events,
					&push,
					block,
					dst_id,
					src_id,
					uvm_va_block_region_start(block, contig_region),
					contig_region_size,
					block_transfer_mode,
					contig_cause,
					&block_context->make_resident);

			contig_start_index = page_index;
			contig_cause = page_cause;
		}

		if (is_src_phys_contig)
			UVM_ASSERT(block_phys_copy_contig_check(block, page_index, &contig_src_address, src_id, copying_gpu));
		if (is_dst_phys_contig)
			UVM_ASSERT(block_phys_copy_contig_check(block, page_index, &contig_dst_address, dst_id, copying_gpu));

		if (!is_src_phys_contig || !is_dst_phys_contig) {
			uvm_gpu_address_t src_address;
			uvm_gpu_address_t dst_address;

			if (is_src_phys_contig) {
				src_address = contig_src_address;
				src_address.address += page_index * PAGE_SIZE;
			}
			else {
				src_address = block_phys_page_copy_address(block, block_phys_page(src_id, page_index), copying_gpu);
			}

			if (is_dst_phys_contig) {
				dst_address = contig_dst_address;
				dst_address.address += page_index * PAGE_SIZE;
			}
			else {
				dst_address = block_phys_page_copy_address(block, block_phys_page(dst_id, page_index), copying_gpu);
			}

			uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
			copying_gpu->ce_hal->memcopy(&push, dst_address, src_address, PAGE_SIZE);
		}

		last_index = page_index;
	}

	// Copy the remaining pages
	if (copying_gpu) {
		uvm_va_block_region_t contig_region = uvm_va_block_region(contig_start_index, last_index + 1);
		size_t contig_region_size = uvm_va_block_region_size(contig_region);
		UVM_ASSERT(uvm_va_block_region_contains_region(region, contig_region));

		if (is_src_phys_contig && is_dst_phys_contig) {
			uvm_gpu_address_t src_address = contig_src_address;
			uvm_gpu_address_t dst_address = contig_dst_address;

			src_address.address += contig_start_index * PAGE_SIZE;
			dst_address.address += contig_start_index * PAGE_SIZE;

			uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
			copying_gpu->ce_hal->memcopy(&push, dst_address, src_address, contig_region_size);
		}

		uvm_perf_event_notify_migration(&va_space->perf_events,
				&push,
				block,
				dst_id,
				src_id,
				uvm_va_block_region_start(block, contig_region),
				contig_region_size,
				block_transfer_mode,
				contig_cause,
				&block_context->make_resident);

		// TODO: Bug 1766424: If the destination is a GPU and the copy was done
		//       by that GPU, use a GPU-local membar if no peer can currently
		//       map this page. When peer access gets enabled, do a MEMBAR_SYS
		//       at that point.
		rhacuvm_uvm_push_end(&push);
		tracker_status = rhacuvm_uvm_tracker_add_push_safe(copy_tracker, &push);
		if (status == NV_OK)
			status = tracker_status;
	}

	// Update VA block status bits
	//
	// Only update the bits for the pages that succeded
	if (status != NV_OK)
		uvm_page_mask_region_clear(copy_mask, uvm_va_block_region(page_index, PAGES_PER_UVM_VA_BLOCK));

	*copied_pages = uvm_page_mask_weight(copy_mask);

	if (*copied_pages) {
		uvm_page_mask_or(migrated_pages, migrated_pages, copy_mask);

		uvm_page_mask_or(dst_resident_mask, dst_resident_mask, copy_mask);
		block_set_resident_processor(block, dst_id);

		if (transfer_mode == BLOCK_TRANSFER_MODE_INTERNAL_MOVE_FROM_STAGE) {
			// Check whether there are any resident pages left on src
			if (!uvm_page_mask_andnot(src_resident_mask, src_resident_mask, copy_mask))
				block_clear_resident_processor(block, src_id);
		}

		// If we are staging the copy due to read duplication, we keep the copy there
		if (transfer_mode == BLOCK_TRANSFER_MODE_INTERNAL_COPY ||
				transfer_mode == BLOCK_TRANSFER_MODE_INTERNAL_COPY_TO_STAGE)
			uvm_page_mask_or(&block->read_duplicated_pages, &block->read_duplicated_pages, copy_mask);

		if (transfer_mode == BLOCK_TRANSFER_MODE_INTERNAL_COPY_FROM_STAGE)
			UVM_ASSERT(uvm_page_mask_subset(copy_mask, &block->read_duplicated_pages));

		// Any move operation implies that mappings have been removed from all
		// non-UVM-Lite GPUs
		if (transfer_mode == BLOCK_TRANSFER_MODE_INTERNAL_MOVE ||
				transfer_mode == BLOCK_TRANSFER_MODE_INTERNAL_MOVE_TO_STAGE) {

			uvm_page_mask_andnot(&block->maybe_mapped_pages, &block->maybe_mapped_pages, copy_mask);
		}

		// Record ReadDuplicate events here, after the residency bits have been
		// updated
		if (block_transfer_mode == UVM_VA_BLOCK_TRANSFER_MODE_COPY)
			rhacuvm_uvm_tools_record_read_duplicate(block, dst_id, region, copy_mask);

		// If we are migrating due to an eviction, set the GPU as evicted and
		// mark the evicted pages. If we are migrating away from the CPU this
		// means that those pages are not evicted.
		if (cause == UVM_MAKE_RESIDENT_CAUSE_EVICTION) {
			uvm_va_block_gpu_state_t *src_gpu_state = block_gpu_state_get(block, src_id);
			UVM_ASSERT(src_gpu_state);
			UVM_ASSERT(UVM_ID_IS_CPU(dst_id));

			uvm_page_mask_or(&src_gpu_state->evicted, &src_gpu_state->evicted, copy_mask);
			uvm_processor_mask_set(&block->evicted_gpus, src_id);
		}
		else if (UVM_ID_IS_GPU(dst_id) && uvm_processor_mask_test(&block->evicted_gpus, dst_id)) {
			uvm_va_block_gpu_state_t *dst_gpu_state;
			RHAC_ASSERT(false);
			dst_gpu_state = block_gpu_state_get(block, dst_id);
			UVM_ASSERT(dst_gpu_state);

			if (!uvm_page_mask_andnot(&dst_gpu_state->evicted, &dst_gpu_state->evicted, copy_mask))
				uvm_processor_mask_clear(&block->evicted_gpus, dst_id);
		}
	}

	return status;
}

static uvm_pte_bits_gpu_t get_gpu_pte_bit_index(uvm_prot_t prot)
{
	uvm_pte_bits_gpu_t pte_bit_index = UVM_PTE_BITS_GPU_MAX;

	if (prot == UVM_PROT_READ_WRITE_ATOMIC)
		pte_bit_index = UVM_PTE_BITS_GPU_ATOMIC;
	else if (prot == UVM_PROT_READ_WRITE)
		pte_bit_index = UVM_PTE_BITS_GPU_WRITE;
	else if (prot == UVM_PROT_READ_ONLY)
		pte_bit_index = UVM_PTE_BITS_GPU_READ;
	else
		UVM_ASSERT_MSG(false, "Invalid access permissions %s\n", uvm_prot_string(prot));

	return pte_bit_index;
}

static uvm_pte_bits_cpu_t get_cpu_pte_bit_index(uvm_prot_t prot)
{
	uvm_pte_bits_cpu_t pte_bit_index = UVM_PTE_BITS_CPU_MAX;

	// ATOMIC and WRITE are synonyms for the CPU
	if (prot == UVM_PROT_READ_WRITE_ATOMIC || prot == UVM_PROT_READ_WRITE)
		pte_bit_index = UVM_PTE_BITS_CPU_WRITE;
	else if (prot == UVM_PROT_READ_ONLY)
		pte_bit_index = UVM_PTE_BITS_CPU_READ;
	else
		UVM_ASSERT_MSG(false, "Invalid access permissions %s\n", uvm_prot_string(prot));

	return pte_bit_index;
}

static const uvm_page_mask_t *block_map_with_prot_mask_get(uvm_va_block_t *block,
		uvm_processor_id_t processor,
		uvm_prot_t prot)
{
	uvm_va_block_gpu_state_t *gpu_state;

	if (UVM_ID_IS_CPU(processor))
		return &block->cpu.pte_bits[get_cpu_pte_bit_index(prot)];

	gpu_state = block_gpu_state_get(block, processor);

	UVM_ASSERT(gpu_state);
	return &gpu_state->pte_bits[get_gpu_pte_bit_index(prot)];
}

static bool check_access_counters_dont_revoke(uvm_va_block_t *block,
		uvm_va_block_context_t *block_context,
		uvm_va_block_region_t region,
		const uvm_processor_mask_t *revoke_processors,
		const uvm_page_mask_t *revoke_page_mask,
		uvm_prot_t revoke_prot)
{
	uvm_processor_id_t id;
	for_each_id_in_mask(id, revoke_processors) {
		const uvm_page_mask_t *mapped_with_prot = block_map_with_prot_mask_get(block, id, revoke_prot);

		uvm_page_mask_and(&block_context->caller_page_mask, revoke_page_mask, mapped_with_prot);

		UVM_ASSERT(uvm_page_mask_region_weight(&block_context->caller_page_mask, region) == 0);
	}

	return true;
}

static bool block_region_might_read_duplicate(uvm_va_block_t *va_block,
		uvm_va_block_region_t region)
{
	uvm_va_range_t *va_range;
	uvm_va_space_t *va_space;

	va_range = va_block->va_range;
	va_space = va_range->va_space;

	if (!rhacuvm_uvm_va_space_can_read_duplicate(va_space, NULL))
		return false;

	if (va_range->read_duplication == UVM_READ_DUPLICATION_DISABLED)
		return false;

	if (va_range->read_duplication == UVM_READ_DUPLICATION_UNSET
			&& uvm_page_mask_region_weight(&va_block->read_duplicated_pages, region) == 0)
		return false;

	return true;
}

static NV_STATUS block_copy_resident_pages_mask(uvm_va_block_t *block,
		uvm_va_block_context_t *block_context,
		uvm_processor_id_t dst_id,
		const uvm_processor_mask_t *src_processor_mask,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *page_mask,
		const uvm_page_mask_t *prefetch_page_mask,
		block_transfer_mode_internal_t transfer_mode,
		NvU32 max_pages_to_copy,
		uvm_page_mask_t *migrated_pages,
		NvU32 *copied_pages_out,
		uvm_tracker_t *tracker_out)
{
	uvm_processor_id_t src_id;


	*copied_pages_out = 0;
	{
		NV_STATUS status;
		src_id = UVM_ID_CPU;
		NvU32 copied_pages_from_src;

		UVM_ASSERT(!uvm_id_equal(src_id, dst_id));

		status = block_copy_resident_pages_between(block,
				block_context,
				dst_id,
				src_id,
				region,
				page_mask,
				prefetch_page_mask,
				transfer_mode,
				migrated_pages,
				&copied_pages_from_src,
				tracker_out);
		*copied_pages_out += copied_pages_from_src;
		UVM_ASSERT(*copied_pages_out <= max_pages_to_copy);

		if (status != NV_OK)
			return status;

		// Break out once we copied max pages already
		if (*copied_pages_out == max_pages_to_copy) {
			return NV_OK;
		}
	}

	uvm_processor_mask_t search_mask;
	uvm_processor_mask_copy(&search_mask, src_processor_mask);
	uvm_processor_mask_clear(&search_mask, UVM_ID_CPU);
	for_each_closest_id(src_id, &search_mask, dst_id, block->va_range->va_space) {
		NV_STATUS status;
		NvU32 copied_pages_from_src;

		UVM_ASSERT(!uvm_id_equal(src_id, dst_id));

		status = block_copy_resident_pages_between(block,
				block_context,
				dst_id,
				src_id,
				region,
				page_mask,
				prefetch_page_mask,
				transfer_mode,
				migrated_pages,
				&copied_pages_from_src,
				tracker_out);
		*copied_pages_out += copied_pages_from_src;
		UVM_ASSERT(*copied_pages_out <= max_pages_to_copy);

		if (status != NV_OK)
			return status;

		// Break out once we copied max pages already
		if (*copied_pages_out == max_pages_to_copy)
			break;
	}

	return NV_OK;
}

static NV_STATUS block_copy_resident_pages(uvm_va_block_t *block,
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

	if (page_mask)
		uvm_page_mask_andnot(copy_page_mask, page_mask, resident_mask);
	else
		uvm_page_mask_complement(copy_page_mask, resident_mask);

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

		UVM_ASSERT(missing_pages_count >= pages_copied);
		missing_pages_count -= pages_copied;

		if (status != NV_OK)
			goto out;

		if (missing_pages_count == 0)
			goto out;

		if (pages_copied)
			uvm_page_mask_andnot(copy_page_mask, copy_page_mask, migrated_pages);
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

static NV_STATUS block_prep_read_duplicate_mapping(uvm_va_block_t *va_block,
                                                   uvm_va_block_context_t *va_block_context,
                                                   uvm_processor_id_t revoke_id,
                                                   uvm_va_block_region_t region,
                                                   const uvm_page_mask_t *page_mask)
{
    uvm_processor_mask_t unmap_processor_mask;
    uvm_processor_id_t unmap_id;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NV_STATUS status, tracker_status;

    // Unmap everybody except revoke_id
    uvm_processor_mask_andnot(&unmap_processor_mask, &va_block->mapped, &va_block->va_range->uvm_lite_gpus);
    uvm_processor_mask_clear(&unmap_processor_mask, revoke_id);

    for_each_id_in_mask(unmap_id, &unmap_processor_mask) {
        status = rhacuvm_uvm_va_block_unmap(va_block,
                                    va_block_context,
                                    unmap_id,
                                    region,
                                    page_mask,
                                    &local_tracker);
        if (status != NV_OK)
            goto out;
    }

    // Revoke WRITE/ATOMIC access permissions from the remaining mapped
    // processor.
    status = rhacuvm_uvm_va_block_revoke_prot(va_block,
                                      va_block_context,
                                      revoke_id,
                                      region,
                                      page_mask,
                                      UVM_PROT_READ_WRITE,
                                      &local_tracker);
    if (status != NV_OK)
        goto out;

out:
    tracker_status = rhacuvm_uvm_tracker_add_tracker_safe(&va_block->tracker, &local_tracker);
    rhacuvm_uvm_tracker_deinit(&local_tracker);
    return status == NV_OK ? tracker_status : status;
}

#include "rhac_nvidia_decl_cpu.h"
#include "rhac_nvidia_decl_gpu.h"

#endif //__RHAC_NVIDIA_DECL_H__
