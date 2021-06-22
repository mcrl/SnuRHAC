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

# include <linux/uaccess.h>

#include "rhac_nvidia_symbols.h"
#include "rhac_nvidia_helpers.h"
#include "rhac_nvidia_decl.h"
#include "rhac_nvidia_common.h"
#include "rhac_nvidia_pipeline.h"
#include "rhac_utils.h"

#include "nvidia-uvm/uvm8_va_range.h"
#include "nvidia-uvm/uvm8_va_space.h"
#include "nvidia-uvm/uvm8_hal.h"

struct rhac_comm;

extern uvm_service_block_context_t service_context_ro;
extern uvm_service_block_context_t service_context_rw;

#define rhac_uvm_range_group_for_each_migratability_in(iter, va_space, start, end) \
	for (rhacuvm_uvm_range_group_range_migratability_iter_first((va_space), (start), (end), (iter)); \
			(iter)->valid; \
			rhacuvm_uvm_range_group_range_migratability_iter_next((va_space), (iter), (end)))

typedef enum {
	UVM_MIGRATE_PASS_FIRST,
	UVM_MIGRATE_PASS_SECOND
} uvm_migrate_pass_t;

// This function determines if the VA range properties avoid the need to remove
// CPU mappings on UvmMigrate. Currently, it only checks whether
// read-duplication is enabled in the VA range. This is because, when migrating
// read-duplicated VA blocks, the source processor doesn't need to be unmapped
// (though it may need write access revoked).
static bool va_range_should_do_cpu_preunmap(uvm_va_range_t *va_range)
{
	return !uvm_va_range_is_read_duplicate(va_range);
}


// Function that determines if the VA block to be migrated contains pages with
// CPU mappings that don't need to be removed (see the comment above). In that
// case false is returned. Otherwise it returns true, and stores in the
// variable pointed by num_unmap_pages the number of pages that do need to
// remove their CPU mappings.
static bool va_block_should_do_cpu_preunmap(uvm_va_block_t *va_block,
		uvm_va_block_context_t *va_block_context, NvU64 start, NvU64 end,
		uvm_processor_id_t dest_id, NvU32 *num_unmap_pages) 
{
	const uvm_page_mask_t *mapped_pages_cpu;
	NvU32 num_cpu_unchanged_pages = 0;
	uvm_va_block_region_t region;

	*num_unmap_pages = 0;

	if (!va_block)
		return true;

	UVM_ASSERT(va_range_should_do_cpu_preunmap(va_block->va_range));

	region = uvm_va_block_region_from_start_end(va_block, max(start, va_block->start), min(end, va_block->end));

	uvm_mutex_lock(&va_block->lock);

	mapped_pages_cpu = rhacuvm_uvm_va_block_map_mask_get(va_block, UVM_ID_CPU);
	if (uvm_processor_mask_test(&va_block->resident, dest_id)) {
		const uvm_page_mask_t *resident_pages_dest = rhacuvm_uvm_va_block_resident_mask_get(va_block, dest_id);
		uvm_page_mask_t *do_not_unmap_pages = &va_block_context->scratch_page_mask;

		// TODO: Bug 1877578
		//
		// We assume that if pages are mapped on the CPU and not resident on
		// the destination, the pages will change residency so the CPU must be
		// unmapped. If we implement automatic read-duplication heuristics in
		// the future, we'll also need to check if the pages are being
		// read-duplicated.
		uvm_page_mask_and(do_not_unmap_pages, mapped_pages_cpu, resident_pages_dest);

		num_cpu_unchanged_pages = uvm_page_mask_region_weight(do_not_unmap_pages, region);
	}

	*num_unmap_pages = uvm_page_mask_region_weight(mapped_pages_cpu, region) - num_cpu_unchanged_pages;

	uvm_mutex_unlock(&va_block->lock);

	return num_cpu_unchanged_pages == 0;
}

static bool is_migration_single_block(
		uvm_va_range_t *first_va_range, NvU64 base, NvU64 length) 
{
	NvU64 end = base + length - 1;

	if (end > first_va_range->node.end)
		return false;

	return rhacuvm_uvm_va_range_block_index(first_va_range, base) == 
		rhacuvm_uvm_va_range_block_index(first_va_range, end);
}

static bool migration_should_do_cpu_preunmap(uvm_va_space_t *va_space,
		uvm_migrate_pass_t pass, bool is_single_block) 
{
	if (pass != UVM_MIGRATE_PASS_FIRST || is_single_block)
		return false;

	if (uvm_processor_mask_get_gpu_count(&va_space->has_nvlink[UVM_ID_CPU_VALUE]) == 0)
		return false;

	return true;
}

static uvm_prot_t __uvm_va_block_page_compute_highest_permission(uvm_va_block_t *va_block,
                                                        uvm_processor_id_t processor_id,
                                                        uvm_page_index_t page_index)
{
    uvm_va_range_t *va_range = va_block->va_range;
    uvm_va_space_t *va_space = va_range->va_space;
    uvm_processor_mask_t resident_processors;
    NvU32 resident_processors_count;

    if (uvm_processor_mask_test(&va_range->uvm_lite_gpus, processor_id))
        return UVM_PROT_READ_WRITE_ATOMIC;

    rhacuvm_uvm_va_block_page_resident_processors(va_block, page_index, &resident_processors);
    resident_processors_count = uvm_processor_mask_get_count(&resident_processors);

    if (resident_processors_count == 0) {
        return UVM_PROT_NONE;
    }
    else if (resident_processors_count > 1) {
        // If there are many copies, we can only map READ ONLY
        //
        // The block state doesn't track the mapping target (aperture) of each
        // individual PTE, just the permissions and where the data is resident.
        // If the data is resident in multiple places, then we have a problem
        // since we can't know where the PTE points. This means we won't know
        // what needs to be unmapped for cases like UvmUnregisterGpu and
        // UvmDisablePeerAccess.
        //
        // The simple way to solve this is to enforce that a read-duplication
        // mapping always points to local memory.
        if (uvm_processor_mask_test(&resident_processors, processor_id))
            return UVM_PROT_READ_ONLY;

        return UVM_PROT_NONE;
    }
    else {
        uvm_processor_id_t atomic_id;
        uvm_processor_id_t residency;
        uvm_processor_mask_t atomic_mappings;
        uvm_processor_mask_t write_mappings;

        // Search the id of the processor with the only resident copy
        residency = uvm_processor_mask_find_first_id(&resident_processors);
        UVM_ASSERT(UVM_ID_IS_VALID(residency));

        // If we cannot map the processor with the resident copy, exit
        if (!uvm_processor_mask_test(&va_space->accessible_from[uvm_id_value(residency)], processor_id))
            return UVM_PROT_NONE;

        // Fast path: if the page is not mapped anywhere else, it can be safely
        // mapped with RWA permission
        if (!uvm_page_mask_test(&va_block->maybe_mapped_pages, page_index))
            return UVM_PROT_READ_WRITE_ATOMIC;

        rhacuvm_uvm_va_block_page_authorized_processors(va_block, page_index, UVM_PROT_READ_WRITE_ATOMIC, &atomic_mappings);

        // Exclude processors with system-wide atomics disabled from atomic_mappings
        uvm_processor_mask_and(&atomic_mappings,
                               &atomic_mappings,
                               &va_space->system_wide_atomics_enabled_processors);

        // Exclude the processor for which the mapping protections are being computed
        uvm_processor_mask_clear(&atomic_mappings, processor_id);

        // If there is any processor with atomic mapping, check if it has native atomics to the processor
        // with the resident copy. If it does not, we can only map READ ONLY
        atomic_id = uvm_processor_mask_find_first_id(&atomic_mappings);
        if (UVM_ID_IS_VALID(atomic_id) &&
            !uvm_processor_mask_test(&va_space->has_native_atomics[uvm_id_value(residency)], atomic_id)) {
            return UVM_PROT_READ_ONLY;
        }

        rhacuvm_uvm_va_block_page_authorized_processors(va_block, page_index, UVM_PROT_READ_WRITE, &write_mappings);

        // Exclude the processor for which the mapping protections are being computed
        uvm_processor_mask_clear(&write_mappings, processor_id);

        // At this point, any processor with atomic mappings either has native atomics support to the
        // processor with the resident copy or has disabled system-wide atomics. If the requesting
        // processor has disabled system-wide atomics or has native atomics to that processor, we can
        // map with ATOMIC privileges. Likewise, if there are no other processors with WRITE or ATOMIC
        // mappings, we can map with ATOMIC privileges.
        if (!uvm_processor_mask_test(&va_space->system_wide_atomics_enabled_processors, processor_id) ||
            uvm_processor_mask_test(&va_space->has_native_atomics[uvm_id_value(residency)], processor_id) ||
            uvm_processor_mask_empty(&write_mappings)) {
            return UVM_PROT_READ_WRITE_ATOMIC;
        }

        return UVM_PROT_READ_WRITE;
    }
}

static void map_get_allowed_destinations(uvm_va_block_t *block,
                                         uvm_processor_id_t id,
                                         uvm_processor_mask_t *allowed_mask)
{
    uvm_va_range_t *va_range = block->va_range;
    uvm_va_space_t *va_space = va_range->va_space;

    if (uvm_processor_mask_test(&va_range->uvm_lite_gpus, id)) {
        // UVM-Lite can only map resident pages on the preferred location
        uvm_processor_mask_zero(allowed_mask);
        uvm_processor_mask_set(allowed_mask, va_range->preferred_location);
    }
    else if ((uvm_va_range_is_read_duplicate(va_range) || uvm_id_equal(va_range->preferred_location, id)) &&
             uvm_va_space_processor_has_memory(va_space, id)) {
        // When operating under read-duplication we should only map the local
        // processor to cause fault-and-duplicate of remote pages.
        //
        // The same holds when this processor is the preferred location: only
        // create local mappings to force remote pages to fault-and-migrate.
        uvm_processor_mask_zero(allowed_mask);
        uvm_processor_mask_set(allowed_mask, id);
    }
    else {
        // Common case: Just map wherever the memory happens to reside
        uvm_processor_mask_and(allowed_mask, &block->resident, &va_space->can_access[uvm_id_value(id)]);
        return;
    }

    // Clamp to resident and accessible processors
    uvm_processor_mask_and(allowed_mask, allowed_mask, &block->resident);
    uvm_processor_mask_and(allowed_mask, allowed_mask, &va_space->can_access[uvm_id_value(id)]);
}

static bool block_gpu_has_page_tables(uvm_va_block_t *block, uvm_gpu_t *gpu)
{
    uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, gpu->id);

    if (!gpu_state)
        return false;

    return gpu_state->page_table_range_4k.table  ||
           gpu_state->page_table_range_big.table ||
           gpu_state->page_table_range_2m.table;
}

// A helper to get a known-to-be-present GPU VA space given a VA block that's
// locked. In order to use this function, the caller must know that at least one
// of these conditions is true:
//
// 1) The VA space lock is held
// 2) The VA block has active page tables for the GPU
//
// If the VA space lock is held (#1), then the gpu_va_space obviously can't go
// away.
//
// On the eviction path, we don't have a lock on the VA space state. However,
// since remove_gpu_va_space walks each block to unmap the GPU and free GPU page
// tables before destroying the gpu_va_space, we're guaranteed that if this GPU
// has page tables (#2), the gpu_va_space can't go away while we're holding the
// block lock.
static uvm_gpu_va_space_t *uvm_va_block_get_gpu_va_space(uvm_va_block_t *va_block, uvm_gpu_t *gpu)
{
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_va_space_t *va_space;

    UVM_ASSERT(gpu);
    UVM_ASSERT(va_block->va_range);

    va_space = va_block->va_range->va_space;

    if (!block_gpu_has_page_tables(va_block, gpu))
        uvm_assert_rwsem_locked(&va_space->lock);

    UVM_ASSERT(uvm_processor_mask_test(&va_space->registered_gpu_va_spaces, gpu->id));

    gpu_va_space = va_space->gpu_va_spaces[uvm_id_gpu_index(gpu->id)];

    UVM_ASSERT(uvm_gpu_va_space_state(gpu_va_space) == UVM_GPU_VA_SPACE_STATE_ACTIVE);
    UVM_ASSERT(gpu_va_space->va_space == va_space);
    UVM_ASSERT(gpu_va_space->gpu == gpu);

    return gpu_va_space;
}

static bool block_gpu_supports_2m(uvm_va_block_t *block, uvm_gpu_t *gpu)
{
    uvm_gpu_va_space_t *gpu_va_space;

    if (uvm_va_block_size(block) < UVM_PAGE_SIZE_2M)
        return false;

    UVM_ASSERT(uvm_va_block_size(block) == UVM_PAGE_SIZE_2M);

    gpu_va_space = uvm_va_block_get_gpu_va_space(block, gpu);
    return uvm_mmu_page_size_supported(&gpu_va_space->page_tables, UVM_PAGE_SIZE_2M);
}

// We use block_gpu_get_processor_to_map to find the destination processor of a
// given GPU mapping. This function is called when the mapping is established to
// sanity check that the destination of the mapping matches the query.
static bool block_check_mapping_residency_region(uvm_va_block_t *block,
                                                 uvm_gpu_t *gpu,
                                                 uvm_processor_id_t mapping_dest,
                                                 uvm_va_block_region_t region,
                                                 const uvm_page_mask_t *page_mask)
{
    uvm_page_index_t page_index;
    for_each_va_block_page_in_region_mask(page_index, page_mask, region) {
        NvU64 va = uvm_va_block_cpu_page_address(block, page_index);
        uvm_processor_id_t proc_to_map = rhacuvm_block_gpu_get_processor_to_map(block, gpu, page_index);
        UVM_ASSERT_MSG(uvm_id_equal(mapping_dest, proc_to_map),
                       "VA 0x%llx on %s: mapping %s, supposed to map %s",
                       va,
                       gpu->name,
                       block_processor_name(block, mapping_dest),
                       block_processor_name(block, proc_to_map));
    }
    return true;
}

// Helper for block_gpu_compute_new_pte_state. For GPUs which swizzle, all
// GPUs mapping the same memory must use the same PTE size. This function
// returns true if all currently-mapped GPUs (aside from mapping_gpu itself) can
// all be promoted to big PTE mappings.
static bool mapped_gpus_can_map_big(uvm_va_block_t *block,
                                    uvm_gpu_t *mapping_gpu,
                                    uvm_va_block_region_t big_page_region)
{
    uvm_processor_mask_t mapped_procs;
    uvm_gpu_t *other_gpu;
    uvm_gpu_t *resident_gpu;
    uvm_pte_bits_gpu_t pte_bit;
    uvm_va_space_t *va_space = block->va_range->va_space;

    // GPUs without swizzling don't care what page size their peers use
    if (!mapping_gpu->big_page.swizzling)
        return true;

    resident_gpu = uvm_va_space_get_gpu(va_space,
                                        rhacuvm_block_gpu_get_processor_to_map(block, mapping_gpu, big_page_region.first));
    UVM_ASSERT(uvm_processor_mask_test(&va_space->accessible_from[uvm_id_value(resident_gpu->id)], mapping_gpu->id));

    // GPUs which don't swizzle can't have peer mappings to those which do.
    // We've also enforced that they all share the same big page size for a
    // given VA space, so we can use the big page regions interchangeably.
    UVM_ASSERT(resident_gpu->big_page.swizzling);

    uvm_processor_mask_and(&mapped_procs, &block->mapped, &va_space->accessible_from[uvm_id_value(resident_gpu->id)]);

    // The caller checks mapping_gpu, since its gpu_state permissions mask
    // hasn't been updated yet.
    uvm_processor_mask_clear(&mapped_procs, mapping_gpu->id);

    // Since UVM-Lite GPUs always map the preferred location and remain mapped
    // even if the memory is resident on a non-UVM-Lite GPU, we ignore UVM-Lite
    // GPUs when mapping non-UVM-Lite GPUs, and vice-versa.
    if (uvm_processor_mask_test(&block->va_range->uvm_lite_gpus, mapping_gpu->id))
        uvm_processor_mask_and(&mapped_procs, &mapped_procs, &block->va_range->uvm_lite_gpus);
    else
        uvm_processor_mask_andnot(&mapped_procs, &mapped_procs, &block->va_range->uvm_lite_gpus);

    // If each peer GPU has matching permissions for this entire region, then
    // they can also map as a swizzled big page. Otherwise, all GPUs must demote
    // to 4k. Note that the GPUs don't have to match permissions with each
    // other.
    for_each_va_space_gpu_in_mask(other_gpu, va_space, &mapped_procs) {
        uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, other_gpu->id);

        UVM_ASSERT(other_gpu->big_page.swizzling);

        for (pte_bit = UVM_PTE_BITS_GPU_ATOMIC; pte_bit >= 0; pte_bit--) {
            // If the highest permissions has a full region, then we match
            if (uvm_page_mask_region_full(&gpu_state->pte_bits[pte_bit], big_page_region)) {
                // Sanity check that all GPUs actually map the same memory
                UVM_ASSERT(block_check_mapping_residency_region(block,
                                                                other_gpu,
                                                                resident_gpu->id,
                                                                big_page_region,
                                                                &gpu_state->pte_bits[pte_bit]));
                break;
            }

            // If some pages are set, then we don't match and we can't map big
            if (!uvm_page_mask_region_empty(&gpu_state->pte_bits[pte_bit], big_page_region))
                return false;

            // Otherwise, try the next lower permissions. A fully-unmapped GPU
            // doesn't factor into the swizzling decisions, so we ignore those.
            if (pte_bit == 0)
                break;
        }
    }

    // All mapped peers can map as a big page
    return true;
}

// When PTE state is about to change (for example due to a map/unmap/revoke
// operation), this function decides how to split and merge the PTEs in response
// to that operation.
//
// The operation is described with the two page masks:
//
// - pages_changing indicates which pages will have their PTE mappings changed
//   on the GPU in some way as a result of the operation (for example, which
//   pages will actually have their mapping permissions upgraded).
//
// - page_mask_after indicates which pages on this GPU will have exactly the
//   same PTE attributes (permissions, residency) as pages_changing after the
//   operation is applied.
//
// PTEs are merged eagerly.
static void rhac_block_gpu_compute_new_pte_state(uvm_va_block_t *block,
                                            uvm_gpu_t *gpu,
                                            uvm_processor_id_t resident_id,
                                            const uvm_page_mask_t *pages_changing,
                                            const uvm_page_mask_t *page_mask_after,
                                            uvm_va_block_new_pte_state_t *new_pte_state)
{
    uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(block, gpu->id);
    uvm_va_block_region_t big_region_all, big_page_region, region;
    NvU32 big_page_size;
    uvm_page_index_t page_index;
    size_t big_page_index;
    DECLARE_BITMAP(big_ptes_not_covered, MAX_BIG_PAGES_PER_UVM_VA_BLOCK);
    bool can_make_new_big_ptes, region_full;

    memset(new_pte_state, 0, sizeof(*new_pte_state));
    new_pte_state->needs_4k = true;

    // TODO: Bug 1676485: Force a specific page size for perf testing

    if (gpu_state->force_4k_ptes)
        return;

    UVM_ASSERT(uvm_page_mask_subset(pages_changing, page_mask_after));

    // TODO: Bug 1766172: Use 2M sysmem pages on x86
    if (block_gpu_supports_2m(block, gpu) && !UVM_ID_IS_CPU(resident_id)) {
        // If all pages in the 2M mask have the same attributes after the
        // operation is applied, we can use a 2M PTE.
        if (uvm_page_mask_full(page_mask_after)) {
            new_pte_state->pte_is_2m = true;
            new_pte_state->needs_4k = false;
            return;
        }
    }

    // Find big PTEs with matching attributes

    // Can this block fit any big pages?
    big_page_size = rhacuvm_uvm_va_block_gpu_big_page_size(block, gpu);
    big_region_all = rhacuvm_uvm_va_block_big_page_region_all(block, big_page_size);
    if (big_region_all.first >= big_region_all.outer)
        return;

    new_pte_state->needs_4k = false;

    can_make_new_big_ptes = true;

    // Big pages can be used when mapping sysmem if PAGE_SIZE >= big_page_size
    // and the GPU supports it (Pascal+).
    if (UVM_ID_IS_CPU(resident_id) && (!gpu->can_map_sysmem_with_large_pages || PAGE_SIZE < big_page_size))
        can_make_new_big_ptes = false;

    // We must not fail during teardown: unmap (resident_id == UVM_ID_INVALID)
    // with no splits required. That means we should avoid allocating PTEs
    // which are only needed for merges.
    //
    // This only matters if we're merging to big PTEs. If we're merging to 2M,
    // then we must already have the 2M level (since it has to be allocated
    // before the lower levels).
    //
    // If pte_is_2m already and we don't have a big table, we're splitting so we
    // have to allocate.
    if (UVM_ID_IS_INVALID(resident_id) && !gpu_state->page_table_range_big.table && !gpu_state->pte_is_2m)
        can_make_new_big_ptes = false;

    for_each_va_block_page_in_region_mask(page_index, pages_changing, big_region_all) {
        big_page_index = rhacuvm_uvm_va_block_big_page_index(block, page_index, big_page_size);
        big_page_region = rhacuvm_uvm_va_block_big_page_region(block, big_page_index, big_page_size);

        __set_bit(big_page_index, new_pte_state->big_ptes_covered);

        region_full = uvm_page_mask_region_full(page_mask_after, big_page_region);

        // RHAC: force as mask is full always
        if (block->va_range->read_duplication == UVM_READ_DUPLICATION_ENABLED)
          region_full = true;

        if (region_full && UVM_ID_IS_INVALID(resident_id))
            __set_bit(big_page_index, new_pte_state->big_ptes_fully_unmapped);

        if (can_make_new_big_ptes && region_full) {
            if (gpu->big_page.swizzling) {
                // If we're fully unmapping, we don't care about the swizzle
                // format. Otherwise we have to check whether all mappings can
                // be promoted to a big PTE.
                if (UVM_ID_IS_INVALID(resident_id) || mapped_gpus_can_map_big(block, gpu, big_page_region))
                    __set_bit(big_page_index, new_pte_state->big_ptes);
            }
            else {
                __set_bit(big_page_index, new_pte_state->big_ptes);
            }
        }

        if (!test_bit(big_page_index, new_pte_state->big_ptes))
            new_pte_state->needs_4k = true;

        // Skip to the end of the region
        page_index = big_page_region.outer - 1;
    }

    if (!new_pte_state->needs_4k) {
        // All big page regions in pages_changing will be big PTEs. Now check if
        // there are any unaligned pages outside of big_region_all which are
        // changing.
        region = uvm_va_block_region(0, big_region_all.first);
        if (!uvm_page_mask_region_empty(pages_changing, region)) {
            new_pte_state->needs_4k = true;
        }
        else {
            region = uvm_va_block_region(big_region_all.outer, uvm_va_block_num_cpu_pages(block));
            if (!uvm_page_mask_region_empty(pages_changing, region))
                new_pte_state->needs_4k = true;
        }
    }

    // Now add in the PTEs which should be big but weren't covered by this
    // operation.
    //
    // Note that we can't assume that a given page table range has been
    // initialized if it's present here, since it could have been allocated by a
    // thread which had to restart its operation due to allocation retry.
    if (gpu_state->pte_is_2m || (block_gpu_supports_2m(block, gpu) && !gpu_state->page_table_range_2m.table)) {
        // We're splitting a 2M PTE so all of the uncovered big PTE regions will
        // become big PTEs which inherit the 2M permissions. If we haven't
        // allocated the 2M table yet, it will start as a 2M PTE until the lower
        // levels are allocated, so it's the same split case regardless of
        // whether this operation will need to retry a later allocation.
        bitmap_complement(big_ptes_not_covered, new_pte_state->big_ptes_covered, MAX_BIG_PAGES_PER_UVM_VA_BLOCK);
    }
    else if (!gpu_state->page_table_range_4k.table && !new_pte_state->needs_4k) {
        // If we don't have 4k PTEs and we won't be allocating them for this
        // operation, all of our PTEs need to be big.
        UVM_ASSERT(!bitmap_empty(new_pte_state->big_ptes, MAX_BIG_PAGES_PER_UVM_VA_BLOCK));
        bitmap_zero(big_ptes_not_covered, MAX_BIG_PAGES_PER_UVM_VA_BLOCK);
        bitmap_set(big_ptes_not_covered, 0, rhacuvm_uvm_va_block_num_big_pages(block, big_page_size));
    }
    else {
        // Otherwise, add in all of the currently-big PTEs which are unchanging.
        // They won't be written, but they need to be carried into the new
        // gpu_state->big_ptes when it's updated.
        bitmap_andnot(big_ptes_not_covered,
                      gpu_state->big_ptes,
                      new_pte_state->big_ptes_covered,
                      MAX_BIG_PAGES_PER_UVM_VA_BLOCK);
    }

    bitmap_or(new_pte_state->big_ptes, new_pte_state->big_ptes, big_ptes_not_covered, MAX_BIG_PAGES_PER_UVM_VA_BLOCK);
}

// Maps the GPU to the given pages which are resident on resident_id.
// map_page_mask is an in/out parameter: the pages which are mapped
// to resident_id are removed from the mask before returning.
//
// Caller must ensure that:
// -  Pages in map_page_mask must not be set in the corresponding pte_bits mask
// for the requested protection on the mapping GPU.
static NV_STATUS rhac_block_map_gpu_to(uvm_va_block_t *va_block,
                                  uvm_va_block_context_t *block_context,
                                  uvm_gpu_t *gpu,
                                  uvm_processor_id_t resident_id,
                                  uvm_page_mask_t *map_page_mask,
                                  uvm_prot_t new_prot,
                                  uvm_tracker_t *out_tracker)
{
    uvm_va_block_gpu_state_t *gpu_state = block_gpu_state_get(va_block, gpu->id);
    uvm_va_range_t *va_range = va_block->va_range;
    uvm_va_space_t *va_space = va_range->va_space;
    uvm_push_t push;
    NV_STATUS status;
    uvm_page_mask_t *pages_to_map = &block_context->mapping.page_mask;
    const uvm_page_mask_t *resident_mask = rhacuvm_uvm_va_block_resident_mask_get(va_block, resident_id);
    uvm_pte_bits_gpu_t pte_bit;
    uvm_pte_bits_gpu_t prot_pte_bit = get_gpu_pte_bit_index(new_prot);
    uvm_va_block_new_pte_state_t *new_pte_state = &block_context->mapping.new_pte_state;
    block_pte_op_t pte_op;

    UVM_ASSERT(map_page_mask);
    UVM_ASSERT(uvm_processor_mask_test(&va_space->accessible_from[uvm_id_value(resident_id)], gpu->id));

    if (uvm_processor_mask_test(&va_range->uvm_lite_gpus, gpu->id))
        UVM_ASSERT(uvm_id_equal(resident_id, va_range->preferred_location));

    UVM_ASSERT(!uvm_page_mask_and(&block_context->scratch_page_mask,
                                  map_page_mask,
                                  &gpu_state->pte_bits[prot_pte_bit]));

    // The pages which will actually change are those in the input page mask
    // which are resident on the target.
    if (!uvm_page_mask_and(pages_to_map, map_page_mask, resident_mask))
        return NV_OK;

    // For PTE merge/split computation, compute all resident pages which will
    // have exactly new_prot after performing the mapping.
    uvm_page_mask_or(&block_context->scratch_page_mask, &gpu_state->pte_bits[prot_pte_bit], pages_to_map);
    if (prot_pte_bit < UVM_PTE_BITS_GPU_ATOMIC) {
        uvm_page_mask_andnot(&block_context->scratch_page_mask,
                             &block_context->scratch_page_mask,
                             &gpu_state->pte_bits[prot_pte_bit + 1]);
    }
    uvm_page_mask_and(&block_context->scratch_page_mask, &block_context->scratch_page_mask, resident_mask);

    rhac_block_gpu_compute_new_pte_state(va_block,
                                    gpu,
                                    resident_id,
                                    pages_to_map,
                                    &block_context->scratch_page_mask,
                                    new_pte_state);

    status = rhacuvm_block_alloc_ptes_new_state(va_block, gpu, new_pte_state, out_tracker);
    if (status != NV_OK)
        return status;

    if (gpu->big_page.swizzling && UVM_ID_IS_GPU(resident_id)) {
        status = rhacuvm_block_gpu_change_swizzling_map(va_block,
                                                block_context,
                                                uvm_va_space_get_gpu(va_space, resident_id),
                                                gpu,
                                                out_tracker);
        if (status != NV_OK)
            return status;
    }

    status = uvm_push_begin_acquire(gpu->channel_manager,
                                    UVM_CHANNEL_TYPE_MEMOPS,
                                    &va_block->tracker,
                                    &push,
                                    "Mapping pages in block [0x%llx, 0x%llx) as %s",
                                    va_block->start,
                                    va_block->end + 1,
                                    rhacuvm_uvm_prot_string(new_prot));
    if (status != NV_OK)
        return status;

    pte_op = BLOCK_PTE_OP_MAP;
    if (new_pte_state->pte_is_2m) {
        // We're either modifying permissions of a pre-existing 2M PTE, or all
        // permissions match so we can merge to a new 2M PTE.
        rhacuvm_block_gpu_map_to_2m(va_block, block_context, gpu, resident_id, new_prot, &push, pte_op);
    }
    else if (gpu_state->pte_is_2m) {
        // Permissions on a subset of the existing 2M PTE are being upgraded, so
        // we have to split it into the appropriate mix of big and 4k PTEs.
        rhacuvm_block_gpu_map_split_2m(va_block, block_context, gpu, resident_id, pages_to_map, new_prot, &push, pte_op);
    }
    else {
        // We're upgrading permissions on some pre-existing mix of big and 4K
        // PTEs into some other mix of big and 4K PTEs.
        rhacuvm_block_gpu_map_big_and_4k(va_block, block_context, gpu, resident_id, pages_to_map, new_prot, &push, pte_op);
    }

    // If we are mapping remotely, record the event
    if (va_space->tools.enabled && !uvm_id_equal(resident_id, gpu->id)) {
        uvm_va_block_region_t subregion, region = uvm_va_block_region_from_block(va_block);

        UVM_ASSERT(block_context->mapping.cause != UvmEventMapRemoteCauseInvalid);

        for_each_va_block_subregion_in_mask(subregion, pages_to_map, region) {
            rhacuvm_uvm_tools_record_map_remote(va_block,
                                        &push,
                                        gpu->id,
                                        resident_id,
                                        uvm_va_block_region_start(va_block, subregion),
                                        uvm_va_block_region_size(subregion),
                                        block_context->mapping.cause);
        }
    }

    rhacuvm_uvm_push_end(&push);

    // Update GPU mapping state
    for (pte_bit = 0; pte_bit <= prot_pte_bit; pte_bit++)
        uvm_page_mask_or(&gpu_state->pte_bits[pte_bit], &gpu_state->pte_bits[pte_bit], pages_to_map);

    uvm_processor_mask_set(&va_block->mapped, gpu->id);

    // If we are mapping a UVM-Lite GPU do not update maybe_mapped_pages
    if (!uvm_processor_mask_test(&va_range->uvm_lite_gpus, gpu->id))
        uvm_page_mask_or(&va_block->maybe_mapped_pages, &va_block->maybe_mapped_pages, pages_to_map);

    // Remove all pages resident on this processor from the input mask, which
    // were newly-mapped.
    uvm_page_mask_andnot(map_page_mask, map_page_mask, pages_to_map);

    return rhacuvm_uvm_tracker_add_push_safe(out_tracker, &push);
}

NV_STATUS __rhac_uvm_va_block_map(uvm_va_block_t *va_block,
                           uvm_va_block_context_t *va_block_context,
                           uvm_processor_id_t id,
                           uvm_va_block_region_t region,
                           const uvm_page_mask_t *map_page_mask,
                           uvm_prot_t new_prot,
                           UvmEventMapRemoteCause cause,
                           uvm_tracker_t *out_tracker) {
	uvm_va_range_t *va_range = va_block->va_range;
	uvm_va_space_t *va_space;
	uvm_gpu_t *gpu = NULL;
	uvm_processor_mask_t allowed_destinations;
	uvm_processor_id_t resident_id;
	const uvm_page_mask_t *pte_mask;
	uvm_page_mask_t *running_page_mask = &va_block_context->mapping.map_running_page_mask;
	NV_STATUS status;

	va_block_context->mapping.cause = cause;

	UVM_ASSERT(new_prot != UVM_PROT_NONE);
	UVM_ASSERT(new_prot < UVM_PROT_MAX);
	uvm_assert_mutex_locked(&va_block->lock);
	UVM_ASSERT(va_range);

	va_space = va_range->va_space;

	// Mapping is not supported on the eviction path that doesn't hold the VA
	// space lock.
	uvm_assert_rwsem_locked(&va_space->lock);

	uvm_va_block_gpu_state_t *gpu_state;
	uvm_pte_bits_gpu_t prot_pte_bit;

	gpu = uvm_va_space_get_gpu(va_space, id);

	// Although this GPU UUID is registered in the VA space, it might not have a
	// GPU VA space registered.
	if (!uvm_gpu_va_space_get(va_space, gpu))
		return NV_OK;

	gpu_state = block_gpu_state_get_alloc(va_block, gpu);
	if (!gpu_state)
		return NV_ERR_NO_MEMORY;

	prot_pte_bit = get_gpu_pte_bit_index(new_prot);
	pte_mask = &gpu_state->pte_bits[prot_pte_bit];

	uvm_page_mask_init_from_region(running_page_mask, region, map_page_mask);

	if (!uvm_page_mask_andnot(running_page_mask, running_page_mask, pte_mask))
		return NV_OK;

	// Map per resident location so we can more easily detect physically-
	// contiguous mappings.
	map_get_allowed_destinations(va_block, id, &allowed_destinations);

	for_each_closest_id(resident_id, &allowed_destinations, id, va_space) {
		status = rhac_block_map_gpu_to(va_block,
				va_block_context,
				gpu,
				resident_id,
				running_page_mask,
				new_prot,
				out_tracker);
		if (status != NV_OK)
			return status;

		// If we've mapped all requested pages, we're done
		if (uvm_page_mask_region_empty(running_page_mask, region))
			break;
	}

	return NV_OK;
}

NV_STATUS rhac_uvm_va_block_map(uvm_va_block_t *va_block,
                           uvm_va_block_context_t *va_block_context,
                           uvm_processor_id_t id,
                           uvm_va_block_region_t region,
                           const uvm_page_mask_t *map_page_mask,
                           uvm_prot_t new_prot,
                           UvmEventMapRemoteCause cause,
                           uvm_tracker_t *out_tracker) {
  NV_STATUS status;
  if (UVM_ID_IS_CPU(id)) {
    status = rhacuvm_uvm_va_block_map(va_block,
        va_block_context,
        id,
        region,
        map_page_mask,
        new_prot,
        cause,
        out_tracker);
  }
  else {
//    status = __rhac_uvm_va_block_map(va_block,
    status = rhacuvm_uvm_va_block_map(va_block,
        va_block_context,
        id,
        region,
        map_page_mask,
        new_prot,
        cause,
        out_tracker);
  }
  return status;
}

static NV_STATUS block_migrate_map_mapped_pages(
		uvm_va_block_t *va_block,
		uvm_va_block_context_t *va_block_context,
		uvm_va_block_region_t region,
    uvm_page_mask_t *page_mask,
		uvm_processor_id_t dest_id) 
{
	uvm_prot_t prot;
	uvm_page_index_t page_index;
	NV_STATUS status = NV_OK;

	const uvm_page_mask_t *pages_mapped_on_destination = rhacuvm_uvm_va_block_map_mask_get(va_block, dest_id);

	for (prot = UVM_PROT_READ_ONLY; prot <= UVM_PROT_READ_WRITE_ATOMIC; ++prot)
		va_block_context->mask_by_prot[prot - 1].count = 0;


	// Only map those pages that are not already mapped on destination
	for_each_va_block_unset_page_in_region_mask(
			page_index, pages_mapped_on_destination, region) {
    if (!uvm_page_mask_test(page_mask, page_index))
      continue;

		prot = __uvm_va_block_page_compute_highest_permission(
				va_block, dest_id, page_index);
		UVM_ASSERT(prot != UVM_PROT_NONE);

		if (va_block_context->mask_by_prot[prot - 1].count++ == 0)
			uvm_page_mask_zero(&va_block_context->mask_by_prot[prot - 1].page_mask);

		uvm_page_mask_set(&va_block_context->mask_by_prot[prot - 1].page_mask, page_index);
	}

	for (prot = UVM_PROT_READ_ONLY; prot <= UVM_PROT_READ_WRITE_ATOMIC; ++prot) {
		if (va_block_context->mask_by_prot[prot - 1].count == 0)
			continue;


		// We pass UvmEventMapRemoteCauseInvalid since the destination processor
		// of a migration will never be mapped remotely
    status = rhac_uvm_va_block_map(va_block,
        va_block_context,
        dest_id,
        region,
        &va_block_context->mask_by_prot[prot - 1].page_mask,
        prot,
        UvmEventMapRemoteCauseInvalid,
        &va_block->tracker);
    if (status != NV_OK)
      break;

		// Whoever added the other mapping(s) should have already added
		// SetAccessedBy processors
	}

	return status;
}

static NV_STATUS block_migrate_map_unmapped_pages(
		uvm_va_block_t *va_block, 
		uvm_va_block_context_t *va_block_context, uvm_va_block_region_t region,
		uvm_page_mask_t *page_mask, bool is_read, uvm_processor_id_t dest_id) 
{
	uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
	NV_STATUS status = NV_OK;
	NV_STATUS tracker_status;

	if (is_read) {
		goto out;
	}

	if (page_mask)
		uvm_page_mask_copy(&va_block_context->caller_page_mask, page_mask);
	else
		uvm_page_mask_region_fill(&va_block_context->caller_page_mask, region);
	// RELAXED: we cannot use maybe_mapped_pages flag because pages can be
	// mapped to multiple processors
	//  // Save the mask of unmapped pages because it will change after the
	//  // first map operation
	//  uvm_page_mask_complement(&va_block_context->caller_page_mask, &va_block->maybe_mapped_pages);
	// Only map those pages that are not mapped anywhere else (likely due
	// to a first touch or a migration). We pass
	// UvmEventMapRemoteCauseInvalid since the destination processor of a
	// migration will never be mapped remotely.

  status = rhac_uvm_va_block_map(va_block,
      va_block_context,
      dest_id,
      region,
      &va_block_context->caller_page_mask,
      UVM_PROT_READ_WRITE_ATOMIC,
      UvmEventMapRemoteCauseInvalid,
      &local_tracker);
  if (status != NV_OK)
    goto out;

	// Add mappings for AccessedBy processors
	//
	// No mappings within this call will operate on dest_id, so we don't
	// need to acquire the map operation above.
	status = rhacuvm_uvm_va_block_add_mappings_after_migration(va_block,
			va_block_context, dest_id, dest_id, region,
			&va_block_context->caller_page_mask, UVM_PROT_READ_WRITE_ATOMIC, NULL);
out:
	tracker_status = rhacuvm_uvm_tracker_add_tracker_safe(&va_block->tracker, &local_tracker);
	rhacuvm_uvm_tracker_deinit(&local_tracker);
	return status == NV_OK ? tracker_status : status;
}

// Pages that are not mapped anywhere can be safely mapped with RWA permission.
// The rest of pages need to individually compute the maximum permission that
// does not require a revocation.
static NV_STATUS block_migrate_add_mappings(
		uvm_va_block_t *va_block,
		uvm_va_block_context_t *va_block_context, uvm_va_block_region_t region,
		uvm_page_mask_t *page_mask, bool is_read, uvm_processor_id_t dest_id) 
{
	NV_STATUS status;
	status = block_migrate_map_unmapped_pages(va_block,
			va_block_context, region, page_mask, is_read, dest_id);
	if (status != NV_OK)
		return status;
	status = block_migrate_map_mapped_pages(va_block, 
			va_block_context, region, page_mask, dest_id);
	return status;
}

static void preunmap_multi_block(uvm_va_range_t *va_range,
		uvm_va_block_context_t *va_block_context,
		NvU64 start,
		NvU64 end,
		uvm_processor_id_t dest_id) 
{
	size_t i;
	const size_t first_block_index = rhacuvm_uvm_va_range_block_index(va_range, start);
	const size_t last_block_index = rhacuvm_uvm_va_range_block_index(va_range, end);
	NvU32 num_unmap_pages = 0;

	UVM_ASSERT(start >= va_range->node.start);
	UVM_ASSERT(end  <= va_range->node.end);
	UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_MANAGED);
	uvm_assert_rwsem_locked(&va_range->va_space->lock);

	UVM_ASSERT(rhacuvm_uvm_range_group_all_migratable(va_range->va_space, start, end));

	for (i = first_block_index; i <= last_block_index; i++) {
		NvU32 num_block_unmap_pages;

		if (!va_block_should_do_cpu_preunmap(uvm_va_range_block(va_range, i),
					va_block_context,
					start,
					end,
					dest_id,
					&num_block_unmap_pages)) {
			return;
		}

		num_unmap_pages += num_block_unmap_pages;
	}

	if (num_unmap_pages > 0)
		unmap_mapping_range(&va_range->va_space->mapping, start, end - start + 1, 1);
}

static NV_STATUS rhac_uvm_va_block_migrate(
		struct rhac_comm *comm,
		uvm_va_block_t *va_block,
		uvm_va_block_retry_t *va_block_retry,
		uvm_va_block_context_t *va_block_context,
		uvm_va_block_region_t region,
    uvm_page_mask_t *page_mask,
		uvm_processor_id_t dest_id,
		uvm_migrate_mode_t mode)
{
	NV_STATUS status;
	uvm_va_range_t *va_range = va_block->va_range;

	uvm_assert_mutex_locked(&va_block->lock);

	if (uvm_va_range_is_read_duplicate(va_range)) {
		status = rhac_uvm_va_block_make_resident_read_duplicate_global(
				comm,
				dest_id,
				va_block,
				va_block_retry,
				va_block_context,
				dest_id,
				region,
				page_mask,
				NULL,
				UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE,
				&service_context_ro
				);
	} else {
		status = rhac_uvm_va_block_make_resident_global(
				comm,
				dest_id,
				va_block,
				va_block_retry,
				va_block_context,
				dest_id,
				region,
				page_mask,
				NULL,
				UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE,
				&service_context_rw
				);
	}
	return status;
}

NV_STATUS rhac_uvm_va_block_migrate_global(
		struct rhac_comm *comm,
		uvm_va_block_t *va_block,
		uvm_va_block_retry_t *va_block_retry,
		uvm_va_block_context_t *va_block_context,
		uvm_va_block_region_t region,
    uvm_page_mask_t *page_mask,
		uvm_processor_id_t dest_id,
		uvm_migrate_mode_t mode)
{
	NV_STATUS status;

	do {
		status = rhac_uvm_va_block_migrate(comm,
				va_block,
				va_block_retry,
				va_block_context,
				region,
        page_mask,
				dest_id,
				mode);
	} while (status == NV_ERR_MORE_PROCESSING_REQUIRED);
	RHAC_ASSERT(status == NV_OK);
	return status;
}

int rhac_uvm_va_block_migrate_wrapup(
		struct rhac_comm *comm,
		uvm_va_block_t *va_block,
		uvm_va_block_context_t *va_block_context,
		uvm_va_block_region_t region,
    uvm_page_mask_t *page_mask,
		uvm_processor_id_t dest_id,
		uvm_migrate_mode_t mode)
{
	NV_STATUS status = NV_OK;

	uvm_assert_mutex_locked(&va_block->lock);
	if (mode == UVM_MIGRATE_MODE_MAKE_RESIDENT_AND_MAP) {
		if (UVM_ID_IS_CPU(dest_id)) {
			struct vm_area_struct *vma = uvm_va_range_vma(va_block->va_range);
			RHAC_ASSERT(vma);
			va_block_context->mm = vma->vm_mm;
			//FIXME
			//uvm_down_read_mmap_sem(&va_block_context->mm->mmap_sem);
		}
		// block_migrate_add_mappings will acquire the work from the above
		// make_resident call and update the VA block tracker.
		status = block_migrate_add_mappings(va_block,
				va_block_context,
				region,
				page_mask,
				uvm_va_range_is_read_duplicate(va_block->va_range),
				dest_id);
		if (UVM_ID_IS_CPU(dest_id)) {
			//uvm_up_read_mmap_sem(&va_block_context->mm->mmap_sem);
		}

	} 

	return status;
}

NV_STATUS rhac_uvm_va_range_migrate_multi_block(
		struct rhac_comm *pa,
		uvm_va_range_t *va_range,
		NvU64 start,
		NvU64 end,
		uvm_processor_id_t dest_id,
		uvm_migrate_mode_t mode,
		uvm_tracker_t *out_tracker,
		uvm_mutex_t *lock
		) 
{
	size_t i;
	const size_t first_block_index = rhacuvm_uvm_va_range_block_index(va_range, start);
	const size_t last_block_index = rhacuvm_uvm_va_range_block_index(va_range, end);

	UVM_ASSERT(start >= va_range->node.start);
	UVM_ASSERT(end  <= va_range->node.end);
	UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_MANAGED);
	uvm_assert_rwsem_locked(&va_range->va_space->lock);

	UVM_ASSERT(rhacuvm_uvm_range_group_all_migratable(va_range->va_space, start, end));

	// Iterate over blocks, populating them if necessary
	for (i = first_block_index; i <= last_block_index; i++) {
		uvm_va_block_region_t region;
		uvm_va_block_t *va_block = NULL;
    uvm_page_mask_t page_mask;
		NV_STATUS status = rhacuvm_uvm_va_range_block_create(va_range, i, &va_block);
		RHAC_ASSERT(va_block);
		if (status != NV_OK)
			return status;

		region = uvm_va_block_region_from_start_end(va_block,
				max(start, va_block->start),
				min(end, va_block->end));

    uvm_page_mask_init_from_region(&page_mask, region, NULL);

		status = rhac_nvidia_pipeline_prefetch(
				pa,
				va_block,
				region,
        &page_mask,
				dest_id,
				mode,
				out_tracker,
				lock);
		RHAC_ASSERT(status == NV_OK);
	}

	return NV_OK;
}

static NV_STATUS rhac_uvm_va_range_migrate(
		struct rhac_comm *pa,
		uvm_va_range_t *va_range,
		NvU64 start,
		NvU64 end,
		uvm_processor_id_t dest_id,
		uvm_migrate_mode_t mode,
		bool should_do_cpu_preunmap,
		uvm_tracker_t *out_tracker,
		uvm_mutex_t *lock
		) 
{
	NvU64 preunmap_range_start = start;

	should_do_cpu_preunmap = should_do_cpu_preunmap && va_range_should_do_cpu_preunmap(va_range);

	// Divide migrations into groups of contiguous VA blocks. This is to trigger
	// CPU unmaps for that region before the migration starts.
	while (preunmap_range_start < end) {
		NV_STATUS status;
		NvU64 preunmap_range_end;

		if (should_do_cpu_preunmap) {
			uvm_va_block_context_t *va_block_context = rhacuvm_uvm_va_block_context_alloc();

			preunmap_range_end = UVM_ALIGN_UP(preunmap_range_start + 1, UVM_VA_BLOCK_SIZE << 2);
			preunmap_range_end = min(preunmap_range_end - 1, end);
			preunmap_multi_block(va_range,
					va_block_context,
					preunmap_range_start,
					preunmap_range_end,
					dest_id);
			rhacuvm_uvm_va_block_context_free(va_block_context);
		}
		else {
			preunmap_range_end = end;
		}

		status = rhac_uvm_va_range_migrate_multi_block(
				pa,
				va_range,
				preunmap_range_start,
				preunmap_range_end,
				dest_id,
				mode,
				out_tracker,
				lock);
		if (status != NV_OK)
			return status;

		preunmap_range_start = preunmap_range_end + 1;
	}

	return NV_OK;
}

static NV_STATUS rhac_uvm_migrate_ranges(
		struct rhac_comm *pa,
		uvm_va_space_t *va_space,
		uvm_va_range_t *first_va_range,
		NvU64 base,
		NvU64 length,
		uvm_processor_id_t dest_id,
		uvm_migrate_mode_t mode,
		bool should_do_cpu_preunmap,
		uvm_tracker_t *out_tracker,
		uvm_mutex_t *lock
		) 
{
	uvm_va_range_t *va_range, *va_range_last;
	NvU64 end = base + length - 1;
	NV_STATUS status = NV_OK;
	bool skipped_migrate = false;

	UVM_ASSERT(first_va_range == rhacuvm_uvm_va_space_iter_first(va_space, base, base));

	va_range_last = NULL;
	uvm_for_each_va_range_in_contig_from(va_range, va_space, first_va_range, end) {
		uvm_range_group_range_iter_t iter;
		va_range_last = va_range;

		// Only managed ranges can be migrated
		if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED) {
			status = NV_ERR_INVALID_ADDRESS;
			break;
		}

		// For UVM-Lite GPUs, the CUDA driver may suballocate a single va_range
		// into many range groups.  For this reason, we iterate over each va_range first
		// then through the range groups within.
		rhac_uvm_range_group_for_each_migratability_in(&iter,
				va_space,
				max(base, va_range->node.start),
				min(end, va_range->node.end)) {
			// Skip non-migratable VA ranges
			if (!iter.migratable) {
				// Only return NV_WARN_MORE_PROCESSING_REQUIRED if the pages aren't
				// already resident at dest_id.
				if (!uvm_id_equal(va_range->preferred_location, dest_id))
					skipped_migrate = true;
			}
			else if (uvm_processor_mask_test(&va_range->uvm_lite_gpus, dest_id) &&
					!uvm_id_equal(dest_id, va_range->preferred_location)) {
				// Don't migrate to a non-faultable GPU that is in UVM-Lite mode,
				// unless it's the preferred location
				status = NV_ERR_INVALID_DEVICE;
				break;
			}
			else {
				status = rhac_uvm_va_range_migrate(pa,
						va_range,
						iter.start,
						iter.end,
						dest_id,
						mode,
						should_do_cpu_preunmap,
						out_tracker,
						lock
						);
				if (status != NV_OK)
					break;
			}
		}
	}


	if (status != NV_OK)
		return status;

	// Check that we were able to iterate over the entire range without any gaps
	if (!va_range_last || va_range_last->node.end < end)
		return NV_ERR_INVALID_ADDRESS;

	if (skipped_migrate)
		return NV_WARN_MORE_PROCESSING_REQUIRED;

	return NV_OK;
}

NV_STATUS rhac_uvm_migrate(
		struct rhac_comm *pa,
		uvm_va_space_t *va_space,
		NvU64 base,
		NvU64 length,
		uvm_processor_id_t dest_id,
		NvU32 migrate_flags,
		uvm_tracker_t *out_tracker,
		uvm_mutex_t *lock)
{
	NV_STATUS status = NV_OK;
	uvm_va_range_t *first_va_range = rhacuvm_uvm_va_space_iter_first(va_space, base, base);
	bool do_mappings;
	bool do_two_passes;
	bool is_single_block;
	bool should_do_cpu_preunmap;

	uvm_assert_mmap_sem_locked(&current->mm->mmap_sem);
	uvm_assert_rwsem_locked(&va_space->lock);

	if (!first_va_range || first_va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
		return NV_ERR_INVALID_ADDRESS;

	// If the GPU has its memory disabled, just skip the migration and let
	// faults take care of things.
	if (!uvm_va_space_processor_has_memory(va_space, dest_id))
		return NV_OK;


	// We perform two passes (unless the migration only covers a single VA
	// block or UVM_MIGRATE_FLAG_SKIP_CPU_MAP is passed). This helps in the
	// following scenarios:
	//
	// - Migrations that add CPU mappings, since they are synchronous operations
	// that delay the migration of the next VA blocks.
	// - Concurrent migrations. This is due to our current channel selection
	// logic that doesn't prevent false dependencies between independent
	// operations. For example, removal of mappings for outgoing transfers are
	// delayed by the mappings added by incoming transfers.
	// TODO: Bug 1764953: Re-evaluate the two-pass logic when channel selection
	// is overhauled.
	//
	// The two passes are as follows:
	//
	// 1- Transfer all VA blocks (do not add mappings)
	// 2- Go block by block reexecuting the transfer (in case someone moved it
	// since the first pass), and adding the mappings.
	is_single_block = is_migration_single_block(first_va_range, base, length);
	do_mappings = UVM_ID_IS_GPU(dest_id) || !(migrate_flags & UVM_MIGRATE_FLAG_SKIP_CPU_MAP);
	do_two_passes = do_mappings && !is_single_block;

	do_two_passes = false;

	if (do_two_passes) {
		should_do_cpu_preunmap = migration_should_do_cpu_preunmap(va_space, UVM_MIGRATE_PASS_FIRST, is_single_block);

		status = rhac_uvm_migrate_ranges(
				pa,
				va_space,
				first_va_range,
				base,
				length,
				dest_id,
				UVM_MIGRATE_MODE_MAKE_RESIDENT,
				should_do_cpu_preunmap,
				out_tracker,
				lock
				);
	}

	if (status == NV_OK) {
		uvm_migrate_mode_t mode = do_mappings? UVM_MIGRATE_MODE_MAKE_RESIDENT_AND_MAP:
			UVM_MIGRATE_MODE_MAKE_RESIDENT;
		uvm_migrate_pass_t pass = do_two_passes? UVM_MIGRATE_PASS_SECOND:
			UVM_MIGRATE_PASS_FIRST;
		should_do_cpu_preunmap = migration_should_do_cpu_preunmap(va_space, pass, is_single_block);

		status = rhac_uvm_migrate_ranges(
				pa,
				va_space,
				first_va_range,
				base,
				length,
				dest_id,
				mode,
				should_do_cpu_preunmap,
				out_tracker,
				lock);
	}

	return status;
}
