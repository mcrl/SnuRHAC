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

#include <linux/kallsyms.h>
#include "rhac_nvidia_symbols.h"
#include "rhac_utils.h"

#define __fnname(s) rhacuvm_ ## s
#define __ptrname(s) s ## _p
#define __quote(s) #s

#define load_function(s) \
	do { \
		__fnname(s) = (typeof(__fnname(s))) kallsyms_lookup_name(__quote(s)); \
		if (IS_ERR_OR_NULL((__fnname(s)))) { \
			RHAC_LOG("Cannot find symbol: %s (%llx)", __quote(s), (uint64_t)__fnname(s)); \
			return -EINVAL; \
		} \
	} while (0)

#define load_ptr(s) \
	do { \
		__ptrname(s) = (typeof(__ptrname(s))) kallsyms_lookup_name(__quote(s)); \
		if (IS_ERR_OR_NULL((__ptrname(s)))) { \
			RHAC_LOG("Cannot find symbol: %s (%llx)", __quote(s), (uint64_t)__ptrname(s)); \
			return -EINVAL; \
		} \
	} while (0)

struct list_head* slab_caches_p;

static struct kmem_cache **g_uvm_va_block_gpu_state_cache_p;
static struct kmem_cache **g_uvm_page_mask_cache_p;
struct kmem_cache *gg_uvm_va_block_gpu_state_cache;
struct kmem_cache *gg_uvm_page_mask_cache;


//
// pointers to nvidia 'variables'
// 
uvm_global_t* g_uvm_global_p;
int *uvm_enable_builtin_tests_p;
int *uvm_enable_debug_procfs_p;
unsigned* uvm_perf_fault_max_batches_per_service_p;
unsigned* uvm_perf_fault_max_throttle_per_service_p;
unsigned* uvm_perf_reenable_prefetch_faults_lapse_msec_p;

//
// pointers to nvidia 'functions'
// 
bool (*rhacuvm___uvm_check_locked)(void *lock, uvm_lock_order_t lock_order, uvm_lock_flags_t flags);
void* (*rhacuvm___uvm_kvmalloc)(size_t, const char *, int, const char *);
void* (*rhacuvm___uvm_kvmalloc_zero)(size_t size, const char *file, int line, const char *function);
bool (*rhacuvm___uvm_locking_initialized)(void);
NV_STATUS (*rhacuvm___uvm_push_begin_acquire_with_info)(uvm_channel_manager_t *, uvm_channel_type_t, uvm_gpu_t *, uvm_tracker_t *, uvm_push_t *, const char *, const char *, int , const char *, ...);
bool (*rhacuvm___uvm_record_lock)(void *lock, uvm_lock_order_t lock_order, uvm_lock_flags_t flags);
bool (*rhacuvm___uvm_record_unlock)(void *lock, uvm_lock_order_t lock_order, uvm_lock_flags_t flags);
int (*rhacuvm_nv_status_to_errno)(NV_STATUS status);
const char* (*rhacuvm_nvstatusToString)(NV_STATUS);
void (*rhacuvm_on_uvm_assert)(void);
const char* (*rhacuvm_uvm_aperture_string)(uvm_aperture_t);
NV_STATUS (*rhacuvm_uvm_ats_invalidate_tlbs)(uvm_gpu_va_space_t *, uvm_ats_fault_invalidate_t *, uvm_tracker_t *);
NV_STATUS (*rhacuvm_uvm_ats_service_fault_entry)(uvm_gpu_va_space_t *, uvm_fault_buffer_entry_t *, uvm_ats_fault_invalidate_t *);
bool (*rhacuvm_uvm_debug_prints_enabled)(void);
const char* (*rhacuvm_uvm_fault_access_type_string)(uvm_fault_access_type_t);
bool (*rhacuvm_uvm_file_is_nvidia_uvm)(struct file *);
NV_STATUS (*rhacuvm_uvm_gpu_check_ecc_error_no_rm)(uvm_gpu_t *gpu);
uvm_gpu_t* (*rhacuvm_uvm_gpu_chunk_get_gpu)(const uvm_gpu_chunk_t *chunk);
void (*rhacuvm_uvm_gpu_disable_prefetch_faults)(uvm_gpu_t *);
void (*rhacuvm_uvm_gpu_enable_prefetch_faults)(uvm_gpu_t *);
NV_STATUS (*rhacuvm_uvm_gpu_fault_entry_to_va_space)(uvm_gpu_t *, uvm_fault_buffer_entry_t *, uvm_va_space_t **);
NV_STATUS (*rhacuvm_uvm_global_mask_check_ecc_error)(uvm_global_processor_mask_t *gpus);
void (*rhacuvm_uvm_global_mask_retain)(const uvm_global_processor_mask_t *);
void (*rhacuvm_uvm_global_mask_release)(const uvm_global_processor_mask_t *);
void (*rhacuvm_uvm_gpu_kref_put)(uvm_gpu_t *);
NV_STATUS (*rhacuvm_uvm_gpu_map_cpu_pages)(uvm_gpu_t *, struct page *, size_t , NvU64 *);
void (*rhacuvm_uvm_gpu_replayable_faults_isr_lock)(uvm_gpu_t *);
void (*rhacuvm_uvm_gpu_replayable_faults_isr_unlock)(uvm_gpu_t *);
void (*rhacuvm_uvm_gpu_unmap_cpu_pages)(uvm_gpu_t *gpu, NvU64 dma_address, size_t size);
void (*rhacuvm_uvm_kvfree)(void *);
NV_STATUS (*rhacuvm_uvm_mem_map_cpu)(uvm_mem_t *mem, struct vm_area_struct *vma);
void (*rhacuvm_uvm_mem_unmap_cpu)(uvm_mem_t *mem);
uvm_gpu_address_t (*rhacuvm_uvm_mmu_gpu_address_for_big_page_physical)(uvm_gpu_address_t physical, uvm_gpu_t *gpu);
void (*rhacuvm_uvm_perf_event_notify)(uvm_perf_va_space_events_t *, uvm_perf_event_t, uvm_perf_event_data_t *);
uvm_perf_prefetch_hint_t (*rhacuvm_uvm_perf_prefetch_get_hint)(uvm_va_block_t *va_block, const uvm_page_mask_t *new_residency_mask);
void (*rhacuvm_uvm_perf_prefetch_prenotify_fault_migrations)(uvm_va_block_t *va_block, uvm_va_block_context_t *va_block_context, uvm_processor_id_t new_residency, const uvm_page_mask_t *faulted_pages, uvm_va_block_region_t region);
uvm_perf_thrashing_hint_t (*rhacuvm_uvm_perf_thrashing_get_hint)(uvm_va_block_t *, NvU64 , uvm_processor_id_t);
uvm_processor_mask_t* (*rhacuvm_uvm_perf_thrashing_get_thrashing_processors)(uvm_va_block_t *va_block, NvU64 address);
NV_STATUS (*rhacuvm_uvm_pmm_gpu_alloc)(uvm_pmm_gpu_t *pmm, size_t num_chunks, uvm_chunk_size_t chunk_size, uvm_pmm_gpu_memory_type_t mem_type, uvm_pmm_alloc_flags_t flags, uvm_gpu_chunk_t **chunks, uvm_tracker_t *out_tracker);
void (*rhacuvm_uvm_pmm_gpu_free)(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, uvm_tracker_t *tracker);
NvU64 (*rhacuvm_uvm_pmm_gpu_indirect_peer_addr)(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, uvm_gpu_t *accessing_gpu);
NV_STATUS (*rhacuvm_uvm_pmm_gpu_indirect_peer_map)(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, uvm_gpu_t *accessing_gpu);
void (*rhacuvm_uvm_pmm_gpu_mark_root_chunk_unused)(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk);
void (*rhacuvm_uvm_pmm_gpu_mark_root_chunk_used)(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk);
uvm_gpu_address_t (*rhacuvm_uvm_pmm_gpu_peer_copy_address)(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, uvm_gpu_t *accessing_gpu); 
uvm_gpu_phys_address_t (*rhacuvm_uvm_pmm_gpu_peer_phys_address)(uvm_pmm_gpu_t *pmm, uvm_gpu_chunk_t *chunk, uvm_gpu_t *accessing_gpu);
NV_STATUS (*rhacuvm_uvm_pmm_sysmem_mappings_add_gpu_mapping)(uvm_pmm_sysmem_mappings_t *sysmem_mappings, NvU64 dma_addr, NvU64 virt_addr, NvU64 region_size, uvm_va_block_t *va_block, uvm_processor_id_t owner);
void (*rhacuvm_uvm_pmm_sysmem_mappings_remove_gpu_mapping)(uvm_pmm_sysmem_mappings_t *sysmem_mappings, NvU64 dma_addr);
uvm_processor_id_t (*rhacuvm_uvm_processor_mask_find_closest_id)(uvm_va_space_t *, const uvm_processor_mask_t *, uvm_processor_id_t);
const char *(*rhacuvm_uvm_prot_string)(uvm_prot_t);
void (*rhacuvm_uvm_push_acquire_tracker)(uvm_push_t *, uvm_tracker_t *);
void (*rhacuvm_uvm_push_end)(uvm_push_t *);
NV_STATUS (*rhacuvm_uvm_push_end_and_wait)(uvm_push_t *push);
bool (*rhacuvm_uvm_range_group_address_migratable)(uvm_va_space_t *va_space, NvU64 address);
uvm_range_group_range_t* (*rhacuvm_uvm_range_group_range_iter_first)(uvm_va_space_t *va_space, NvU64 start, NvU64 end);
uvm_range_group_range_t* (*rhacuvm_uvm_range_group_range_iter_next)(uvm_va_space_t *va_space, uvm_range_group_range_t *range, NvU64 end);
void (*rhacuvm_uvm_range_group_range_migratability_iter_first)(uvm_va_space_t *, NvU64 , NvU64 , uvm_range_group_range_iter_t *);
void (*rhacuvm_uvm_range_group_range_migratability_iter_next)(uvm_va_space_t *, uvm_range_group_range_iter_t *, NvU64);
NV_STATUS (*rhacuvm_uvm_spin_loop)(uvm_spin_loop_t *);
bool (*rhacuvm_uvm_thread_context_add)(uvm_thread_context_t *thread_context);
void (*rhacuvm_uvm_thread_context_remove)(uvm_thread_context_t *thread_context);
bool (*rhacuvm_uvm_thread_context_wrapper_is_used)(void);
void (*rhacuvm_uvm_tlb_batch_begin)(uvm_page_tree_t *tree, uvm_tlb_batch_t *batch);
void (*rhacuvm_uvm_tlb_batch_invalidate)(uvm_tlb_batch_t *, NvU64, NvU64, NvU32, uvm_membar_t);
void (*rhacuvm_uvm_tools_broadcast_replay)(uvm_gpu_t *, uvm_push_t *, NvU32, uvm_fault_client_type_t);
void (*rhacuvm_uvm_tools_flush_events)(void);
void (*rhacuvm_uvm_tools_record_block_migration_begin)(uvm_va_block_t *va_block, uvm_push_t *push, uvm_processor_id_t dst_id, uvm_processor_id_t src_id, NvU64 start, uvm_make_resident_cause_t cause);
void (*rhacuvm_uvm_tools_record_cpu_fatal_fault)(uvm_va_space_t *, NvU64, bool, UvmEventFatalReason);
void (*rhacuvm_uvm_tools_record_gpu_fatal_fault)(uvm_gpu_id_t , uvm_va_space_t *, const uvm_fault_buffer_entry_t *, UvmEventFatalReason);
void (*rhacuvm_uvm_tools_record_read_duplicate)(uvm_va_block_t *va_block, uvm_processor_id_t dst, uvm_va_block_region_t region, const uvm_page_mask_t *page_mask);
void (*rhacuvm_uvm_tools_record_read_duplicate_invalidate)(uvm_va_block_t *va_block, uvm_processor_id_t dst, uvm_va_block_region_t region, const uvm_page_mask_t *page_mask);
void (*rhacuvm_uvm_tools_record_throttling_end)(uvm_va_space_t *, NvU64, uvm_processor_id_t);
void (*rhacuvm_uvm_tools_record_throttling_start)(uvm_va_space_t *, NvU64, uvm_processor_id_t );
void (*rhacuvm_uvm_tools_record_map_remote)(uvm_va_block_t *va_block, uvm_push_t *push, uvm_processor_id_t processor, uvm_processor_id_t residency, NvU64 address, size_t region_size, UvmEventMapRemoteCause cause);
NV_STATUS (*rhacuvm_uvm_tracker_add_push_safe)(uvm_tracker_t *, uvm_push_t *);
NV_STATUS (*rhacuvm_uvm_tracker_add_tracker_safe)(uvm_tracker_t *, uvm_tracker_t *);
void (*rhacuvm_uvm_tracker_deinit)(uvm_tracker_t *);
NV_STATUS (*rhacuvm_uvm_tracker_wait)(uvm_tracker_t *);
NV_STATUS (*rhacuvm_uvm_va_block_add_mappings_after_migration)(uvm_va_block_t *va_block, uvm_va_block_context_t *va_block_context, uvm_processor_id_t new_residency, uvm_processor_id_t processor_id, uvm_va_block_region_t region, const uvm_page_mask_t *map_page_mask, uvm_prot_t max_prot, const uvm_processor_mask_t *thrashing_processors);
size_t (*rhacuvm_uvm_va_block_big_page_index)(uvm_va_block_t *va_block, uvm_page_index_t page_index, NvU32 big_page_size);
uvm_va_block_region_t (*rhacuvm_uvm_va_block_big_page_region)(uvm_va_block_t *, size_t , NvU32 );
uvm_va_block_region_t (*rhacuvm_uvm_va_block_big_page_region_all)(uvm_va_block_t *va_block, NvU32 big_page_size);
uvm_va_block_context_t* (*rhacuvm_uvm_va_block_context_alloc)(void);
void (*rhacuvm_uvm_va_block_context_free)(uvm_va_block_context_t *va_block_context);
NV_STATUS (*rhacuvm_uvm_va_block_find_create)(uvm_va_space_t *, NvU64 , uvm_va_block_t **);
NvU32 (*rhacuvm_uvm_va_block_gpu_big_page_size)(uvm_va_block_t *va_block, uvm_gpu_t *gpu);
size_t (*rhacuvm_uvm_va_block_gpu_chunk_index_range)(NvU64 start, NvU64 size, uvm_gpu_t *gpu, uvm_page_index_t page_index, uvm_chunk_size_t *out_chunk_size);
NV_STATUS (*rhacuvm_uvm_va_block_map)(uvm_va_block_t *va_block, uvm_va_block_context_t *va_block_context, uvm_processor_id_t id, uvm_va_block_region_t region, const uvm_page_mask_t *map_page_mask, uvm_prot_t new_prot, UvmEventMapRemoteCause cause, uvm_tracker_t *out_tracker);
const uvm_page_mask_t* (*rhacuvm_uvm_va_block_map_mask_get)(uvm_va_block_t *block, uvm_processor_id_t processor);
void (*rhacuvm_uvm_va_block_page_authorized_processors)(uvm_va_block_t *va_block, uvm_page_index_t page_index, uvm_prot_t access_permission, uvm_processor_mask_t *authorized_processors);
bool (*rhacuvm_uvm_va_block_page_is_gpu_authorized)(uvm_va_block_t *, uvm_page_index_t , uvm_gpu_id_t , uvm_prot_t);
bool (*rhacuvm_uvm_va_block_page_is_processor_authorized)(uvm_va_block_t *va_block, uvm_page_index_t page_index, uvm_processor_id_t processor_id, uvm_prot_t required_prot);
void (*rhacuvm_uvm_va_block_page_resident_processors)(uvm_va_block_t *va_block, uvm_page_index_t page_index, uvm_processor_mask_t *resident_processors);
uvm_page_mask_t* (*rhacuvm_uvm_va_block_resident_mask_get)(uvm_va_block_t *block, uvm_processor_id_t processor);
void (*rhacuvm_uvm_va_block_retry_deinit)(uvm_va_block_retry_t *, uvm_va_block_t *);
void (*rhacuvm_uvm_va_block_retry_init)(uvm_va_block_retry_t *);
NV_STATUS (*rhacuvm_uvm_va_block_revoke_prot)(uvm_va_block_t *va_block, uvm_va_block_context_t *va_block_context, uvm_processor_id_t id, uvm_va_block_region_t region, const uvm_page_mask_t *revoke_page_mask, uvm_prot_t prot_to_revoke, uvm_tracker_t *out_tracker);
NV_STATUS (*rhacuvm_uvm_va_block_revoke_prot_mask)(uvm_va_block_t *va_block, uvm_va_block_context_t *va_block_context, const uvm_processor_mask_t *revoke_processor_mask, uvm_va_block_region_t region, const uvm_page_mask_t *revoke_page_mask, uvm_prot_t prot_to_revoke);
uvm_processor_id_t (*rhacuvm_uvm_va_block_select_residency)(uvm_va_block_t *, uvm_page_index_t , uvm_processor_id_t , NvU32 , const uvm_perf_thrashing_hint_t *, uvm_service_operation_t , bool *);
NV_STATUS (*rhacuvm_uvm_va_block_service_locked)(uvm_processor_id_t, uvm_va_block_t *, uvm_va_block_retry_t *, uvm_service_block_context_t *);
NV_STATUS (*rhacuvm_uvm_va_block_set_cancel)(uvm_va_block_t *, uvm_gpu_t *);
NV_STATUS (*rhacuvm_uvm_va_block_unmap)(uvm_va_block_t *va_block, uvm_va_block_context_t *va_block_context, uvm_processor_id_t id, uvm_va_block_region_t region, const uvm_page_mask_t *unmap_page_mask, uvm_tracker_t *out_tracker);
NV_STATUS (*rhacuvm_uvm_va_block_unmap_mask)(uvm_va_block_t *va_block, uvm_va_block_context_t *va_block_context, const uvm_processor_mask_t *unmap_processor_mask, uvm_va_block_region_t region, const uvm_page_mask_t *unmap_page_mask);
NV_STATUS (*rhacuvm_uvm_va_range_check_logical_permissions)(uvm_va_range_t *, uvm_processor_id_t, uvm_fault_type_t, bool);
NV_STATUS (*rhacuvm_uvm_va_range_create_mmap)(uvm_va_space_t *va_space, uvm_vma_wrapper_t *vma_wrapper, uvm_va_range_t **out_va_range);
void (*rhacuvm_uvm_va_range_destroy)(uvm_va_range_t *va_range, struct list_head *deferred_free_list);
uvm_va_range_t* (*rhacuvm_uvm_va_range_find)(uvm_va_space_t *va_space, NvU64 addr);
uvm_prot_t (*rhacuvm_uvm_va_range_logical_prot)(uvm_va_range_t *);
NV_STATUS (*rhacuvm_uvm_va_range_split)(uvm_va_range_t *existing_va_range, NvU64 new_end, uvm_va_range_t **new_va_range);
void (*rhacuvm_uvm_va_range_zombify)(uvm_va_range_t *va_range);
bool (*rhacuvm_uvm_va_space_can_read_duplicate)(uvm_va_space_t *va_space, uvm_gpu_t *changing_gpu);
uvm_va_range_t* (*rhacuvm_uvm_va_space_iter_first)(uvm_va_space_t *va_space, NvU64 start, NvU64 end);
uvm_va_range_t* (*rhacuvm_uvm_va_space_iter_next)(uvm_va_range_t *va_range, NvU64 end);
bool (*rhacuvm_uvm_va_space_mm_enabled)(uvm_va_space_t *va_space);
void (*rhacuvm_uvm_va_space_mm_release)(uvm_va_space_t *);
struct mm_struct* (*rhacuvm_uvm_va_space_mm_retain)(uvm_va_space_t *);
bool (*rhacuvm_uvm_va_space_peer_enabled)(uvm_va_space_t *va_space, uvm_gpu_t *gpu1, uvm_gpu_t *gpu2);
void (*rhacuvm_uvm_va_space_stop_all_user_channels)(uvm_va_space_t *va_space);
uvm_vma_wrapper_t* (*rhacuvm_uvm_vma_wrapper_alloc)(struct vm_area_struct *vma);
void (*rhacuvm_uvm_vma_wrapper_destroy)(uvm_vma_wrapper_t *vma_wrapper);
NV_STATUS (*rhacuvm_read_duplication_set)(uvm_va_space_t *, NvU64, NvU64, bool);
NV_STATUS (*rhacuvm_uvm_va_space_split_span_as_needed)(uvm_va_space_t *, NvU64, NvU64, uvm_va_range_is_split_needed_t split_needed_cb, void *);
NV_STATUS (*rhacuvm_uvm_va_range_block_create)(uvm_va_range_t *, size_t, uvm_va_block_t **);
size_t (*rhacuvm_uvm_va_range_block_index)(uvm_va_range_t *, NvU64);
size_t (*rhacuvm_uvm_va_range_block_next)(uvm_va_range_t *, uvm_va_block_t *);
uvm_prot_t (*rhacuvm_uvm_va_block_page_compute_highest_permission)(uvm_va_block_t *, uvm_processor_id_t, uvm_page_index_t);

NV_STATUS (*rhacuvm_fault_buffer_flush_locked)(uvm_gpu_t *gpu, uvm_gpu_buffer_flush_mode_t flush_mode, uvm_fault_replay_type_t fault_replay, uvm_fault_service_batch_context_t *batch_context);
NV_STATUS (*rhacuvm_push_replay_on_gpu)(uvm_gpu_t *gpu, uvm_fault_replay_type_t type, uvm_fault_service_batch_context_t *batch_context);
NV_STATUS (*rhacuvm_push_cancel_on_gpu)(uvm_gpu_t *gpu, uvm_gpu_phys_address_t instance_ptr, NvU32 gpc_id, NvU32 client_id, uvm_tracker_t *tracker);


NV_STATUS (*rhacuvm_preprocess_fault_batch)(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context);
void (*rhacuvm_fetch_fault_buffer_entries)(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context, int fetch_mode);
NV_STATUS (*rhacuvm_service_fault_batch)(uvm_gpu_t *gpu, int service_mode, uvm_fault_service_batch_context_t *batch_context);

NV_STATUS (*rhacuvm_uvm_va_block_make_resident)(uvm_va_block_t *va_block, uvm_va_block_retry_t *va_block_retry, uvm_va_block_context_t *va_block_context, uvm_processor_id_t dest_id, uvm_va_block_region_t region, const uvm_page_mask_t *page_mask, const uvm_page_mask_t *prefetch_page_mask, uvm_make_resident_cause_t cause);
NV_STATUS (*rhacuvm_uvm_va_block_make_resident_read_duplicate)(uvm_va_block_t *va_block, uvm_va_block_retry_t *va_block_retry, uvm_va_block_context_t *va_block_context, uvm_processor_id_t dest_id, uvm_va_block_region_t region, const uvm_page_mask_t *page_mask, const uvm_page_mask_t *prefetch_page_mask, uvm_make_resident_cause_t cause);

bool (*rhacuvm_uvm_range_group_all_migratable)(uvm_va_space_t *va_space, NvU64 start, NvU64 end);
size_t (*rhacuvm_uvm_va_range_num_blocks)(uvm_va_range_t *va_range);

size_t (*rhacuvm_uvm_va_block_num_big_pages)(uvm_va_block_t *va_block, NvU32 big_page_size);
NV_STATUS (*rhacuvm_block_alloc_ptes_new_state)(uvm_va_block_t *, uvm_gpu_t *, uvm_va_block_new_pte_state_t *, uvm_tracker_t *);
NV_STATUS (*rhacuvm_block_gpu_change_swizzling_map)(uvm_va_block_t *, uvm_va_block_context_t *, uvm_gpu_t *, uvm_gpu_t *, uvm_tracker_t *);
void (*rhacuvm_block_gpu_map_to_2m)(uvm_va_block_t *, uvm_va_block_context_t *, uvm_gpu_t *, uvm_processor_id_t, uvm_prot_t, uvm_push_t *, block_pte_op_t);
void (*rhacuvm_block_gpu_map_split_2m)(uvm_va_block_t *, uvm_va_block_context_t *, uvm_gpu_t *, uvm_processor_id_t, const uvm_page_mask_t *, uvm_prot_t, uvm_push_t *, block_pte_op_t);
void (*rhacuvm_block_gpu_map_big_and_4k)(uvm_va_block_t *, uvm_va_block_context_t *, uvm_gpu_t *, uvm_processor_id_t, const uvm_page_mask_t *, uvm_prot_t, uvm_push_t *, block_pte_op_t);
uvm_processor_id_t (*rhacuvm_block_gpu_get_processor_to_map)(uvm_va_block_t *, uvm_gpu_t *, uvm_page_index_t);

int rhac_nvidia_symbols_load(void)
{

	/* 1. Load Pointers */
	load_ptr(g_uvm_va_block_gpu_state_cache);
	load_ptr(g_uvm_page_mask_cache);
	gg_uvm_va_block_gpu_state_cache = *g_uvm_va_block_gpu_state_cache_p;
	gg_uvm_page_mask_cache = *g_uvm_page_mask_cache_p;

	load_ptr(slab_caches);
	load_ptr(g_uvm_global);
	load_ptr(uvm_enable_builtin_tests);
	load_ptr(uvm_enable_debug_procfs);
	load_ptr(uvm_perf_fault_max_batches_per_service);
	load_ptr(uvm_perf_fault_max_throttle_per_service);
	load_ptr(uvm_perf_reenable_prefetch_faults_lapse_msec);

	/* 2. Load Functions */
	load_function(__uvm_check_locked);
	load_function(__uvm_kvmalloc);
	load_function(__uvm_kvmalloc_zero);
	load_function(__uvm_locking_initialized);
	load_function(__uvm_push_begin_acquire_with_info);
	load_function(__uvm_record_lock);
	load_function(__uvm_record_unlock);
	load_function(nv_status_to_errno);
	load_function(nvstatusToString);
	load_function(on_uvm_assert);
	load_function(uvm_aperture_string);
	load_function(uvm_ats_invalidate_tlbs);
	load_function(uvm_ats_service_fault_entry);
	load_function(uvm_debug_prints_enabled);
	load_function(uvm_fault_access_type_string);
	load_function(uvm_file_is_nvidia_uvm);
	load_function(uvm_gpu_check_ecc_error_no_rm);
	load_function(uvm_gpu_disable_prefetch_faults);
	load_function(uvm_gpu_enable_prefetch_faults);
	load_function(uvm_gpu_fault_entry_to_va_space);
	load_function(uvm_global_mask_check_ecc_error);
	load_function(uvm_global_mask_retain);
	load_function(uvm_global_mask_release);
	load_function(uvm_gpu_kref_put);
	load_function(uvm_gpu_map_cpu_pages);
	load_function(uvm_gpu_replayable_faults_isr_lock);
	load_function(uvm_gpu_replayable_faults_isr_unlock);
	load_function(uvm_gpu_unmap_cpu_pages);
	load_function(uvm_kvfree);
	load_function(uvm_mem_map_cpu);
	load_function(uvm_mem_unmap_cpu);
	load_function(uvm_mmu_gpu_address_for_big_page_physical);
	load_function(uvm_perf_event_notify);
	load_function(uvm_perf_prefetch_get_hint);
	load_function(uvm_perf_prefetch_prenotify_fault_migrations);
	load_function(uvm_perf_thrashing_get_hint);
	load_function(uvm_perf_thrashing_get_thrashing_processors);
	load_function(uvm_pmm_gpu_alloc);
	load_function(uvm_pmm_gpu_free);
	load_function(uvm_pmm_gpu_indirect_peer_addr);
	load_function(uvm_pmm_gpu_indirect_peer_map);
	load_function(uvm_pmm_gpu_mark_root_chunk_unused);
	load_function(uvm_pmm_gpu_mark_root_chunk_used);
	load_function(uvm_pmm_gpu_peer_copy_address);
	load_function(uvm_pmm_gpu_peer_phys_address);
	load_function(uvm_pmm_sysmem_mappings_add_gpu_mapping);
	load_function(uvm_pmm_sysmem_mappings_remove_gpu_mapping);
	load_function(uvm_processor_mask_find_closest_id);
	load_function(uvm_prot_string);
	load_function(uvm_push_acquire_tracker);
	load_function(uvm_push_end);
	load_function(uvm_push_end_and_wait);
	load_function(uvm_range_group_address_migratable);
	load_function(uvm_range_group_range_iter_first);
	load_function(uvm_range_group_range_iter_next);
	load_function(uvm_range_group_range_migratability_iter_first);
	load_function(uvm_range_group_range_migratability_iter_next);
	load_function(uvm_spin_loop);
	load_function(uvm_thread_context_add);
	load_function(uvm_thread_context_remove);
	load_function(uvm_thread_context_wrapper_is_used);
	load_function(uvm_tlb_batch_begin);
	load_function(uvm_tlb_batch_invalidate);
	load_function(uvm_tools_broadcast_replay);
	load_function(uvm_tools_flush_events);
	load_function(uvm_tools_record_block_migration_begin);
	load_function(uvm_tools_record_cpu_fatal_fault);
	load_function(uvm_tools_record_gpu_fatal_fault);
	load_function(uvm_tools_record_read_duplicate);
	load_function(uvm_tools_record_read_duplicate_invalidate);
	load_function(uvm_tools_record_throttling_end);
	load_function(uvm_tools_record_throttling_start);
	load_function(uvm_tools_record_map_remote);
	load_function(uvm_tracker_add_push_safe);
	load_function(uvm_tracker_add_tracker_safe);
	load_function(uvm_tracker_deinit);
	load_function(uvm_tracker_wait);
	load_function(uvm_va_block_add_mappings_after_migration);
	load_function(uvm_va_block_big_page_index);
	load_function(uvm_va_block_big_page_region);
	load_function(uvm_va_block_big_page_region_all);
	load_function(uvm_va_block_context_alloc);
	load_function(uvm_va_block_context_free);
	load_function(uvm_va_block_find_create);
	load_function(uvm_va_block_gpu_big_page_size);
	load_function(uvm_gpu_chunk_get_gpu);
	load_function(uvm_va_block_gpu_chunk_index_range);
	load_function(uvm_va_block_make_resident_read_duplicate);
	load_function(uvm_va_block_map);
	load_function(uvm_va_block_map_mask_get);
	load_function(uvm_va_block_page_authorized_processors);
	load_function(uvm_va_block_page_is_gpu_authorized);
	load_function(uvm_va_block_page_is_processor_authorized);
	load_function(uvm_va_block_page_resident_processors);
	load_function(uvm_va_block_resident_mask_get);
	load_function(uvm_va_block_retry_deinit);
	load_function(uvm_va_block_retry_init);
	load_function(uvm_va_block_revoke_prot);
	load_function(uvm_va_block_revoke_prot_mask);
	load_function(uvm_va_block_select_residency);
	load_function(uvm_va_block_service_locked);
	load_function(uvm_va_block_set_cancel);
	load_function(uvm_va_block_unmap);
	load_function(uvm_va_block_unmap_mask);
	load_function(uvm_va_range_check_logical_permissions);
	load_function(uvm_va_range_create_mmap);
	load_function(uvm_va_range_destroy);
	load_function(uvm_va_range_find);
	load_function(uvm_va_range_logical_prot);
	load_function(uvm_va_range_split);
	load_function(uvm_va_range_zombify);
	load_function(uvm_va_space_can_read_duplicate);
	load_function(uvm_va_space_iter_first);
	load_function(uvm_va_space_iter_next);
	load_function(uvm_va_space_mm_enabled);
	load_function(uvm_va_space_mm_release);
	load_function(uvm_va_space_mm_retain);
	load_function(uvm_va_space_peer_enabled);
	load_function(uvm_va_space_stop_all_user_channels);
	load_function(uvm_vma_wrapper_alloc);
	load_function(uvm_vma_wrapper_destroy);

	load_function(read_duplication_set);
	load_function(uvm_va_space_split_span_as_needed);
	load_function(uvm_va_range_block_create);
	load_function(uvm_va_range_block_index);
	load_function(uvm_va_range_block_next);
	load_function(uvm_va_block_page_compute_highest_permission);

	load_function(fault_buffer_flush_locked);
	load_function(push_replay_on_gpu);
	load_function(push_cancel_on_gpu);
	load_function(preprocess_fault_batch);
	load_function(fetch_fault_buffer_entries);
	load_function(service_fault_batch);

	load_function(uvm_va_block_make_resident);
	load_function(uvm_va_block_make_resident_read_duplicate);

	load_function(uvm_range_group_all_migratable);
	load_function(uvm_va_range_num_blocks);

	load_function(uvm_va_block_num_big_pages);
  load_function(block_alloc_ptes_new_state);
  load_function(block_gpu_change_swizzling_map);
  load_function(block_gpu_map_to_2m);
  load_function(block_gpu_map_split_2m);
  load_function(block_gpu_map_big_and_4k);
  load_function(block_gpu_get_processor_to_map);

	return 0;
}

void rhac_nvidia_symbols_unload(void)
{
	/* Do nothing */
}
