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

#include "rhac_utils.h"
#include "rhac_nvidia_symbols.h"
#include "rhac_nvidia.h"
#include "rhac_nvidia_cpu.h"
#include "rhac_nvidia_mm.h"
#include "rhac_nvidia_helpers.h"
#include "rhac_comm.h"

#include "rhac_nvidia_decl.h"
#include "rhac_nvidia_common.h"
#include "rhac_nvidia_pipeline.h"


static NV_STATUS rhac_block_cpu_fault_locked(
		struct rhac_comm *comm,
		uvm_va_block_t *va_block,
		NvU64 fault_addr,
		uvm_fault_access_type_t fault_access_type,
		uvm_service_block_context_t *service_context
		)
{
	uvm_va_range_t *va_range = va_block->va_range;
	NV_STATUS status = NV_OK;
	uvm_page_index_t page_index;
	uvm_perf_thrashing_hint_t thrashing_hint;
	uvm_processor_id_t new_residency;
	bool read_duplicate;

	UVM_ASSERT(va_range);
	uvm_assert_rwsem_locked(&va_range->va_space->lock);
	UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_MANAGED);

	UVM_ASSERT(fault_addr >= va_block->start);
	UVM_ASSERT(fault_addr <= va_block->end);

	// There are up to three mm_structs to worry about, and they might all be
	// different:
	//
	// 1) vma->vm_mm
	// 2) current->mm
	// 3) va_space->va_space_mm.mm
	//
	// The kernel guarantees that vma->vm_mm has a reference taken with mmap_sem
	// held on the CPU fault path, so tell the fault handler to use that one.
	// current->mm might differ if we're on the access_process_vm (ptrace) path
	// or if another driver is calling get_user_pages.
	service_context->block_context.mm = uvm_va_range_vma(va_range)->vm_mm;

	if (service_context->num_retries == 0) {
		// notify event to tools/performance heuristics
		uvm_perf_event_notify_cpu_fault(&va_range->va_space->perf_events,
				va_block,
				fault_addr,
				fault_access_type > UVM_FAULT_ACCESS_TYPE_READ,
				KSTK_EIP(current));
	}

	// Check logical permissions
	status = rhacuvm_uvm_va_range_check_logical_permissions(va_block->va_range,
			UVM_ID_CPU,
			fault_access_type,
			rhacuvm_uvm_range_group_address_migratable(va_range->va_space, fault_addr));
	if (status != NV_OK)
		return status;

	uvm_processor_mask_zero(&service_context->cpu_fault.gpus_to_check_for_ecc);

	page_index = uvm_va_block_cpu_page_index(va_block, fault_addr);
	if (skip_cpu_fault_with_valid_permissions(va_block, page_index, fault_access_type))
		return NV_OK;

	thrashing_hint = rhacuvm_uvm_perf_thrashing_get_hint(va_block, fault_addr, UVM_ID_CPU);
#define RHAC_ISR_THRASHING_OFF
#ifdef RHAC_ISR_THRASHING_OFF
	thrashing_hint.type = UVM_PERF_THRASHING_HINT_TYPE_NONE;
#endif
	// Throttling is implemented by sleeping in the fault handler on the CPU
	if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_THROTTLE) {
		service_context->cpu_fault.wakeup_time_stamp = thrashing_hint.throttle.end_time_stamp;
		return NV_WARN_MORE_PROCESSING_REQUIRED;
	}

	service_context->read_duplicate_count = 0;
	service_context->thrashing_pin_count = 0;
	service_context->operation = UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS;

	if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_PIN) {
		uvm_page_mask_zero(&service_context->thrashing_pin_mask);
		uvm_page_mask_set(&service_context->thrashing_pin_mask, page_index);
		service_context->thrashing_pin_count = 1;
	}

	// Compute new residency and update the masks
	new_residency = rhacuvm_uvm_va_block_select_residency(va_block,
			page_index,
			UVM_ID_CPU,
			uvm_fault_access_type_mask_bit(fault_access_type),
			&thrashing_hint,
			UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS,
			&read_duplicate);

//	// RELAXED: read_duplicate is set depending on access type
//	if (fault_access_type <= UVM_FAULT_ACCESS_TYPE_READ) {
//		read_duplicate = true;
//	}

	// Initialize the minimum necessary state in the fault service context
	uvm_processor_mask_zero(&service_context->resident_processors);

	// Set new residency and update the masks
	uvm_processor_mask_set(&service_context->resident_processors, new_residency);

	// The masks need to be fully zeroed as the fault region may grow due to prefetching
	uvm_page_mask_zero(&service_context->per_processor_masks[uvm_id_value(new_residency)].new_residency);
	uvm_page_mask_set(&service_context->per_processor_masks[uvm_id_value(new_residency)].new_residency, page_index);

	if (read_duplicate) {
		uvm_page_mask_zero(&service_context->read_duplicate_mask);
		uvm_page_mask_set(&service_context->read_duplicate_mask, page_index);
		service_context->read_duplicate_count = 1;
	}

	service_context->access_type[page_index] = fault_access_type;

	service_context->region = uvm_va_block_region_for_page(page_index);

	status = rhac_uvm_va_block_service_locked_global(comm, UVM_ID_CPU, va_block, service_context);

	++service_context->num_retries;

	return status;
}

int rhac_uvm_cpu_fault_start(
		struct rhac_comm *comm,
		uvm_va_block_t *va_block,
		NvU64 fault_addr,
		bool is_write,
		uvm_service_block_context_t *service_context
		)
{
	uvm_fault_access_type_t fault_access_type;

	if (is_write)
		fault_access_type = UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG;
	else
		fault_access_type = UVM_FAULT_ACCESS_TYPE_READ;

	service_context->num_retries = 0;
	service_context->cpu_fault.did_migrate = false;

	// We have to use vm_insert_page instead of handing the page to the kernel
	// and letting it insert the mapping, and we must do that while holding the
	// lock on this VA block. Otherwise there will be a window in which we think
	// we've mapped the page but the CPU mapping hasn't actually been created
	// yet. During that window a GPU fault event could arrive and claim
	// ownership of that VA, "unmapping" it. Then later the kernel would
	// eventually establish the mapping, and we'd end up with both CPU and GPU
	// thinking they each owned the page.
	//
	// This function must only be called when it's safe to call vm_insert_page.
	// That is, there's a reference held on the vma's vm_mm and vm_mm->mmap_sem
	// is held in at least read mode, but current->mm might not be vma->vm_mm.

	NV_STATUS status;
	do {                                                            
		status = rhac_block_cpu_fault_locked(
				comm,
				va_block,
				fault_addr,
				fault_access_type,
				service_context
				);
		RHAC_ASSERT(status != NV_ERR_MORE_PROCESSING_REQUIRED);
	} while (status == NV_ERR_MORE_PROCESSING_REQUIRED);            
	return status != NV_OK;
}

int rhac_uvm_cpu_fault_done(
		struct rhac_comm *comm,
		uvm_va_block_t *va_block
		)
{
	return 0;
}


static int rhac_vm_fault(struct vm_area_struct *vma, struct vm_fault *vmf)
{
  //RHAC_LOG("CPU START");
	uvm_va_space_t *va_space = uvm_va_space_get(vma->vm_file);
	uvm_va_block_t *va_block;
	NvU64 fault_addr = nv_page_fault_va(vmf);
	bool is_write = vmf->flags & FAULT_FLAG_WRITE;
	NV_STATUS status = uvm_global_get_status();
	bool tools_enabled;
	bool major_fault = false;
	uvm_service_block_context_t *service_context;
	uvm_global_processor_mask_t gpus_to_check_for_ecc;

	if (status != NV_OK)
		goto convert_error;

	if (!uvm_down_read_trylock_no_tracking(&g_uvm_global_p->pm.lock)) {
		status = NV_ERR_BUSY_RETRY;
		goto convert_error;
	}

	service_context = get_cpu_fault_service_context();
	if (!service_context) {
		status = NV_ERR_NO_MEMORY;
		goto unlock;
	}

	service_context->cpu_fault.wakeup_time_stamp = 0;

	uvm_record_lock_mmap_sem_read(&vma->vm_mm->mmap_sem);

	do {
		bool do_sleep = false;
		if (status == NV_WARN_MORE_PROCESSING_REQUIRED) {
			NvU64 now = NV_GETTIME();
			if (now < service_context->cpu_fault.wakeup_time_stamp)
				do_sleep = true;

			if (do_sleep)
				rhacuvm_uvm_tools_record_throttling_start(va_space, fault_addr, UVM_ID_CPU);

			//uvm_va_space_up_read(va_space);

			if (do_sleep) {
				unsigned long nap_us = (service_context->cpu_fault.wakeup_time_stamp - now) / 1000;

				usleep_range(nap_us, nap_us + nap_us / 2);
			}
		}

		//uvm_va_space_down_read(va_space);

		if (do_sleep)
			rhacuvm_uvm_tools_record_throttling_end(va_space, fault_addr, UVM_ID_CPU);

		status = rhacuvm_uvm_va_block_find_create(va_space, fault_addr, &va_block);
		if (status != NV_OK) {
			UVM_ASSERT_MSG(status == NV_ERR_NO_MEMORY, "status: %s\n", rhacuvm_nvstatusToString(status));
			break;
		}

		UVM_ASSERT(vma == uvm_va_range_vma(va_block->va_range));

		struct rhac_comm *comm = rhac_comm_alloc();
		if (!comm) return NV_ERR_GENERIC;

		comm->type = 0;
		status = rhac_nvidia_pipeline_cpu_fault(comm, va_block, fault_addr, is_write, service_context);
		if (rhac_comm_wait(comm)) {
			RHAC_ASSERT(false);
			status = NV_ERR_GENERIC;
		}
		rhac_comm_free(comm);
	} while (status == NV_WARN_MORE_PROCESSING_REQUIRED);

	RHAC_ASSERT(status == NV_OK);

	if (status != NV_OK) {
		UvmEventFatalReason reason;

		reason = uvm_tools_status_to_fatal_fault_reason(status);
		UVM_ASSERT(reason != UvmEventFatalReasonInvalid);

		rhacuvm_uvm_tools_record_cpu_fatal_fault(va_space, fault_addr, is_write, reason);
	}

	tools_enabled = va_space->tools.enabled;

	if (status == NV_OK) {
		uvm_va_space_global_gpus_in_mask(va_space,
				&gpus_to_check_for_ecc,
				&service_context->cpu_fault.gpus_to_check_for_ecc);
		rhacuvm_uvm_global_mask_retain(&gpus_to_check_for_ecc);
	}

	//uvm_va_space_up_read(va_space);
	uvm_record_unlock_mmap_sem_read(&vma->vm_mm->mmap_sem);

	if (status == NV_OK) {
		status = rhacuvm_uvm_global_mask_check_ecc_error(&gpus_to_check_for_ecc);
		rhacuvm_uvm_global_mask_release(&gpus_to_check_for_ecc);
	}

	if (tools_enabled)
		rhacuvm_uvm_tools_flush_events();

	major_fault = service_context->cpu_fault.did_migrate;
	put_cpu_fault_service_context(service_context);

  //RHAC_LOG("CPU END");
unlock:
	uvm_up_read_no_tracking(&g_uvm_global_p->pm.lock);

convert_error:
	switch (status) {
		case NV_OK:
		case NV_ERR_BUSY_RETRY:
			return VM_FAULT_NOPAGE | (major_fault ? VM_FAULT_MAJOR : 0);
		case NV_ERR_NO_MEMORY:
			return VM_FAULT_OOM;
		default:
			return VM_FAULT_SIGBUS;
	}
}

static vm_fault_t rhac_vm_fault_wrapper(struct vm_fault *vmf)
{
#if defined(NV_VM_OPS_FAULT_REMOVED_VMA_ARG)
	return rhac_vm_fault(vmf->vma, vmf);
#else
	return rhac_vm_fault(NULL, vmf);
#endif
}

static vm_fault_t rhac_vm_fault_entry(struct vm_area_struct *vma, struct vm_fault *vmf)
{
	UVM_ENTRY_RET(rhac_vm_fault(vma, vmf));
}

static vm_fault_t rhac_vm_fault_wrapper_entry(struct vm_fault *vmf)
{
	UVM_ENTRY_RET(rhac_vm_fault_wrapper(vmf));
}



static vm_fault_t (*cpu_fault)(struct vm_fault *vmf);
static vm_fault_t (*cpu_page_mkwrite)(struct vm_fault *vmf);
static bool cpu_initialized;

int rhac_nvidia_cpu_init(void)
{
	struct vm_area_struct *vma;
	struct vm_operations_struct *vm_ops;
	struct mm_struct *mm = current->mm;

	down_write(&mm->mmap_sem);
	for (vma = mm->mmap; vma; vma = vma->vm_next) {
		if (rhacuvm_uvm_file_is_nvidia_uvm(vma->vm_file)) {
			vm_ops = (struct vm_operations_struct*)vma->vm_ops;
			if (!cpu_fault) 
				cpu_fault = vm_ops->fault;
			if (!cpu_page_mkwrite)
				cpu_page_mkwrite = vm_ops->page_mkwrite;

#if defined(NV_VM_OPS_FAULT_REMOVED_VMA_ARG)
			vm_ops->fault = rhac_vm_fault_wrapper_entry;
			vm_ops->page_mkwrite = rhac_vm_fault_wrapper_entry;
#else
			vm_ops->fault = rhac_vm_fault_entry;
			vm_ops->page_mkwrite = rhac_vm_fault_entry;
#endif
		}
	}
	up_write(&mm->mmap_sem);
	alloc_cpu_service_block_context_list();
	cpu_initialized = true;

	return 0;
}

void rhac_nvidia_cpu_deinit(void)
{
	struct vm_area_struct *vma;
	struct vm_operations_struct *vm_ops;
	struct mm_struct *mm = current->mm;
	if (!mm) return;

	free_cpu_service_block_context_list();

	if (cpu_initialized) {
		down_write(&mm->mmap_sem);
		for (vma = mm->mmap; vma; vma = vma->vm_next) {
			if (rhacuvm_uvm_file_is_nvidia_uvm(vma->vm_file)) {
				vm_ops = (struct vm_operations_struct*)vma->vm_ops;
				vm_ops->fault = cpu_fault;
				vm_ops->page_mkwrite = cpu_page_mkwrite;
			}
		}
		up_write(&mm->mmap_sem);
		cpu_fault = NULL;
		cpu_page_mkwrite = NULL;
		cpu_initialized = false;
	}
}
