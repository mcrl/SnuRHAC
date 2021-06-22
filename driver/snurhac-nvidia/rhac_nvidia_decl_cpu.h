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

#ifndef __RHAC_NVIDIA_DECL_CPU_H__
#define __RHAC_NVIDIA_DECL_CPU_H__

// 
// NVIDIA-CPU
//

static LIST_HEAD(g_cpu_service_block_context_list);
static uvm_spinlock_t g_cpu_service_block_context_list_lock;
static NvU64 uvm_perf_authorized_cpu_fault_tracking_window_ns = 300000;

static int alloc_cpu_service_block_context_list(void)
{
	unsigned num_preallocated_contexts = 4;

	uvm_spin_lock_init(&g_cpu_service_block_context_list_lock, UVM_LOCK_ORDER_LEAF);

	// Pre-allocate some fault service contexts for the CPU and add them to the global list
	while (num_preallocated_contexts-- > 0) {
		uvm_service_block_context_t *service_context = uvm_kvmalloc(sizeof(*service_context));
		if (!service_context)
			return -ENOMEM;

		list_add(&service_context->cpu_fault.service_context_list, &g_cpu_service_block_context_list);
	}

	return 0;
}

static void free_cpu_service_block_context_list(void)
{
	uvm_service_block_context_t *service_context, *service_context_tmp;

	// Free fault service contexts for the CPU and add clear the global list
	list_for_each_entry_safe(service_context, service_context_tmp, &g_cpu_service_block_context_list,
			cpu_fault.service_context_list) {
		rhacuvm_uvm_kvfree(service_context);
	}
	INIT_LIST_HEAD(&g_cpu_service_block_context_list);
}

static uvm_service_block_context_t *get_cpu_fault_service_context(void)
{
	uvm_service_block_context_t *service_context;

	uvm_spin_lock(&g_cpu_service_block_context_list_lock);

	service_context = list_first_entry_or_null(&g_cpu_service_block_context_list, uvm_service_block_context_t,
			cpu_fault.service_context_list);

	if (service_context)
		list_del(&service_context->cpu_fault.service_context_list);

	uvm_spin_unlock(&g_cpu_service_block_context_list_lock);

	if (!service_context)
		service_context = uvm_kvmalloc(sizeof(*service_context));

	return service_context;
}

static void put_cpu_fault_service_context(uvm_service_block_context_t *service_context)
{
	uvm_spin_lock(&g_cpu_service_block_context_list_lock);

	list_add(&service_context->cpu_fault.service_context_list, &g_cpu_service_block_context_list);

	uvm_spin_unlock(&g_cpu_service_block_context_list_lock);
}

static bool skip_cpu_fault_with_valid_permissions(uvm_va_block_t *va_block,
		uvm_page_index_t page_index,
		uvm_fault_access_type_t fault_access_type)
{
	if (rhacuvm_uvm_va_block_page_is_processor_authorized(va_block,
				page_index,
				UVM_ID_CPU,
				uvm_fault_access_type_to_prot(fault_access_type))) {
		NvU64 now = NV_GETTIME();
		pid_t pid = current->pid;

		// Latch the pid/timestamp/page_index values for the first time
		if (!va_block->cpu.fault_authorized.first_fault_stamp) {
			va_block->cpu.fault_authorized.first_fault_stamp = now;
			va_block->cpu.fault_authorized.first_pid = pid;
			va_block->cpu.fault_authorized.page_index = page_index;

			return true;
		}

		// If the same thread shows up again, this means that the kernel
		// downgraded the page's PTEs. Service the fault to force a remap of
		// the page.
		if (va_block->cpu.fault_authorized.first_pid == pid &&
				va_block->cpu.fault_authorized.page_index == page_index) {
			va_block->cpu.fault_authorized.first_fault_stamp = 0;
		}
		else {
			// If the window has expired, clear the information and service the
			// fault. Otherwise, just return
			if (now - va_block->cpu.fault_authorized.first_fault_stamp > uvm_perf_authorized_cpu_fault_tracking_window_ns)
				va_block->cpu.fault_authorized.first_fault_stamp = 0;
			else
				return true;
		}
	}

	return false;
}


#endif //__RHAC_NVIDIA_DECL_CPU_H__
