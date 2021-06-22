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

#include "rhac_utils.h"
#include "rhac_nvidia_helpers.h"

#include "rhac_nvidia_decl.h"
#include "nvidia-uvm/uvm8_va_block.h"

static char buffer[UVM_PAGE_MASK_PRINT_MIN_BUFFER_SIZE];

void __rhac_print_all(const char *prefix, uvm_va_block_t *va_block)
{
	RHAC_LOG("%s-resident %u:%u:%u:%u:%u", prefix,
			uvm_processor_mask_test(&va_block->resident, UVM_ID_CPU),
			uvm_processor_mask_test(&va_block->resident, uvm_gpu_id(1)),
			uvm_processor_mask_test(&va_block->resident, uvm_gpu_id(2)),
			uvm_processor_mask_test(&va_block->resident, uvm_gpu_id(3)),
			uvm_processor_mask_test(&va_block->resident, uvm_gpu_id(4)));
	RHAC_LOG("%s-mapped %u:%u:%u:%u:%u", prefix,
			uvm_processor_mask_test(&va_block->mapped, UVM_ID_CPU),
			uvm_processor_mask_test(&va_block->mapped, uvm_gpu_id(1)),
			uvm_processor_mask_test(&va_block->mapped, uvm_gpu_id(2)),
			uvm_processor_mask_test(&va_block->mapped, uvm_gpu_id(3)),
			uvm_processor_mask_test(&va_block->mapped, uvm_gpu_id(4)));
	uvm_page_mask_print(&va_block->read_duplicated_pages, buffer);
	RHAC_LOG("%s-readdu %s", prefix, buffer);
	uvm_page_mask_print(&va_block->cpu.resident, buffer);
	RHAC_LOG("%s-cpu-resi %s", prefix, buffer);
	uvm_page_mask_print(&va_block->cpu.pte_bits[UVM_PTE_BITS_CPU_READ], buffer);
	RHAC_LOG("%s-cpu-read %s", prefix, buffer);
	uvm_page_mask_print(&va_block->cpu.pte_bits[UVM_PTE_BITS_CPU_WRITE], buffer);
	RHAC_LOG("%s-cpu-writ %s", prefix, buffer);

	int i;
	for (i = 1; i <= 4; i++) {
		uvm_va_block_gpu_state_t *state = block_gpu_state_get(va_block, uvm_gpu_id(i));
		if (state) {
			uvm_page_mask_print(&state->resident, buffer);
			RHAC_LOG("%s-gpu%d-resi %s", prefix, i-1, buffer);

			uvm_page_mask_print(&state->pte_bits[UVM_PTE_BITS_GPU_READ], buffer);
			RHAC_LOG("%s-gpu%d-read %s", prefix, i-1, buffer);
			uvm_page_mask_print(&state->pte_bits[UVM_PTE_BITS_GPU_WRITE], buffer);
			RHAC_LOG("%s-gpu%d-writ %s", prefix, i-1, buffer);
			uvm_page_mask_print(&state->pte_bits[UVM_PTE_BITS_GPU_ATOMIC], buffer);
			RHAC_LOG("%s-gpu%d-atom %s", prefix, i-1, buffer);
		}
	}
}

void __rhac_print_page_mask(const char *prefix, const uvm_page_mask_t *mask)
{
	if (mask) {
		uvm_page_mask_print(mask, buffer);
		RHAC_LOG("%s - %s", prefix, buffer);
	}
}

void __rhac_print_processor_mask(const char *prefix, const uvm_processor_mask_t *mask) 
{
	RHAC_LOG("%s - %u:%u:%u:%u:%u", prefix,
			uvm_processor_mask_test(mask, UVM_ID_CPU),
			uvm_processor_mask_test(mask, uvm_gpu_id(1)),
			uvm_processor_mask_test(mask, uvm_gpu_id(2)),
			uvm_processor_mask_test(mask, uvm_gpu_id(3)),
			uvm_processor_mask_test(mask, uvm_gpu_id(4)));
}

