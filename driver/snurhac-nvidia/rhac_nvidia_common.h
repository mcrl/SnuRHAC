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

#ifndef __RHAC_NVIDIA_COMMON_H__
#define __RHAC_NVIDIA_COMMON_H__

struct rhac_comm;
#include "nvidia-uvm/uvm8_va_block.h"

int rhac_uvm_va_block_service_locked_global(
		struct rhac_comm *comm,
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		uvm_service_block_context_t *service_context
		);
int rhac_uvm_va_block_service_locked_local(
		struct rhac_comm *comm,
		uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		uvm_service_block_context_t *service_context
		);

NV_STATUS uvm_va_block_service_locked(uvm_processor_id_t processor_id,
		uvm_va_block_t *va_block,
		uvm_va_block_retry_t *block_retry,
		uvm_service_block_context_t *service_context);

NV_STATUS rhac_uvm_va_block_make_resident_local(
		struct rhac_comm *comm,
		uvm_va_block_t *block,
		uvm_processor_id_t dst_id
		);

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
		);
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
		);

int block_copy_resident_pages_local(uvm_va_block_t *block,
		uvm_va_block_context_t *block_context,
		uvm_processor_id_t dst_id,
		uvm_va_block_region_t region,
		const uvm_page_mask_t *page_mask,
		const uvm_page_mask_t *prefetch_page_mask,
		uvm_va_block_transfer_mode_t transfer_mode);

int rhac_nvidia_make_resident_from_cpu(
		uvm_va_block_t *va_block,
		uvm_va_block_context_t *block_context,
		uvm_processor_id_t dst_id,
		uvm_va_block_region_t region,
    const uvm_page_mask_t *page_mask,
		const uvm_page_mask_t *copy_mask,
		uvm_va_block_transfer_mode_t mode
		);
#endif //__RHAC_NVIDIA_COMMON_H__
