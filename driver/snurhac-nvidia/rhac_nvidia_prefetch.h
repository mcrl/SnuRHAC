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

#ifndef __RHAC_NVIDIA_PREFETCH_H__
#define __RHAC_NVIDIA_PREFETCH_H__

#include <linux/cdev.h>
#include <linux/mm.h>

#include "rhac_config.h"

struct rhac_comm;

NV_STATUS rhac_uvm_va_range_migrate_multi_block(
		struct rhac_comm *pa,
		uvm_va_range_t *va_range,
		NvU64 start,
		NvU64 end,
		uvm_processor_id_t dest_id,
		uvm_migrate_mode_t mode,
		uvm_tracker_t *out_tracker,
		uvm_mutex_t *lock);

NV_STATUS rhac_uvm_migrate(
		struct rhac_comm *pa,
		uvm_va_space_t *va_space,
		NvU64 base,
		NvU64 length,
		uvm_processor_id_t dest_id,
		NvU32 migrate_flags,
		uvm_tracker_t *out_tracker,
		uvm_mutex_t *lock);

int rhac_uvm_va_block_migrate_global(
		struct rhac_comm *comm,
		uvm_va_block_t *va_block,
		uvm_va_block_retry_t *va_block_retry,
		uvm_va_block_context_t *va_block_context,
		uvm_va_block_region_t region,
    uvm_page_mask_t *page_mask,
		uvm_processor_id_t dest_id,
		uvm_migrate_mode_t mode);

int rhac_uvm_va_block_migrate_wrapup(
		struct rhac_comm *comm,
		uvm_va_block_t *va_block,
		uvm_va_block_context_t *va_block_context,
		uvm_va_block_region_t region,
    uvm_page_mask_t *page_mask,
		uvm_processor_id_t dest_id,
		uvm_migrate_mode_t mode);

NV_STATUS rhac_uvm_va_block_map(uvm_va_block_t *va_block,
    uvm_va_block_context_t *va_block_context,
    uvm_processor_id_t id,
    uvm_va_block_region_t region,
    const uvm_page_mask_t *map_page_mask,
    uvm_prot_t new_prot,
    UvmEventMapRemoteCause cause,
    uvm_tracker_t *out_tracker);

#endif //__RHAC_NVIDIA_PREFETCH_H__
