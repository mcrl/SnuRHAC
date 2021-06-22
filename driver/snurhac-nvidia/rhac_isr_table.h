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

#ifndef __RHAC_ISR_TABLE_H__
#define __RHAC_ISR_TABLE_H__

#include <linux/spinlock_types.h>
#include <linux/bitmap.h>
#include <linux/scatterlist.h>

#include "rhac_config.h"

#include "nvidia-uvm/uvm8_va_block_types.h"

struct rhac_isr_block {
	atomic_t refcnt;

	uint32_t owner_ids[RHAC_PDSC_PER_PBLK];
	uvm_va_block_context_t block_context;

	struct page *pages[RHAC_PDSC_PER_PBLK];

	void *buf;
	uint64_t dma_addr;
	uint64_t dma_addrs[RHAC_PDSC_PER_PBLK];


	spinlock_t lock;
	atomic_t locked;
	struct list_head head;
};

struct rhac_isr_table {
	uint64_t base;
	uint64_t num_blocks;
	struct rhac_isr_block *blocks;
};

int rhac_isr_table_init(struct rhac_isr_table *isr_table, uint64_t base, uint64_t capacity);
void rhac_isr_table_deinit(struct rhac_isr_table *isr_table);

struct rhac_isr_block* rhac_isr_table_blk_find(struct rhac_isr_table *isr_table, uint64_t vaddr);
struct rhac_isr_block* rhac_isr_table_blk_get(struct rhac_isr_table *isr_table, uint64_t vaddr);
void rhac_isr_table_blk_put(struct rhac_isr_block *blk);


bool rhac_isr_table_try_local_lock(struct rhac_isr_table *isr_table, uint64_t blk_vaddr, struct list_head *list);
struct list_head* rhac_isr_table_local_unlock(struct rhac_isr_table *isr_table, uint64_t blk_vaddr);
#endif //__RHAC_ISR_TABLE_H__
