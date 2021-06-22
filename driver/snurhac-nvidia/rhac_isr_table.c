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

#include <linux/slab.h>
#include "rhac_utils.h"
#include "rhac_config.h"
#include "rhac_isr_table.h"


#define RHAC_ISR_LOCAL_LOCK_CNT 128

int rhac_isr_table_init(struct rhac_isr_table *isr_table, uint64_t base, uint64_t capacity)
{
	uint64_t cap_roundup = round_up(capacity, RHAC_PBLK_SIZE);
	uint64_t num_blocks = cap_roundup >> RHAC_PBLK_SIZE_SHIFT;

	isr_table->base = base;
	isr_table->num_blocks = num_blocks;
	isr_table->blocks = vzalloc(num_blocks  * sizeof(struct rhac_isr_block));
	if (!isr_table->blocks)
		return -ENOMEM;


	uint64_t i;
	for (i = 0; i < num_blocks; i++) {
		spin_lock_init(&isr_table->blocks[i].lock);
		atomic_set(&isr_table->blocks[i].refcnt, 0);
		atomic_set(&isr_table->blocks[i].locked, 0);
		INIT_LIST_HEAD(&isr_table->blocks[i].head);
	}

	RHAC_LOG("isr table total size: %llu (%llu blocks)", num_blocks  * sizeof(struct rhac_isr_block), num_blocks);

	return 0;
}

void rhac_isr_table_deinit(struct rhac_isr_table *isr_table)
{
	if (isr_table->blocks)
		vfree(isr_table->blocks);
	isr_table->blocks = NULL;
	isr_table->base = 0;
	isr_table->num_blocks = 0;
}

struct rhac_isr_block* rhac_isr_table_blk_find(struct rhac_isr_table *isr_table, uint64_t vaddr)
{
	uint64_t bidx = RHAC_BIDX(vaddr, isr_table->base);
	if (bidx >= isr_table->num_blocks) {
		RHAC_LOG("bidx: %llu, baddr: %llx, num_blocks: %llu", bidx, vaddr, isr_table->num_blocks);
		RHAC_ASSERT(false);
		return 0;
	}

	return &isr_table->blocks[bidx];
}

struct rhac_isr_block* rhac_isr_table_blk_get(struct rhac_isr_table *isr_table, uint64_t vaddr)
{
	uint64_t bidx = RHAC_BIDX(vaddr, isr_table->base);
	if (bidx >= isr_table->num_blocks) {
		RHAC_ASSERT(false);
		return 0;
	}
	struct rhac_isr_block *blk = &isr_table->blocks[bidx];

	atomic_inc(&blk->refcnt);

	return blk;
}

void rhac_isr_table_blk_put(struct rhac_isr_block *blk)
{
	atomic_dec(&blk->refcnt);
}

bool rhac_isr_table_try_local_lock(struct rhac_isr_table *isr_table, uint64_t blk_vaddr, struct list_head *list)
{
	return false;

	struct rhac_isr_block *blk = rhac_isr_table_blk_find(isr_table, blk_vaddr);
	bool locked = false;
	unsigned long flags;
	spin_lock_irqsave(&blk->lock, flags);
	if (atomic_read(&blk->locked)) {
		list_add_tail(list, &blk->head);
		locked = true;
	} else {
		atomic_set(&blk->locked, RHAC_ISR_LOCAL_LOCK_CNT);
	}
	spin_unlock_irqrestore(&blk->lock, flags);
	return locked;
}

struct list_head* rhac_isr_table_local_unlock(struct rhac_isr_table *isr_table, uint64_t blk_vaddr)
{
	return NULL;

	struct rhac_isr_block *blk = rhac_isr_table_blk_find(isr_table, blk_vaddr);
	struct list_head *list = NULL;

	unsigned long flags;
	spin_lock_irqsave(&blk->lock, flags);
	if (!list_empty(&blk->head)) {
		list = blk->head.next;
		list_del(list);
		atomic_dec(&blk->locked);
	} else {
		atomic_set(&blk->locked, 0);
	}
	spin_unlock_irqrestore(&blk->lock, flags);

	return list;
}
