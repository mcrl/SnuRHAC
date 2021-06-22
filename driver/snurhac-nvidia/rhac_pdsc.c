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

#include <linux/bitops.h>
#include <linux/types.h>
#include <linux/slab.h>
#include <linux/delay.h>

#include "rhac_utils.h"
#include "rhac_pdsc.h"


#include "rhac_comm.h"
#include "rhac_protocol.h"

#define rhac_pdsc_for_each_pdsc(pdsc, i, blk)                    \
	for (i = 0; i < RHAC_PDSC_PER_PBLK; i++)                \
			if ((pdsc = &(blk)->pdscs[i]) != NULL)    \


inline static uint64_t find_page_idx(uint64_t vaddr, uint64_t vaddr_base)
{
	return (vaddr - vaddr_base) >> PAGE_SHIFT;
}

uint32_t rhac_pdsc_owner(struct rhac_pdsc_table *table, uint64_t vaddr)
{
	return (find_page_idx(vaddr, table->vaddr_base) >> RHAC_PDSC_PER_PDIR_SHIFT) % table->num_nodes;
}

inline static struct rhac_pdsc_blk* find_blk(struct rhac_pdsc_table *table, uint64_t vaddr)
{

  RHAC_ASSERT(rhac_pdsc_owner(table, vaddr) == table->node_id);
  RHAC_ASSERT(vaddr >= (table->vaddr_base));
//  RHAC_ASSERT(vaddr < (table->vaddr_base) + (table->ndirs * RHAC_PDIR_SIZE));
//
//  if (!(vaddr < (table->vaddr_base) + (table->ndirs * RHAC_PDIR_SIZE))) {
//    RHAC_LOG("base: 0x%llx, vaddr: 0x%llx", table->vaddr_base, vaddr);
//  }

	uint64_t i = find_page_idx(vaddr, table->vaddr_base);

	RHAC_ASSERT((i >> RHAC_PDSC_PER_PDIR_SHIFT) % table->num_nodes == table->node_id);
	uint64_t di = (i >> RHAC_PDSC_PER_PDIR_SHIFT) / table->num_nodes;
	uint64_t bi = (i >> RHAC_PDSC_PER_PBLK_SHIFT) % RHAC_PBLK_PER_PDIR;

	return &table->dirs[di]->blks[bi];
}

bool rhac_pdsc_lock_blk_or_append(struct rhac_pdsc_table *table, uint64_t blk_vaddr, struct list_head *list)
{
	struct rhac_pdsc_blk *blk = find_blk(table, blk_vaddr);

	/*
	unsigned long flags;
	spin_lock_irqsave(&blk->lock, flags);
	*/
  mutex_lock(&blk->mutex);

	if (blk->flag & RHAC_PBLK_FLAG_LOCKED) {
		list_add_tail(list, &blk->waitlist);
	} else {
		blk->flag |= RHAC_PBLK_FLAG_LOCKED;
    mutex_unlock(&blk->mutex);
		//spin_unlock_irqrestore(&blk->lock, flags);
		return true;
	}
  RHAC_ASSERT(blk->flag & RHAC_PBLK_FLAG_LOCKED);

	mutex_unlock(&blk->mutex);
	//spin_unlock_irqrestore(&blk->lock, flags);

	return false;
}

struct list_head* rhac_pdsc_unlock_blk_and_return_waitee(struct rhac_pdsc_table *table, uint64_t blk_vaddr)
{
	struct rhac_pdsc_blk *blk = find_blk(table, blk_vaddr);

	struct list_head *list = NULL;
	/*
	unsigned long flags;
	spin_lock_irqsave(&blk->lock, flags);
	*/
  mutex_lock(&blk->mutex);

	RHAC_ASSERT(blk->flag & RHAC_PBLK_FLAG_LOCKED);
	if (!list_empty(&blk->waitlist)) {
		list = blk->waitlist.next;
		list_del(list);
	} else {
		blk->flag &= (~RHAC_PBLK_FLAG_LOCKED);
	}
  mutex_unlock(&blk->mutex);
	//spin_unlock_irqrestore(&blk->lock, flags);

	return list;
}

static int rhac_pdsc_inv_blk(
    struct rhac_pdsc_table *table,
		struct rhac_comm *comm,
		struct rhac_pdsc_blk *blk,
		uint64_t vaddr)
{
	RHAC_ASSERT(!(blk->flag & RHAC_PBLK_FLAG_LOCKED));

	DECLARE_BITMAP(invmask, RHAC_PDSC_PER_PBLK);
	unsigned long *page_mask, *prot_mask;
	
  atomic_inc(&comm->cnt);
	int i, nsync = 0;
	for (i = 0; i < table->num_nodes; i++) {
		page_mask = blk->page_mask[i];
		prot_mask = blk->prot_mask[i];

		if (bitmap_andnot(invmask, page_mask, prot_mask, RHAC_PDSC_PER_PBLK)) {
		  bitmap_and(page_mask, page_mask, prot_mask, RHAC_PDSC_PER_PBLK);
			atomic_inc(&comm->cnt);

			int err;
			err = rhac_protocol_post_inv(comm, vaddr, invmask, i);
			RHAC_ASSERT(!err);
			nsync++;
		}
	}

	atomic_dec(&comm->cnt);
	return nsync;
}

int rhac_pdsc_inv(struct rhac_pdsc_table *table)
{
	int i, j, err;
	struct rhac_comm *pa = rhac_comm_alloc();

	uint64_t vaddr_base = table->vaddr_base + table->node_id * RHAC_PDIR_SIZE ;
	int total_sync = 0;
	for (j = 0; j < table->ndirs; j++) {
		struct rhac_pdsc_dir *dir = table->dirs[j];
		uint64_t vaddr = vaddr_base;
		for (i = 0; i < RHAC_PBLK_PER_PDIR; i++) {
			struct rhac_pdsc_blk *blk = &dir->blks[i];
      if (!(blk->flag & RHAC_PBLK_FLAG_READONLY)) {
        total_sync += rhac_pdsc_inv_blk(table, pa, blk, vaddr);
      }
			vaddr += RHAC_PBLK_SIZE;
		}
		vaddr_base += RHAC_PDIR_SIZE * table->num_nodes;
	}

	atomic_inc(&table->sync_cnt);

	err = rhac_comm_wait(pa);
	RHAC_ASSERT(!err);
	rhac_comm_free(pa);
	return 0;
}

int rhac_pdsc_set_readonly(
		struct rhac_pdsc_table *table,
		uint64_t blk_vaddr,
		uint64_t len
)
{

	uint64_t vaddr;
	for (vaddr = blk_vaddr; vaddr < blk_vaddr + len; vaddr += RHAC_PBLK_SIZE) {
    if (rhac_pdsc_owner(table, vaddr) == table->node_id) {
      struct rhac_pdsc_blk *blk = find_blk(table, vaddr);
      // NO lock required,
      blk->flag |= RHAC_PBLK_FLAG_READONLY;
    }
  }
  return 0;
}

int rhac_pdsc_unset_readonly(
		struct rhac_pdsc_table *table,
		uint64_t blk_vaddr,
		uint64_t len
)
{
	uint64_t vaddr;
	for (vaddr = blk_vaddr; vaddr < blk_vaddr + len; vaddr += RHAC_PBLK_SIZE) {
    if (rhac_pdsc_owner(table, vaddr) == table->node_id) {
      struct rhac_pdsc_blk *blk = find_blk(table, vaddr);
      // NO lock required,
      blk->flag &= ~RHAC_PBLK_FLAG_READONLY;
    }
  }
  return 0;
}

#include "rhac_nvidia_helpers.h"
void rhac_pdsc_update_blk(
		struct rhac_pdsc_table *table,
		uint64_t blk_vaddr,
		const unsigned long *page_mask,
		const unsigned long *prot_mask,
		const unsigned long *atom_mask,
		uint32_t src_id,
		uint32_t *owner_ids,
		bool *has_another_owner,
		bool *require_unlock
		)
{
	int i, j;

	struct rhac_pdsc_blk *blk = find_blk(table, blk_vaddr);

  RHAC_ASSERT(blk->flag & RHAC_PBLK_FLAG_LOCKED);

	DECLARE_BITMAP(mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(mask_, RHAC_PDSC_PER_PBLK);

	*has_another_owner = false;
	*require_unlock = false;


	unsigned long *this_page_mask = blk->page_mask[src_id];
	unsigned long *this_prot_mask = blk->prot_mask[src_id];

	uint32_t id;


	// NO WRITE SHARE, so ignore atom_mask;

	// If it's the first time, map to the requesting node
	for_each_set_bit(i, page_mask, RHAC_PDSC_PER_PBLK) {
		owner_ids[i] = src_id;
	}

	/*
	unsigned long flags;
	spin_lock_irqsave(&blk->lock, flags);
	*/
  mutex_lock(&blk->mutex);

  // Write
  if (bitmap_andnot(mask, prot_mask, this_prot_mask, RHAC_PDSC_PER_PBLK)) {
    for (j = 0; j < table->num_nodes; j++) {
      id = (src_id + (table->num_nodes - j)) % table->num_nodes;

      // Get from WRITE PERMITTED NODE for write pages
      unsigned long *target_mask = blk->prot_mask[id];
      if (bitmap_and(mask_, mask, target_mask, RHAC_PDSC_PER_PBLK)) {
        if (id != src_id) {
          *has_another_owner = true;
          *require_unlock = true;
        }
        for_each_set_bit(i, mask_, RHAC_PDSC_PER_PBLK) {
          owner_ids[i] = id;
        }
        bitmap_andnot(target_mask, target_mask, prot_mask, RHAC_PDSC_PER_PBLK);
        bitmap_andnot(mask, mask, target_mask, RHAC_PDSC_PER_PBLK);
      }
    }
  }
  bitmap_or(this_prot_mask, this_prot_mask, prot_mask, RHAC_PDSC_PER_PBLK);

  // Read
  bitmap_andnot(mask, page_mask, prot_mask, RHAC_PDSC_PER_PBLK);
  if (bitmap_andnot(mask, mask, this_page_mask, RHAC_PDSC_PER_PBLK)) {
		for (j = 0; j < table->num_nodes; j++) {
      id = (src_id + (table->num_nodes - j)) % table->num_nodes;

			unsigned long *target_mask = blk->page_mask[id];
      if (bitmap_and(mask_, mask, target_mask, RHAC_PDSC_PER_PBLK)) {
        if (id != src_id) {
          *has_another_owner = true;
        }
        for_each_set_bit(i, mask_, RHAC_PDSC_PER_PBLK) {
          owner_ids[i] = id;
        }
        bitmap_andnot(mask, mask, target_mask, RHAC_PDSC_PER_PBLK);
      }
    }
  }
  bitmap_or(this_page_mask, this_page_mask, page_mask, RHAC_PDSC_PER_PBLK);

  mutex_unlock(&blk->mutex);
	//spin_unlock_irqrestore(&blk->lock, flags);
}

static void pdsc_init_pblk(struct rhac_pdsc_blk *blk, uint32_t num_nodes)
{
	uint64_t i;

	/*
	spin_lock_init(&blk->lock);
	*/
	mutex_init(&blk->mutex);
	INIT_LIST_HEAD(&blk->waitlist);
	blk->flag = 0;

	for (i = 0; i < RHAC_MAX_NODES; i++) {
		bitmap_zero(blk->page_mask[i], RHAC_PDSC_PER_PBLK);
		bitmap_zero(blk->prot_mask[i], RHAC_PDSC_PER_PBLK);
		bitmap_zero(blk->prot_mask[i], RHAC_PDSC_PER_PBLK);
	}
}

static void pdsc_init_pdir(struct rhac_pdsc_dir *dir, uint32_t num_nodes)
{
	uint64_t i;

	for (i = 0; i < RHAC_PBLK_PER_PDIR; i++) {
		pdsc_init_pblk(&dir->blks[i], num_nodes);
	}
}

int rhac_pdsc_table_init(struct rhac_pdsc_table *table, uint64_t base, uint64_t capacity, uint32_t node_id, uint32_t num_nodes)
{
	table->node_id = node_id;
	table->vaddr_base = base;
	table->num_nodes = num_nodes;
	atomic_set(&table->sync_cnt, 0);

	uint64_t ndirs;
	uint64_t cap_roundup = round_up(capacity, RHAC_PDIR_SIZE);
//	ndirs = cap_roundup >> RHAC_PDIR_SIZE_SHIFT;
	ndirs = DIV_ROUND_UP_ULL(cap_roundup >> RHAC_PDIR_SIZE_SHIFT, num_nodes);

	// shrinking case: return immediatley
	if (ndirs <= table->ndirs) return 0;

	uint64_t i, j;
	struct rhac_pdsc_dir *dir;
	for (i = table->ndirs; i < ndirs; i++) {
		dir = kzalloc(sizeof(struct rhac_pdsc_dir), GFP_KERNEL);
		if (!dir) {
			for (j = table->ndirs; j < i; j++) {
				kfree(table->dirs[j]);
				table->dirs[j] = NULL;
			}
			return -EINVAL;
		}
		pdsc_init_pdir(dir, num_nodes);
		table->dirs[i] = dir;
	}

	table->ndirs = ndirs;

	//RHAC_LOG("pdsc: %lu, pblk: %lu, pdir: %lu", sizeof(struct rhac_pdsc), sizeof(struct rhac_pdsc_blk), sizeof(struct rhac_pdsc_dir));

	return 0;
}

void rhac_pdsc_table_deinit(struct rhac_pdsc_table *table)
{

	RHAC_LOG("SYNC CNT: %u", atomic_read(&table->sync_cnt));
	atomic_set(&table->sync_cnt, 0);

	uint64_t i;
	for (i = 0; i < RHAC_MAX_PDIR_PER_PTABLE; i++) {
		struct rhac_pdsc_dir *dir = table->dirs[i];
		if (!dir) break;
		kfree(dir);
		table->dirs[i] = NULL;
	}

	table->ndirs = 0;
	table->vaddr_base = 0;
	table->num_nodes = 0;
	table->node_id = 0;
}
