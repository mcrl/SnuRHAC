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

#ifndef __RHAC_PDSC_H__ 
#define __RHAC_PDSC_H__ 

#include <linux/mutex.h>
#include <linux/spinlock_types.h>
#include <linux/bitmap.h>
#include <linux/types.h>

#include "rhac_config.h"

struct list_head;
struct rhac_ctx;


enum rhac_pdsc_state {
	PDSC_STATE_INVALID          = 0,
	PDSC_STATE_READ_ONLY        = 1,
	PDSC_STATE_WRITE_EXCLUSIVE  = 2,
	PDSC_STATE_WRITE_SHARED     = 3,
	PDSC_STATE_MAX              = 4, 
};

enum rhac_pdsc_flag {
	PDSC_FLAG_BUG 				= 0, 
	PDSC_FLAG_ENABLE_WS			= 1,
	PDSC_FLAG_READ				= 2,
	PDSC_FLAG_WRITE				= 3,
	PDSC_FLAG_WRITE_EXCLUSIVE	        = 4,
	PDSC_FLAG_LOCKED			= 5,
	PDSC_FLAG_MAX				= 6,
};

enum rhac_pblk_flag {
	RHAC_PBLK_FLAG_LOCKED   = (1<<0),
	RHAC_PBLK_FLAG_READONLY = (1<<1),
	RHAC_PBLK_FLAG_MAX      = (1<<2),
};

typedef struct {
	char __state[(RHAC_MAX_NODES * PDSC_STATE_MAX) >> 3];
} rhac_pdsc_status_t;

typedef struct {
	DECLARE_BITMAP(__bitmap, PDSC_FLAG_MAX);
} rhac_pdsc_flag_t;


struct rhac_pdsc {
	rhac_pdsc_status_t state;
	rhac_pdsc_flag_t flag;
};

struct rhac_pdsc_blk {
	struct mutex mutex;
	spinlock_t lock;
	int flag;
	struct list_head waitlist;

	DECLARE_BITMAP(page_mask, RHAC_PDSC_PER_PBLK)[RHAC_MAX_NODES];
	DECLARE_BITMAP(prot_mask, RHAC_PDSC_PER_PBLK)[RHAC_MAX_NODES];
};

struct rhac_pdsc_dir {
	struct rhac_pdsc_blk blks[RHAC_PBLK_PER_PDIR];
};

struct rhac_pdsc_table {
	uint64_t ndirs;
	uint64_t vaddr_base;
	uint64_t num_nodes;
	uint32_t node_id;

	struct rhac_pdsc_dir* dirs[RHAC_MAX_PDIR_PER_PTABLE];

	atomic_t sync_cnt;
};

#define rhac_pdsc_for_each(blk, dir, vaddr, i, j, table)                                  \
	for (i = 0, vaddr = (table)->vaddr_base + (table)->node_id * RHAC_PDIR_SIZE;      \
			i < (table)->ndirs;                                               \
			i++, vaddr += (RHAC_PDIR_SIZE * (table)->num_nodes))              \
			if ((dir = (table)->dirs[i]) != NULL)                             \
				for (j = 0; j < RHAC_PBLK_PER_PDIR; j++)                  \
						if ((blk = &(dir)->blks[j]) != NULL)      \

int rhac_pdsc_table_init(struct rhac_pdsc_table *table, uint64_t base, uint64_t capacity, uint32_t node_id, uint32_t num_nodes);
void rhac_pdsc_table_deinit(struct rhac_pdsc_table *table);
uint32_t rhac_pdsc_owner(struct rhac_pdsc_table *table, uint64_t vaddr);
int rhac_pdsc_set_readonly(struct rhac_pdsc_table *table, uint64_t blk_vaddr, uint64_t len);
int rhac_pdsc_unset_readonly(struct rhac_pdsc_table *table, uint64_t blk_vaddr, uint64_t len);
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
		);
bool rhac_pdsc_lock_blk_or_append(struct rhac_pdsc_table *table, uint64_t blk_vaddr, struct list_head *list);
struct list_head* rhac_pdsc_unlock_blk_and_return_waitee(struct rhac_pdsc_table *table, uint64_t blk_vaddr);
int rhac_pdsc_inv(struct rhac_pdsc_table *table);

bool pdsc_inv_blk(struct rhac_pdsc_blk *blk, uint32_t node_id, uint64_t vaddr, unsigned long *page_mask);
#endif //__RHAC_PDSC_H__
