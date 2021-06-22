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

#include <linux/types.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/wait.h>

#include "rhac_utils.h"
#include "rhac_comm.h"

static struct kmem_cache *comm_handle_cache = NULL;

int rhac_comm_init(void)
{
	comm_handle_cache = KMEM_CACHE(rhac_comm, SLAB_HWCACHE_ALIGN);
	if (!comm_handle_cache) return -EINVAL;
	return 0;
}

void rhac_comm_deinit(void)
{
	if (comm_handle_cache) kmem_cache_destroy(comm_handle_cache);
	comm_handle_cache = NULL;
}

struct rhac_comm* rhac_comm_alloc(void)
{
	struct rhac_comm *comm = kmem_cache_alloc(comm_handle_cache, GFP_KERNEL);
	if (!comm) return NULL;

	init_waitqueue_head(&comm->wait_queue);
	atomic_set(&comm->posted, 0);
	atomic_set(&comm->cnt, 0);
	bitmap_zero(comm->copy_mask, RHAC_PDSC_PER_PBLK);

	comm->err = 0;
	comm->processing = 0;
	comm->parent = NULL;

	comm->cur = -1;
	atomic_set(&comm->req_unlock, 0);


	spin_lock_init(&comm->lock);
	INIT_LIST_HEAD(&comm->list);
	INIT_LIST_HEAD(&comm->local_list);

	return comm;
}

struct rhac_comm* rhac_comm_spawn(struct rhac_comm *pa)
{
  unsigned long flags;
	if (pa) {
		spin_lock_irqsave(&pa->lock, flags);
  }

	struct rhac_comm *comm = rhac_comm_alloc();
	comm->parent = pa;

	if (pa) {
		atomic_inc(&pa->cnt);
		list_add(&comm->list, &pa->list);
		spin_unlock_irqrestore(&pa->lock, flags);
	}

	//RHAC_LOG("SPAWN %llx (%llx)", (uint64_t)comm, (uint64_t) pa);
	return comm;
}

void rhac_comm_free(struct rhac_comm *comm)
{
  RHAC_ASSERT(atomic_read(&comm->cnt) == 0);
  bool wake = false;
  if (comm->parent) {
    unsigned long flags;
    spin_lock_irqsave(&comm->parent->lock, flags);
    if (atomic_dec_return(&comm->parent->cnt) == 0) {
      wake = true;
    }
    spin_unlock_irqrestore(&comm->parent->lock, flags);
    if (wake)
      wake_up_interruptible_sync(&comm->parent->wait_queue);
	}
	kmem_cache_free(comm_handle_cache, comm);
}

void rhac_comm_fail(struct rhac_comm *comm, int err)
{
	if (comm->parent)
		rhac_comm_fail(comm->parent, err);
	else
		comm->err = err;
}

#include "rhac_nvidia_pipeline.h"
int rhac_comm_wait(struct rhac_comm *comm)
{
	int err;

	// wait itself
	err = wait_event_interruptible_timeout(
			comm->wait_queue,
			atomic_read(&comm->cnt) == 0,
			20 * HZ);
  if (err == 0 && atomic_read(&comm->cnt) != 0) {
    struct rhac_comm *ch = list_entry(comm->list.next, struct rhac_comm, list);
    RHAC_LOG("comm timeout: (%llx, %d, %u, %u, %llx, %d, %llx)",
        (uint64_t)comm,
        comm->type,
        atomic_read(&comm->cnt),
        atomic_read(&comm->posted),
        (uint64_t)ch,
        ch->cur,
        comm->isr_ctx ? comm->isr_ctx->va_block->start : 5
        );
    return -EINVAL;
  }
	return comm->err;
}

void rhac_comm_post(struct rhac_comm *comm)
{
	atomic_inc(&comm->posted);
}

unsigned int rhac_comm_unpost(struct rhac_comm *comm)
{
	return atomic_dec_return(&comm->posted);
}
