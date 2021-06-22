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

#include <linux/dmapool.h>
#include <linux/dma-mapping.h>
#include <linux/smp.h>
#include <linux/topology.h>
#include <linux/cpumask.h>
#include <linux/bitmap.h>
#include <linux/wait.h>
#include <rdma/ib_verbs.h>

#include "rhac_ctx.h"
#include "rhac_rdma.h"
#include "rhac_protocol.h"
#include "rhac_config.h"
#include "rhac_utils.h"
#include "rhac_pdsc.h"
#include "rhac_msg.h"
#include "rhac_nvidia_mm.h"
#include "rhac_nvidia_pipeline.h"

#include "rhac_comm.h"

#include "rhac_nvidia_helpers.h"


#define RHAC_BUFFER_NUM_POOLS 10
#define RHAC_BUFFER_CACHE_INIT 1024
#define RHAC_PROT_NUM_RECV 8191

struct rdma_buffer_12 { char bytes[(1 << 12)]; };
struct rdma_buffer_13 { char bytes[(1 << 13)]; };
struct rdma_buffer_14 { char bytes[(1 << 14)]; };
struct rdma_buffer_15 { char bytes[(1 << 15)]; };
struct rdma_buffer_16 { char bytes[(1 << 16)]; };
struct rdma_buffer_17 { char bytes[(1 << 17)]; };
struct rdma_buffer_18 { char bytes[(1 << 18)]; };
struct rdma_buffer_19 { char bytes[(1 << 19)]; };
struct rdma_buffer_20 { char bytes[(1 << 20)]; };
struct rdma_buffer_21 { char bytes[(1 << 21)]; };

static struct kmem_cache *msg_cache = NULL;

static struct ib_cqe rdma_cqe;

static struct rhac_msg *recv_msgs[RHAC_PROT_NUM_RECV];

static struct workqueue_struct* workqueue[RHAC_MSG_MAX];
static struct dma_pool *rdma_msg_pool;
static struct dma_pool *buffer_pool[RHAC_BUFFER_NUM_POOLS];
//static struct kmem_cache *rdma_buffer_pool[RHAC_BUFFER_NUM_POOLS];

static void send_done(struct ib_cq *cq, struct ib_wc *wc);
static void recv_done(struct ib_cq *cq, struct ib_wc *wc);
static void rdma_done(struct ib_cq *cq, struct ib_wc *wc);
static void send_done_msg(struct rhac_msg *msg);

static void rhac_protocol_process_lock_req(struct work_struct *work);
static void rhac_protocol_process_lock_rsp(struct work_struct *work);
static void rhac_protocol_process_update_req(struct work_struct *work);
static void rhac_protocol_process_update_rsp(struct work_struct *work);
static void rhac_protocol_process_unlock_req(struct work_struct *work);
static void rhac_protocol_process_map_req(struct work_struct *work);
static void rhac_protocol_process_map_rsp(struct work_struct *work);
static void rhac_protocol_process_unmap_req(struct work_struct *work);
static void rhac_protocol_process_inv_req(struct work_struct *work);
static void rhac_protocol_process_inv_rsp(struct work_struct *work);

static void rhac_protocol_process_map_req_copy(struct work_struct *work);

typedef void (*handler_t)(struct work_struct *);
static handler_t handlers[RHAC_MSG_MAX] = {
	[RHAC_MSG_LOCK_REQ]   =  rhac_protocol_process_lock_req,
	[RHAC_MSG_LOCK_RSP]   =  rhac_protocol_process_lock_rsp,
	[RHAC_MSG_UNLOCK_REQ] =  rhac_protocol_process_unlock_req,
	[RHAC_MSG_UPDATE_REQ] =  rhac_protocol_process_update_req,
	[RHAC_MSG_UPDATE_RSP] =  rhac_protocol_process_update_rsp,
	[RHAC_MSG_MAP_REQ]    =  rhac_protocol_process_map_req,
	[RHAC_MSG_MAP_RSP]    =  rhac_protocol_process_map_rsp,
	[RHAC_MSG_UNMAP_REQ]  =  rhac_protocol_process_unmap_req,
	[RHAC_MSG_INV_REQ]    =  rhac_protocol_process_inv_req,
	[RHAC_MSG_INV_RSP]    =  rhac_protocol_process_inv_rsp,
};

static uint64_t msg_size[RHAC_MSG_MAX] = {
	[RHAC_MSG_LOCK_REQ]   = sizeof(struct rhac_msg_lock_req),
	[RHAC_MSG_LOCK_RSP]   = sizeof(struct rhac_msg_lock_rsp),
	[RHAC_MSG_UNLOCK_REQ] = sizeof(struct rhac_msg_unlock_req),
	[RHAC_MSG_UPDATE_REQ] = sizeof(struct rhac_msg_update_req),
	[RHAC_MSG_UPDATE_RSP] = sizeof(struct rhac_msg_update_rsp),
	[RHAC_MSG_MAP_REQ]    = sizeof(struct rhac_msg_map_req),
	[RHAC_MSG_MAP_RSP]    = sizeof(struct rhac_msg_map_rsp),
	[RHAC_MSG_UNMAP_REQ]  = sizeof(struct rhac_msg_unmap_req),
	[RHAC_MSG_INV_REQ]    = sizeof(struct rhac_msg_inv_req),
	[RHAC_MSG_INV_RSP]    = sizeof(struct rhac_msg_inv_rsp),
};

static const char *msg_strs[] = {
	[RHAC_MSG_LOCK_REQ]  	    = "RHAC_MSG_LOCK_REQ    ",
	[RHAC_MSG_LOCK_RSP]  	    = "RHAC_MSG_LOCK_RSP    ",
	[RHAC_MSG_UNLOCK_REQ]       = "RHAC_MSG_UNLOCK_REQ  ",
	[RHAC_MSG_UPDATE_REQ] 	    = "RHAC_MSG_UPDATE_REQ  ",
	[RHAC_MSG_UPDATE_RSP] 	    = "RHAC_MSG_UPDATE_RSP  ",
	[RHAC_MSG_MAP_REQ] 	    = "RHAC_MSG_MAP_REQ     ",
	[RHAC_MSG_MAP_RSP] 	    = "RHAC_MSG_MAP_RSP     ",
	[RHAC_MSG_UNMAP_REQ] 	    = "RHAC_MSG_UNMAP_REQ   ",
	[RHAC_MSG_INV_REQ] 	    = "RHAC_MSG_INV_REQ     ",
	[RHAC_MSG_INV_RSP] 	    = "RHAC_MSG_INV_RSP     ",
};

inline static struct rhac_comm* msg_to_comm(struct rhac_msg *msg)
{
	return (struct rhac_comm*) (msg->tag);
}

static atomic_t total_cnt;
static atomic_t send_msg_cnt[RHAC_MSG_MAX];
static atomic_t recv_msg_cnt[RHAC_MSG_MAX];
static atomic_t buffer_read_cnt[RHAC_MAX_NODES][RHAC_MSG_MAX];
static atomic_t buffer_read_npages[RHAC_MAX_NODES];

inline int get_buffer_idx(int npages)
{
	int i, np = 1;
	for (i = 0; i < RHAC_BUFFER_NUM_POOLS; i++) {
		if (npages <= np) {
			break;
		}
		np <<= 1;
	}
	return i;
}

enum {
  RDMA_BUFFER_SEND,
  RDMA_BUFFER_RECV,
};

static void* rdma_buffer_alloc(int npages, uint64_t *dma_addr, int type)
{
	int i = get_buffer_idx(npages);
	void *buf = dma_pool_alloc(buffer_pool[i], GFP_KERNEL, dma_addr);
	RHAC_ASSERT(buf && !dma_mapping_error(rhac_rdma_device(), *dma_addr));

	/*
	void *buf = kmem_cache_alloc(rdma_buffer_pool[get_buffer_idx(npages)], GFP_KERNEL);
	RHAC_ASSERT(buf);

	*dma_addr = rhac_rdma_map_single(buf, npages * PAGE_SIZE, DMA_BIDIRECTIONAL);
	RHAC_ASSERT(!dma_mapping_error(rhac_rdma_device(), *dma_addr));
	*/

	return buf;
}

static void* rdma_send_buffer_alloc(int npages, uint64_t *dma_addr)
{
  return rdma_buffer_alloc(npages, dma_addr, RDMA_BUFFER_SEND);
}

static void* rdma_recv_buffer_alloc(int npages, uint64_t *dma_addr)
{
  return rdma_buffer_alloc(npages, dma_addr, RDMA_BUFFER_RECV);
}

static void rdma_buffer_free(int npages, void *buf, uint64_t dma_addr)
{

	dma_pool_free(buffer_pool[get_buffer_idx(npages)], buf, dma_addr);

	/*
	rhac_rdma_unmap_single(dma_addr, npages * PAGE_SIZE, DMA_BIDIRECTIONAL);
	kmem_cache_free(rdma_buffer_pool[get_buffer_idx(npages)], buf);
	*/
}

inline static struct workqueue_struct* get_queue(int type)
{
	return workqueue[type];
}

static struct rhac_msg* msg_buf_alloc(void)
{
	struct rhac_msg *msg = kmem_cache_alloc(msg_cache, GFP_KERNEL);
	RHAC_ASSERT(msg);
	msg->dma_addr = 0;
	return msg;
}

static void msg_buf_free(struct rhac_msg *msg)
{
	kmem_cache_free(msg_cache, msg);
}

#ifndef RHAC_RDMA_SEND_INLINE
static struct rhac_msg *msg_dma_alloc(void)
{
	uint64_t dma_addr;
	struct rhac_msg *msg = dma_pool_alloc(rdma_msg_pool, GFP_KERNEL, &dma_addr);
	RHAC_ASSERT(msg && !dma_mapping_error(rhac_rdma_device(), dma_addr));
	if (!msg || dma_mapping_error(rhac_rdma_device(), dma_addr))
		return NULL;
	msg->dma_addr = dma_addr;
	return msg;
}

static void msg_dma_free(struct rhac_msg *msg)
{
	dma_pool_free(rdma_msg_pool, msg, msg->dma_addr);
}

static struct rhac_msg* msg_alloc(uint64_t dst_id)
{
  struct rhac_ctx *ctx = rhac_ctx_get_global();
  struct rhac_msg *msg = NULL;

  if (ctx->node_id == dst_id) {
    msg = msg_buf_alloc();
  } else {
    msg = msg_dma_alloc();
  }

  RHAC_ASSERT(!IS_ERR_OR_NULL(msg));
  return msg;
}

static void msg_free(struct rhac_msg *msg)
{
	if (msg->dma_addr == 0) {
		msg_buf_free(msg);
	} else {
		msg_dma_free(msg);
	}
}

#else

static struct rhac_msg* msg_alloc(uint64_t dst_id)
{
  struct rhac_msg *msg = msg_buf_alloc();
  RHAC_ASSERT(!IS_ERR_OR_NULL(msg));
  return msg;
}

static void msg_free(struct rhac_msg *msg)
{
		msg_buf_free(msg);
}

#endif

static int enqueue_work(int type, struct work_struct *work)
{
  int err;
  err = queue_work(get_queue(type), work);
  RHAC_ASSERT(err != 0);
  return err != 0;
}

static void do_post_send(struct work_struct *work, bool fence)
{
  struct rhac_msg *msg = container_of(work, struct rhac_msg, work);
  atomic_inc(&send_msg_cnt[msg->msg_type]);

  int err;

  int signal;

#ifdef RHAC_RDMA_SEND_INLINE
  signal = atomic_inc_return(&total_cnt) % 512 == 0;
#else
  singal = true;
#endif
  // FIXME
  signal = true;

  msg->cqe.done = send_done;

retry:
  err = rhac_rdma_post_send(
#ifdef RHAC_RDMA_SEND_INLINE
      (uint64_t) msg,
#else
      msg->dma_addr,
#endif
      // FIXME
      //msg_size[msg->msg_type],
      sizeof(struct rhac_msg_core),
      msg->dst_id,
      &msg->cqe,
      signal,
      fence
      );
  RHAC_ASSERT(!err);
  if (err) {
    RHAC_LOG("err");
    signal = true;
    fence = true;
    usleep_range(10, 20);
    goto retry;
  }

#ifdef RHAC_RDMA_SEND_INLINE
  if (!signal) send_done_msg(msg);
#endif
}


static void __send_msg( struct rhac_msg *msg, uint32_t dst_id, bool fence)
{
	msg->dst_id = dst_id;
	if (msg->src_id == dst_id) {
		INIT_WORK(&msg->work, handlers[msg->msg_type]);
    enqueue_work(msg->msg_type, &msg->work);
  } else {
    do_post_send(&msg->work, fence);
  }
}

static void send_msg(struct rhac_msg *msg, uint32_t dst_id)
{
  __send_msg(msg, dst_id, false);
}

static void send_msg_fence(struct rhac_msg *msg, uint32_t dst_id)
{
  __send_msg(msg, dst_id, true);
}

static void post_recv_msg(struct rhac_msg *msg)
{
	int err;
	msg->cqe.done = recv_done;
	err = rhac_rdma_post_recv(msg->dma_addr, sizeof(struct rhac_msg_core), &msg->cqe);
	RHAC_ASSERT(!err);
}

static void wrapup(struct rhac_comm *comm)
{
	rhac_nvidia_pipeline_enqueue(comm, comm->next);
}

//
// MSG POST
// 
int rhac_protocol_post_lock(struct rhac_comm *comm, uint64_t blk_vaddr)
{
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	if (rhac_isr_table_try_local_lock(&ctx->isr_table, blk_vaddr, &comm->local_list)) {
		RHAC_ASSERT(false);
		return 0;
	}

	uint32_t pdsc_owner_id = rhac_pdsc_owner(&ctx->page_table, blk_vaddr);
	struct rhac_msg *msg = msg_alloc(pdsc_owner_id);

	rhac_msg_assemble_lock_req(msg, ctx->node_id, (uint64_t)comm, blk_vaddr);
	send_msg(msg, pdsc_owner_id);
	return 0;
}

int rhac_protocol_post_unlock(struct rhac_comm *comm, uint64_t blk_vaddr)
{
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	struct list_head *list;


	if ((list = rhac_isr_table_local_unlock(&ctx->isr_table, blk_vaddr)) != NULL) {
		struct rhac_comm *waitee = container_of(list, struct rhac_comm, local_list);
		rhac_nvidia_pipeline_enqueue(waitee, waitee->next);
		return 0;
	}

	uint32_t pdsc_owner_id = rhac_pdsc_owner(&ctx->page_table, blk_vaddr);
	struct rhac_msg *msg = msg_alloc(pdsc_owner_id);
	rhac_msg_assemble_unlock_req(msg, ctx->node_id, (uint64_t)comm, blk_vaddr);
	send_msg(msg, pdsc_owner_id);
	return 0;
}

int rhac_protocol_post_update(struct rhac_comm *comm,
		uint64_t blk_vaddr,
		const unsigned long *pagemask,
		const unsigned long *protmask,
		const unsigned long *atommask)
{
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	uint32_t pdsc_owner_id = rhac_pdsc_owner(&ctx->page_table, blk_vaddr);

	//rhac_nvidia_mm_populate(blk_vaddr, pagemask);

	struct rhac_msg *msg = msg_alloc(pdsc_owner_id);
	rhac_msg_assemble_update_req(msg, ctx->node_id, (uint64_t)comm,
			blk_vaddr, pagemask, protmask, atommask);

	send_msg(msg, pdsc_owner_id);
	return 0;
}

int rhac_protocol_post_inv(struct rhac_comm *comm, uint64_t blk_vaddr,
		const unsigned long *invmask, uint32_t dst_id)
{
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	struct rhac_msg *msg = msg_alloc(dst_id);
	rhac_msg_assemble_inv_req(msg, ctx->node_id, (uint64_t)comm, blk_vaddr, invmask);
	send_msg(msg, dst_id);
	return 0;
}

//
// MSG HANDLERS
//
static void rhac_protocol_process_lock_req(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	bool locked;

	locked = rhac_pdsc_lock_blk_or_append(&ctx->page_table, msg->blk_vaddr, &msg->list);

	if (locked) {
		uint32_t dst_id = msg->src_id;
		uint64_t blk_vaddr = msg->blk_vaddr;
		uint64_t tag = msg->tag;
		struct rhac_msg *rsp = msg;
		if (msg->src_id != ctx->node_id) {
			rsp = msg_alloc(dst_id);
			msg_free(msg);
		}
		rhac_msg_assemble_lock_rsp(rsp, ctx->node_id, tag, blk_vaddr);
		send_msg(rsp, dst_id);
	}
}

static void rhac_protocol_process_lock_rsp(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);
	struct rhac_comm *comm = msg_to_comm(msg);

	msg_free(msg);
	rhac_nvidia_pipeline_enqueue(comm, comm->next);
}

static void rhac_protocol_process_unlock_req(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);
	struct rhac_ctx *ctx = rhac_ctx_get_global();

	uint64_t blk_vaddr = msg->blk_vaddr;
	msg_free(msg);

	struct list_head *list = rhac_pdsc_unlock_blk_and_return_waitee(&ctx->page_table, blk_vaddr);
	if (list) {
		struct rhac_msg *waitee = container_of(list, struct rhac_msg, list);
		struct rhac_msg *rsp = waitee;
		uint64_t tag = waitee->tag;
		uint32_t dst_id = waitee->src_id;
      uint64_t blk_vaddr = waitee->blk_vaddr;
		if (waitee->src_id != ctx->node_id) {
			rsp = msg_alloc(waitee->src_id);
			msg_free(waitee);
		}

		rhac_msg_assemble_lock_rsp(rsp, ctx->node_id, tag, blk_vaddr);
		send_msg(rsp, dst_id);
	}
}

static void rhac_protocol_process_update_req(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	bool is_local_msg = (msg->src_id == ctx->node_id);
	struct rhac_isr_block *isr_blk = rhac_isr_table_blk_find(&ctx->isr_table, msg->blk_vaddr);

	bool has_another_owner = true;
	bool require_unlock = false;
	rhac_pdsc_update_blk(&ctx->page_table,
			msg->blk_vaddr,
			msg->page_mask,
			msg->prot_mask,
			msg->atom_mask,
			msg->src_id,
			isr_blk->owner_ids,
			&has_another_owner,
			&require_unlock
			);


	if (is_local_msg && !has_another_owner) {
		struct rhac_comm *comm = msg_to_comm(msg);
		atomic_set(&comm->req_unlock, require_unlock);
		wrapup(comm);
	  goto out;
	}


	DECLARE_BITMAP(resident_mask, RHAC_MAX_NODES);
	bitmap_zero(resident_mask, RHAC_MAX_NODES);

	int owner_id, i;
	for_each_set_bit(i, msg->page_mask, RHAC_PDSC_PER_PBLK) {
		set_bit(isr_blk->owner_ids[i], resident_mask);
	}

	int nnodes = bitmap_weight(resident_mask, RHAC_MAX_NODES);
	for_each_set_bit(owner_id, resident_mask, RHAC_MAX_NODES) {
		struct rhac_msg *rsp = msg_alloc(owner_id);

		bitmap_zero(rsp->page_mask, RHAC_PDSC_PER_PBLK);
		for_each_set_bit(i, msg->page_mask, RHAC_PDSC_PER_PBLK) {
			if (isr_blk->owner_ids[i] == owner_id) {
				set_bit(i, rsp->page_mask);
			}
		}
		bitmap_and(rsp->prot_mask, rsp->page_mask, msg->prot_mask, RHAC_PDSC_PER_PBLK);
		bitmap_and(rsp->atom_mask, rsp->page_mask, msg->atom_mask, RHAC_PDSC_PER_PBLK);
		rhac_msg_assemble_update_rsp(rsp, ctx->node_id, msg->tag, msg->blk_vaddr, owner_id, nnodes, require_unlock);
		send_msg(rsp, msg->src_id);
	}

out:
  /*
  if (!require_unlock) {
    uint64_t blk_vaddr = msg->blk_vaddr;
    uint64_t tag = msg->tag;
    rhac_msg_assemble_unlock_req(msg, ctx->node_id, tag, blk_vaddr);
    send_msg(msg, ctx->node_id);
    } else {
    msg_free(msg);
    }
   */
  msg_free(msg);
}

static void rhac_protocol_process_update_rsp(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	struct rhac_comm *comm = msg_to_comm(msg);

	atomic_cmpxchg(&comm->posted, 0, msg->num);

	atomic_set(&comm->req_unlock, msg->require_unlock);
	if (msg->owner_id == ctx->node_id) {
		  if (rhac_comm_unpost(comm) == 0) {
        wrapup(comm);
      }
      goto out;
  }


	unsigned long flags;
	spin_lock_irqsave(&comm->lock, flags);
	bitmap_or(comm->copy_mask, comm->copy_mask, msg->page_mask, RHAC_PDSC_PER_PBLK);
	spin_unlock_irqrestore(&comm->lock, flags);

	struct rhac_msg *rsp = msg_alloc(msg->owner_id);
	rhac_msg_assemble_map_req(rsp, ctx->node_id, (uint64_t)comm, msg->blk_vaddr, msg->page_mask, msg->prot_mask, msg->atom_mask);
	send_msg(rsp, msg->owner_id);

out:
	msg_free(msg);
}

static void rhac_protocol_process_map_req(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);

	int err;

	rhac_nvidia_mm_lock_blk(msg->blk_vaddr);

	err = rhac_nvidia_mm_disable_write_async(msg->blk_vaddr, msg->prot_mask);
	RHAC_ASSERT(!err);

	err = rhac_nvidia_mm_stage_to_cpu_async(msg->blk_vaddr, msg->page_mask);
	RHAC_ASSERT(!err);

	INIT_WORK(&msg->work, rhac_protocol_process_map_req_copy);
	err = enqueue_work(msg->msg_type, &msg->work);
	RHAC_ASSERT(err);
}

static void rhac_protocol_process_map_req_copy(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	struct rhac_msg *rsp = msg_alloc(msg->src_id);

	int npages = bitmap_weight(msg->page_mask, RHAC_PDSC_PER_PBLK);

	void *buf;
	uint64_t dma_addr;
	buf = rdma_send_buffer_alloc(npages, &dma_addr);
	RHAC_ASSERT(buf);

	rhac_nvidia_mm_copy_to_buf(msg->blk_vaddr, msg->page_mask, buf);

	rhac_msg_assemble_map_rsp(rsp, ctx->node_id, msg->tag, msg->blk_vaddr, msg->page_mask, dma_addr, (uint64_t)buf);
	send_msg(rsp, msg->src_id);
	msg_free(msg);
}

static void rhac_protocol_process_map_rsp(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	struct rhac_comm *comm = msg_to_comm(msg);
	int err = 0;

	uint64_t blk_vaddr = msg->blk_vaddr;


	// 1. RDMA READ
	int npages = bitmap_weight(msg->page_mask, RHAC_PDSC_PER_PBLK);
	uint64_t dma_addr = 0;
	void *buf = rdma_recv_buffer_alloc(npages, &dma_addr);
	RHAC_ASSERT(buf && !dma_mapping_error(rhac_rdma_device(), dma_addr));

  atomic_inc(&buffer_read_cnt[msg->src_id][get_buffer_idx(npages)]);
  atomic_add(npages, &buffer_read_npages[msg->src_id]);
	rhac_rdma_sync(dma_addr, npages * PAGE_SIZE, DMA_TO_DEVICE);
  bool signal = true;

  err = rhac_rdma_read(dma_addr, msg->raddr, npages * PAGE_SIZE, msg->src_id, &rdma_cqe, signal);
  RHAC_ASSERT(!err);
	if (err) {
		rhac_comm_fail(comm, -ENOMEM);
		rhac_comm_free(comm);
		return;
	}

	// 2. SEND UNMAP_REQ
	struct rhac_msg *rsp = msg_alloc(msg->src_id);
	rhac_msg_assemble_unmap_req(rsp, ctx->node_id, (uint64_t)comm, blk_vaddr, msg->page_mask, msg->raddr, msg->buf_addr, dma_addr, (uint64_t)buf);
	send_msg_fence(rsp, msg->src_id);

	msg_free(msg);

}

static void post_unmap_done(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);
	struct rhac_comm *comm = msg_to_comm(msg);


  void *buf = (void*)msg->__buf_addr;
  uint64_t dma_addr = msg->__raddr;


	int npages = bitmap_weight(msg->page_mask, RHAC_PDSC_PER_PBLK);
	rhac_rdma_sync(dma_addr, npages * PAGE_SIZE, DMA_FROM_DEVICE);

	rhac_nvidia_mm_copy_from_buf(msg->blk_vaddr, msg->page_mask, buf);

	rdma_buffer_free(npages, buf, dma_addr);

	if (rhac_comm_unpost(comm) == 0) {
		wrapup(comm);
	}

	msg_free(msg);
}

static void rhac_protocol_process_unmap_req(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);

	int npages = bitmap_weight(msg->page_mask, RHAC_PDSC_PER_PBLK);
	rdma_buffer_free(npages, (void*)msg->buf_addr, msg->raddr);

	rhac_nvidia_mm_unlock_blk(msg->blk_vaddr);

	msg_free(msg);
}


static void rhac_protocol_process_inv_req(struct work_struct *work)
{
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);

	int err;
	rhac_nvidia_mm_lock_blk(msg->blk_vaddr);

	err = rhac_nvidia_mm_inv(msg->blk_vaddr, msg->page_mask);
	RHAC_ASSERT(!err);

	rhac_nvidia_mm_unlock_blk(msg->blk_vaddr);

	struct rhac_msg *rsp = msg;
	uint64_t tag = msg->tag;
	uint64_t blk_vaddr = msg->blk_vaddr;
	uint32_t dst_id = msg->src_id;
	if (msg->src_id != ctx->node_id) {
		rsp = msg_alloc(msg->src_id);
		msg_free(msg);
	}

	rhac_msg_assemble_inv_rsp(rsp, ctx->node_id, tag, blk_vaddr);
	send_msg(rsp, dst_id);

}

static void rhac_protocol_process_inv_rsp(struct work_struct *work)
{
	struct rhac_msg *msg = container_of(work, struct rhac_msg, work);

	struct rhac_comm *comm = msg_to_comm(msg);
	if (atomic_dec_return(&comm->cnt) == 0) {
    wake_up_interruptible_sync(&comm->wait_queue);
  }
	msg_free(msg);
}

static void recv_done(struct ib_cq *cq, struct ib_wc *wc)
{
	RHAC_ASSERT(wc->status == IB_WC_SUCCESS);
	RHAC_ASSERT(wc->opcode == IB_WC_RECV);

	struct rhac_msg *msg = container_of(wc->wr_cqe, struct rhac_msg, cqe);
	rhac_msg_print(msg);

	struct rhac_msg *tmp = msg_buf_alloc();
	memcpy(tmp, msg, msg_size[msg->msg_type]);
	//memcpy(tmp, msg, sizeof(struct rhac_msg_core));
	post_recv_msg(msg);

	atomic_inc(&recv_msg_cnt[tmp->msg_type]);
	INIT_WORK(&tmp->work, handlers[tmp->msg_type]);
  enqueue_work(tmp->msg_type, &tmp->work);
}

static void rdma_done(struct ib_cq *cq, struct ib_wc *wc)
{
	RHAC_ASSERT(wc->status == IB_WC_SUCCESS);
	RHAC_ASSERT(wc->opcode == IB_WC_RDMA_READ);
}

static void send_done_msg(struct rhac_msg *msg)
{
	if (msg->msg_type == RHAC_MSG_UNMAP_REQ) {
		int err;
		INIT_WORK(&msg->work, post_unmap_done);
    err = enqueue_work(msg->msg_type, &msg->work);
    RHAC_ASSERT(err);
	} else {
		msg_free(msg);
	}
}

static void send_done(struct ib_cq *cq, struct ib_wc *wc)
{
	RHAC_ASSERT(wc->status == IB_WC_SUCCESS);
	if (wc->status != IB_WC_SUCCESS)
	  RHAC_LOG("STATUS: %d", wc->status);
	RHAC_ASSERT(wc->opcode == IB_WC_SEND);

	struct rhac_msg *msg = container_of(wc->wr_cqe, struct rhac_msg, cqe);
	send_done_msg(msg);
}

int rhac_protocol_init(void)
{
  rdma_cqe.done = rdma_done;

	rhac_comm_init();

  if (rhac_ctx_get_global()->num_nodes == 1)
    return 0;

	int i, err;
	uint64_t dma_addr;
	//RHAC_LOG("MSG SIZE: %lu", sizeof(struct rhac_msg_core));

	rdma_msg_pool = dma_pool_create("rhac_rdma_pool", rhac_rdma_device(),
			sizeof(struct rhac_msg), 0, 0);
	if (!rdma_msg_pool) {
		return -ENOMEM;
	}

	msg_cache = KMEM_CACHE(rhac_msg, SLAB_HWCACHE_ALIGN);
	if (!msg_cache) return -EINVAL;

	for (i = 0; i < RHAC_BUFFER_NUM_POOLS; i++) {
		buffer_pool[i] = dma_pool_create("rhac_buffer_pool", rhac_rdma_device(), 1 << (i + PAGE_SHIFT), 0, 0);
		if (!buffer_pool[i]) {
			return -ENOMEM;
		}
	}

  /*
  rdma_buffer_pool[0] = KMEM_CACHE(rdma_buffer_12, SLAB_HWCACHE_ALIGN);
  rdma_buffer_pool[1] = KMEM_CACHE(rdma_buffer_13, SLAB_HWCACHE_ALIGN);
  rdma_buffer_pool[2] = KMEM_CACHE(rdma_buffer_14, SLAB_HWCACHE_ALIGN);
  rdma_buffer_pool[3] = KMEM_CACHE(rdma_buffer_15, SLAB_HWCACHE_ALIGN);
  rdma_buffer_pool[4] = KMEM_CACHE(rdma_buffer_16, SLAB_HWCACHE_ALIGN);
  rdma_buffer_pool[5] = KMEM_CACHE(rdma_buffer_17, SLAB_HWCACHE_ALIGN);
  rdma_buffer_pool[6] = KMEM_CACHE(rdma_buffer_18, SLAB_HWCACHE_ALIGN);
  rdma_buffer_pool[7] = KMEM_CACHE(rdma_buffer_19, SLAB_HWCACHE_ALIGN);
  rdma_buffer_pool[8] = KMEM_CACHE(rdma_buffer_20, SLAB_HWCACHE_ALIGN);
  rdma_buffer_pool[9] = KMEM_CACHE(rdma_buffer_21, SLAB_HWCACHE_ALIGN);
  if (!rdma_buffer_pool[0]) { return -ENOMEM; }
  if (!rdma_buffer_pool[1]) { return -ENOMEM; }
  if (!rdma_buffer_pool[2]) { return -ENOMEM; }
  if (!rdma_buffer_pool[3]) { return -ENOMEM; }
  if (!rdma_buffer_pool[4]) { return -ENOMEM; }
  if (!rdma_buffer_pool[5]) { return -ENOMEM; }
  if (!rdma_buffer_pool[6]) { return -ENOMEM; }
  if (!rdma_buffer_pool[7]) { return -ENOMEM; }
  if (!rdma_buffer_pool[8]) { return -ENOMEM; }
  if (!rdma_buffer_pool[9]) { return -ENOMEM; }
  */

	atomic_set(&total_cnt, 0);

  int j;
  for (j = 0; j < RHAC_MAX_NODES; j++) {
    for (i = 0; i < RHAC_BUFFER_NUM_POOLS; i++) {
      atomic_set(&buffer_read_cnt[j][i], 0);
    }
    atomic_set(&buffer_read_npages[j], 0);
  }

	for (i = 0; i < RHAC_MSG_MAX; i++) {
		atomic_set(&recv_msg_cnt[i], 0);
		atomic_set(&send_msg_cnt[i], 0);
	}

	for (i = 0; i < RHAC_PROT_NUM_RECV; i++) {
		recv_msgs[i] = dma_pool_alloc(rdma_msg_pool, GFP_KERNEL, &dma_addr);
		RHAC_ASSERT(recv_msgs[i] && !dma_mapping_error(rhac_rdma_device(), dma_addr));
		recv_msgs[i]->dma_addr = dma_addr;
		recv_msgs[i]->cqe.done = recv_done;
		err = rhac_rdma_post_recv(recv_msgs[i]->dma_addr, sizeof(struct rhac_msg_core), &recv_msgs[i]->cqe);
		if (err) goto FAIL;

	}

	for (i = 0; i < RHAC_MSG_MAX; i++) {
		//workqueue[i] = create_workqueue("rhac protocol");
		int max_active = 20;
		workqueue[i] = alloc_workqueue("rhac-protocol %d",  WQ_UNBOUND | WQ_MEM_RECLAIM, max_active, i);
	}

	return 0;

FAIL:
	rhac_protocol_deinit();
	return -EINVAL;
}

void rhac_protocol_deinit(void)
{
	rhac_protocol_flush();

	int i;
	atomic_set(&total_cnt, 0);

	int send_cnt = 0, recv_cnt = 0;
	for (i = 0; i < RHAC_MSG_MAX; i++) {
		int scnt = atomic_read(&send_msg_cnt[i]);
		int rcnt = atomic_read(&recv_msg_cnt[i]);
		atomic_set(&recv_msg_cnt[i], 0);
		atomic_set(&send_msg_cnt[i], 0);

		RHAC_LOG("MSG %s - recv: %u send: %u", msg_strs[i], scnt, rcnt);

		send_cnt += scnt;
		recv_cnt += rcnt;

	}
	RHAC_LOG("TOTAL MSG SEND CNT: %d, RECV CNT: %d", send_cnt, recv_cnt);

	int rdma_cnt = 0;
	uint64_t size = 0;
  int j;
  for (j = 0; j < rhac_ctx_get_global()->num_nodes; j++) {
    uint64_t read_size = 0;
    for (i = 0; i < RHAC_BUFFER_NUM_POOLS; i++) {
      rdma_cnt += atomic_read(&buffer_read_cnt[j][i]);
      atomic_set(&buffer_read_cnt[j][i], 0);
    }
    read_size = atomic_read(&buffer_read_npages[j]) * PAGE_SIZE;
    atomic_set(&buffer_read_npages[j], 0);
    size += read_size;
    RHAC_LOG("[%d -> %d] RDMA READ %llu MB ", j, rhac_ctx_get_global()->node_id, read_size >> 20);

  }
  RHAC_LOG("TOTAL RDMA READ CNT: %d, READ SIZE: %llu MB", rdma_cnt, size >> 20);


	rhac_comm_deinit();

	for (i = 0; i < RHAC_MSG_MAX; i++) {
		if (workqueue[i]) {
			destroy_workqueue(workqueue[i]);
		}
		workqueue[i] = NULL;
	}

  /*
  if (send_queue)
    destroy_workqueue(send_queue);
  send_queue = NULL;
  */

	for (i = 0; i < RHAC_PROT_NUM_RECV; i++) {
		if (recv_msgs[i])  {
			dma_pool_free(rdma_msg_pool, recv_msgs[i], recv_msgs[i]->dma_addr);
			recv_msgs[i] = NULL;
		}
	}

	if (rdma_msg_pool)
		dma_pool_destroy(rdma_msg_pool);
	rdma_msg_pool = NULL;

	for (i = 0; i < RHAC_BUFFER_NUM_POOLS; i++) {
		if (buffer_pool[i])
			dma_pool_destroy(buffer_pool[i]);
		buffer_pool[i] = NULL;
	}

	/*
	for (i = 0; i < RHAC_BUFFER_NUM_POOLS; i++) {
		if (rdma_buffer_pool[i]) kmem_cache_destroy(rdma_buffer_pool[i]);
		rdma_buffer_pool[i] = NULL;
	}
	*/

	if (msg_cache) kmem_cache_destroy(msg_cache);
	msg_cache = NULL;
}

void rhac_protocol_flush(void)
{
	int i;
	for (i = 0; i < RHAC_MSG_MAX; i++) {
		if (workqueue[i]) {
			flush_workqueue(workqueue[i]);
		}
	}
	/*
	if (send_queue)
	  flush_workqueue(send_queue);
  */

	rhac_nvidia_pipeline_flush();
}
