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

#ifndef __RHAC_MSG_H__
#define __RHAC_MSG_H__

#include "rhac_types.h"
#include "rhac_config.h"
#include <rdma/ib_verbs.h>

enum rhac_msg_type {
	RHAC_MSG_BUG = 0,

	RHAC_MSG_LOCK_REQ   ,
	RHAC_MSG_LOCK_RSP   ,
	RHAC_MSG_UNLOCK_REQ ,
	RHAC_MSG_UPDATE_REQ ,
	RHAC_MSG_UPDATE_RSP ,
	RHAC_MSG_MAP_REQ    ,
	RHAC_MSG_MAP_RSP    ,
	RHAC_MSG_UNMAP_REQ  ,
	RHAC_MSG_INV_REQ    ,
	RHAC_MSG_INV_RSP    ,
	RHAC_MSG_MAX        ,
};

struct rhac_msg_lock_req {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;
};

struct rhac_msg_lock_rsp {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;
};

struct rhac_msg_unlock_req {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;
};

struct rhac_msg_update_req {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;

	DECLARE_BITMAP(page_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(prot_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(atom_mask, RHAC_PDSC_PER_PBLK);
};

struct rhac_msg_update_rsp {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;

	DECLARE_BITMAP(page_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(prot_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(atom_mask, RHAC_PDSC_PER_PBLK);

	uint32_t owner_id;
	uint32_t num;
	uint8_t require_unlock;
};

struct rhac_msg_map_req {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;

	DECLARE_BITMAP(page_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(prot_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(atom_mask, RHAC_PDSC_PER_PBLK);
};

struct rhac_msg_map_rsp {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;

	DECLARE_BITMAP(page_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(prot_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(atom_mask, RHAC_PDSC_PER_PBLK);

	uint64_t raddr;
	uint64_t buf_addr;

	//char payload[2*1024*1024];
};

struct rhac_msg_unmap_req {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;

	DECLARE_BITMAP(page_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(prot_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(atom_mask, RHAC_PDSC_PER_PBLK);

	uint64_t raddr;
	uint64_t buf_addr;
  uint64_t __buf_addr; /* for internal use*/
  uint64_t __raddr; /* for internal use*/
};

struct rhac_msg_inv_req {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;

	DECLARE_BITMAP(page_mask, RHAC_PDSC_PER_PBLK);
};

struct rhac_msg_inv_rsp {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;
};

struct rhac_msg_core {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;

	DECLARE_BITMAP(page_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(prot_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(atom_mask, RHAC_PDSC_PER_PBLK);

	union {
		struct {
			uint64_t raddr;
			uint64_t buf_addr;
			uint64_t __buf_addr; /* for internal use*/
			uint64_t __raddr; /* for internal use*/
		};
		struct {
			uint32_t owner_id;
			uint32_t num;
      uint8_t require_unlock;
		};
	};

	//char payload[2*1024*1024];
};

struct rhac_msg {
	uint8_t msg_type;
	uint32_t src_id;
	uint64_t tag;
	uint64_t blk_vaddr;

	DECLARE_BITMAP(page_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(prot_mask, RHAC_PDSC_PER_PBLK);
	DECLARE_BITMAP(atom_mask, RHAC_PDSC_PER_PBLK);

	union {
		struct {
			uint64_t raddr;
			uint64_t buf_addr;
			uint64_t __buf_addr; /* for internal use*/
			uint64_t __raddr; /* for internal use*/
		};
		struct {
			uint32_t owner_id;
			uint32_t num;
      uint8_t require_unlock;
		};
	};

	//char payload[2*1024*1024];


	// for local management of recv buffers,
	// do not send from here!
	uint64_t dma_addr;
	struct ib_cqe cqe;
	bool is_dma;


	uint32_t dst_id;

	struct work_struct work;
	struct list_head list;
};

inline static void rhac_msg_assemble_lock_req(
		struct rhac_msg *msg,
		uint32_t src_id,
		uint64_t tag,
		uint64_t blk_vaddr)
{
	msg->msg_type = RHAC_MSG_LOCK_REQ;
	msg->src_id = src_id;
	msg->tag = tag;
	msg->blk_vaddr = blk_vaddr;
}

inline static void rhac_msg_assemble_lock_rsp(
		struct rhac_msg *msg,
		uint32_t src_id,
		uint64_t tag,
		uint64_t blk_vaddr)
{
	msg->msg_type = RHAC_MSG_LOCK_RSP;
	msg->src_id = src_id;
	msg->tag = tag;
	msg->blk_vaddr = blk_vaddr;
}

inline static void rhac_msg_assemble_unlock_req(
		struct rhac_msg *msg,
		uint32_t src_id,
		uint64_t tag,
		uint64_t blk_vaddr)
{
	msg->msg_type = RHAC_MSG_UNLOCK_REQ;
	msg->src_id = src_id;
	msg->tag = tag;
	msg->blk_vaddr = blk_vaddr;
}

inline static void rhac_msg_assemble_update_req(
		struct rhac_msg *msg,
		uint32_t src_id,
		uint64_t tag,
		uint64_t blk_vaddr,
		const unsigned long *pagemask,
		const unsigned long *protmask,
		const unsigned long *atommask)
{
	msg->msg_type = RHAC_MSG_UPDATE_REQ;
	msg->src_id = src_id;
	msg->tag = tag;
	msg->blk_vaddr = blk_vaddr;
	bitmap_copy(msg->page_mask, pagemask, RHAC_PDSC_PER_PBLK);
	bitmap_copy(msg->prot_mask, protmask, RHAC_PDSC_PER_PBLK);
	bitmap_copy(msg->atom_mask, atommask, RHAC_PDSC_PER_PBLK);
}

inline static void rhac_msg_assemble_update_rsp(
		struct rhac_msg *msg,
		uint32_t src_id,
		uint64_t tag,
		uint64_t blk_vaddr,
		uint32_t owner_id,
		uint32_t num,
		uint8_t require_unlock
		)
{
	msg->msg_type = RHAC_MSG_UPDATE_RSP;
	msg->src_id = src_id;
	msg->tag = tag;
	msg->blk_vaddr = blk_vaddr;
	msg->owner_id = owner_id;
	msg->num = num;
	msg->require_unlock = require_unlock;
}

inline static void rhac_msg_assemble_map_req(
		struct rhac_msg *msg,
		uint32_t src_id,
		uint64_t tag,
		uint64_t blk_vaddr,
		const unsigned long *pagemask,
		const unsigned long *protmask,
		const unsigned long *atommask)
{
	msg->msg_type = RHAC_MSG_MAP_REQ;
	msg->src_id = src_id;
	msg->tag = tag;
	msg->blk_vaddr = blk_vaddr;
	bitmap_copy(msg->page_mask, pagemask, RHAC_PDSC_PER_PBLK);
	bitmap_copy(msg->prot_mask, protmask, RHAC_PDSC_PER_PBLK);
	bitmap_copy(msg->atom_mask, atommask, RHAC_PDSC_PER_PBLK);
}

inline static void rhac_msg_assemble_map_rsp(
		struct rhac_msg *msg,
		uint32_t src_id,
		uint64_t tag,
		uint64_t blk_vaddr,
		const unsigned long *pagemask,
		uint64_t raddr,
		uint64_t buf_addr)
{
	msg->msg_type = RHAC_MSG_MAP_RSP;
	msg->src_id = src_id;
	msg->tag = tag;
	msg->blk_vaddr = blk_vaddr;
	bitmap_copy(msg->page_mask, pagemask, RHAC_PDSC_PER_PBLK);
	msg->raddr = raddr;
	msg->buf_addr = buf_addr;
}

inline static void rhac_msg_assemble_unmap_req(
		struct rhac_msg *msg,
		uint32_t src_id,
		uint64_t tag,
		uint64_t blk_vaddr,
		const unsigned long *pagemask,
		uint64_t raddr,
		uint64_t buf_addr,
		uint64_t __raddr,
		uint64_t __buf_addr
		)

{
	msg->msg_type = RHAC_MSG_UNMAP_REQ;
	msg->src_id = src_id;
	msg->tag = tag;
	msg->blk_vaddr = blk_vaddr;
	bitmap_copy(msg->page_mask, pagemask, RHAC_PDSC_PER_PBLK);
	msg->raddr = raddr;
	msg->buf_addr = buf_addr;
	msg->__raddr = __raddr;
	msg->__buf_addr = __buf_addr;
}

inline static void rhac_msg_assemble_inv_req(
		struct rhac_msg *msg,
		uint32_t src_id,
		uint64_t tag,
		uint64_t blk_vaddr,
		const unsigned long *invmask)
{
	msg->msg_type = RHAC_MSG_INV_REQ;
	msg->src_id = src_id;
	msg->tag = tag;
	msg->blk_vaddr = blk_vaddr;
	bitmap_copy(msg->page_mask, invmask, RHAC_PDSC_PER_PBLK);
}

inline static void rhac_msg_assemble_inv_rsp(
		struct rhac_msg *msg,
		uint32_t src_id,
		uint64_t tag,
		uint64_t blk_vaddr)
{
	msg->msg_type = RHAC_MSG_INV_RSP;
	msg->src_id = src_id;
	msg->tag = tag;
	msg->blk_vaddr = blk_vaddr;
}

void rhac_msg_print(struct rhac_msg *msg);
#endif //__RHAC_MSG_H__
