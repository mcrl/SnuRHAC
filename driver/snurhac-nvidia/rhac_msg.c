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
#include "rhac_msg.h"
#include "rhac_nvidia_helpers.h"

void rhac_msg_print(struct rhac_msg *msg)
{
  return;
	static const char *strs[] = {
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

	if (msg->tag == 0)
		msg->tag = 0xffffffffffffffff;

	switch (msg->msg_type) {
		case RHAC_MSG_LOCK_REQ:
			RHAC_LOG("| %s | node_id: %u | tag: %llx | blk_vaddr: %llx", strs[msg->msg_type], msg->src_id, msg->tag, msg->blk_vaddr);
			break;
		case RHAC_MSG_LOCK_RSP:
			RHAC_LOG("| %s | node_id: %u | tag: %llx | blk_vaddr: %llx", strs[msg->msg_type], msg->src_id, msg->tag, msg->blk_vaddr);
			break;
		case RHAC_MSG_UNLOCK_REQ:
			RHAC_LOG("| %s | node_id: %u | tag: %llx | blk_vaddr: %llx", strs[msg->msg_type], msg->src_id, msg->tag, msg->blk_vaddr);
			break;
		case RHAC_MSG_UPDATE_REQ:
			RHAC_LOG("| %s | node_id: %u | tag: %llx | blk_vaddr: %llx", strs[msg->msg_type], msg->src_id, msg->tag, msg->blk_vaddr);
			__rhac_print_page_mask("page_mask", (uvm_page_mask_t*) msg->page_mask);
			__rhac_print_page_mask("prot_mask", (uvm_page_mask_t*) msg->prot_mask);
			//__rhac_print_page_mask("atom_mask", (uvm_page_mask_t*) msg->atom_mask);
			break;
		case RHAC_MSG_UPDATE_RSP:
			RHAC_LOG("| %s | node_id: %u | tag: %llx | blk_vaddr: %llx | owner_id: %u | num: %u | require_unlock: %u",
					strs[msg->msg_type], msg->src_id, msg->tag, msg->blk_vaddr, msg->owner_id, msg->num, msg->require_unlock);
			__rhac_print_page_mask("page_mask", (uvm_page_mask_t*) msg->page_mask);
			__rhac_print_page_mask("prot_mask", (uvm_page_mask_t*) msg->prot_mask);
			//__rhac_print_page_mask("atom_mask", (uvm_page_mask_t*) msg->atom_mask);
			break;
		case RHAC_MSG_MAP_REQ:
			RHAC_LOG("| %s | node_id: %u | tag: %llx | blk_vaddr: %llx", strs[msg->msg_type], msg->src_id, msg->tag, msg->blk_vaddr);
			__rhac_print_page_mask("page_mask", (uvm_page_mask_t*) msg->page_mask);
			__rhac_print_page_mask("prot_mask", (uvm_page_mask_t*) msg->prot_mask);
			//__rhac_print_page_mask("atom_mask", (uvm_page_mask_t*) msg->atom_mask);
			break;
		case RHAC_MSG_MAP_RSP:
			RHAC_LOG("| %s | node_id: %u | tag: %llx | blk_vaddr: %llx | raddr: %llx", strs[msg->msg_type], msg->src_id, msg->tag, msg->blk_vaddr, msg->raddr);
			__rhac_print_page_mask("page_mask", (uvm_page_mask_t*) msg->page_mask);
			break;
		case RHAC_MSG_UNMAP_REQ:
			RHAC_LOG("| %s | node_id: %u | tag: %llx | blk_vaddr: %llx", strs[msg->msg_type], msg->src_id, msg->tag, msg->blk_vaddr);
			__rhac_print_page_mask("page_mask", (uvm_page_mask_t*) msg->page_mask);
			break;
		case RHAC_MSG_INV_REQ:
			RHAC_LOG("| %s | node_id: %u | tag: %llx | blk_vaddr: %llx", strs[msg->msg_type], msg->src_id, msg->tag, msg->blk_vaddr);
			__rhac_print_page_mask("page_mask", (uvm_page_mask_t*) msg->page_mask);
			break;
		case RHAC_MSG_INV_RSP:
			RHAC_LOG("| %s | node_id: %u | tag: %llx | blk_vaddr: %llx", strs[msg->msg_type], msg->src_id, msg->tag, msg->blk_vaddr);
			break;
		default:
			RHAC_LOG("TYPE: %u", msg->msg_type);
			RHAC_BUG();
	}
}
