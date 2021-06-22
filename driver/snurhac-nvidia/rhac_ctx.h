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

#ifndef __RHAC_CTX_H__
#define __RHAC_CTX_H__

#include <linux/types.h>
#include <linux/rbtree.h>

#include "rhac_config.h"
#include "rhac_pdsc.h"
#include "rhac_rdma.h"
#include "rhac_protocol.h"
#include "rhac_isr_table.h"


#define rhac_ctx_from_filp(filp) ((struct rhac_ctx*) filp->private_data)

struct file;

struct rhac_ctx {
	struct file *filp;

	uint64_t vaddr_base;
	uint32_t node_id;
	uint32_t num_nodes;
	uint32_t num_local_gpus;

	struct mm_struct *mm;
	atomic_t refcnt;

	//
	struct rhac_pdsc_table page_table;

	//
	struct rhac_isr_table isr_table;

};



struct rhac_ctx* rhac_ctx_get_global(void);

struct rhac_ctx* rhac_ctx_create(struct file *filp);
void rhac_ctx_destroy(struct rhac_ctx* ctx);

inline static uint64_t rhac_ctx_get_base(struct rhac_ctx *ctx)
{
	return ctx->vaddr_base;
}

inline static uint32_t rhac_ctx_get_num_nodes(struct rhac_ctx *ctx)
{
	return ctx->num_nodes;
}

int rhac_ctx_reserve(struct rhac_ctx *ctx, uint64_t capacity);
struct rhac_ctx* rhac_ctx_find(struct mm_struct *mm);

#endif //__RHAC_CTX_H__
