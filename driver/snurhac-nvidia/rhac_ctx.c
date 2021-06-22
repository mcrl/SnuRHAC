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

#include "rhac_correlator.h"
#include "rhac_ctx.h"
#include "rhac_nvidia.h"
#include "rhac_utils.h"

static struct rhac_ctx global_ctx;

struct rhac_ctx* rhac_ctx_create(struct file *filp)
{
	struct rhac_ctx *ctx = kzalloc(sizeof(struct rhac_ctx), GFP_KERNEL);
	if (!ctx) return NULL;
	ctx->filp = filp;
	atomic_set(&ctx->refcnt, 1);
	return ctx;
}

struct rhac_ctx* rhac_ctx_get_global(void)
{
	return &global_ctx;
}

void rhac_ctx_destroy(struct rhac_ctx* ctx)
{
	//sleep for some delay
	ssleep(2);
	rhac_protocol_flush();


#ifdef RHAC_DYNAMIC_PREFETCH_NON_READONLY
  rhac_destroy_correlator_threads();
  rhac_destroy_correlation_tables();
#endif

	rhac_isr_table_deinit(&ctx->isr_table);
	rhac_pdsc_table_deinit(&ctx->page_table);
	rhac_nvidia_deinit();
	rhac_protocol_deinit();
	rhac_rdma_deinit();
}
