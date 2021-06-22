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

#include "rhac_correlator.h"
#include "rhac_ctx.h"
#include "rhac_iocx.h"
#include "rhac_rdma.h"
#include "rhac_utils.h"
#include "rhac_nvidia.h"
#include "rhac_nvidia_mm.h"
#include "rhac_protocol.h"


int rhac_iocx_init(struct file *filp, rhac_iocx_init_param_t *param)
{
	int err = 0;

	struct rhac_ctx *ctx = rhac_ctx_from_filp(filp);
	RHAC_ASSERT(ctx);

	ctx->vaddr_base = param->vaddr_base;
	ctx->node_id = param->node_id;
	ctx->num_nodes = param->num_nodes;
	ctx->num_local_gpus = param->num_local_gpus;

  if (ctx->num_nodes > 1) {
    err = rhac_rdma_init(ctx->node_id, ctx->num_nodes);
    if (err) RHAC_GOTO_ERR(out, "Cannot initialize rdma");
  }


	// Call protocol init after rdma
	err = rhac_protocol_init();
	if (err) RHAC_GOTO_ERR(fail1, "Cannot initialize protocol handler");

	err = rhac_nvidia_init();
	if (err) RHAC_GOTO_ERR(fail2, "Cannot initialize nvidia isr");

#ifdef RHAC_DYNAMIC_PREFETCH_NON_READONLY
  rhac_create_correlation_tables();
  rhac_create_correlator_threads();
#endif


	RHAC_LOG("IOCX_INIT completed (base: 0x%llx, node_id: %u, num_nodes: %u)",
			ctx->vaddr_base,
			ctx->node_id,
			ctx->num_nodes);

	return 0;

fail2: rhac_protocol_deinit();
fail1: rhac_rdma_deinit();
out:
       return err;
}

int rhac_iocx_reserve(struct file *filp, rhac_iocx_reserve_param_t *param)
{
	int ret;
	struct rhac_ctx *ctx = rhac_ctx_from_filp(filp);

	if (ctx->num_nodes <= 1)
	  return 0;

	ret = rhac_pdsc_table_init(&ctx->page_table, ctx->vaddr_base, param->capacity, ctx->node_id, ctx->num_nodes);
	if (ret) return ret;

	ret = rhac_isr_table_init(&ctx->isr_table, ctx->vaddr_base, param->capacity);
	if (ret) {
		rhac_pdsc_table_deinit(&ctx->page_table);
		return ret;
	}
	RHAC_LOG("IOCX_RESERVED %llu Gbytes", param->capacity >> 30);

	return 0;
}

int rhac_iocx_sync(struct file *filp, rhac_iocx_sync_param_t *param)
{
	int err = 0;
	struct rhac_ctx *ctx = rhac_ctx_from_filp(filp);

	if (ctx->num_nodes > 1) {
    err = rhac_pdsc_inv(&ctx->page_table);
  }

#ifdef RHAC_DYNAMIC_PREFETCH_NON_READONLY
  rhac_clear_correlator_threads();
#endif


	return err;
}

int rhac_iocx_split_va_range(struct file *filp, rhac_iocx_split_va_range_param_t *param)
{
	int err;
	err = rhac_nvidia_mm_split_va_range(param->vaddr, param->length);
	RHAC_LOG("IOCX_SPLIT %llx by %llx", param->vaddr, param->length);
	return err;
}

int rhac_iocx_toggle_dup_flag(struct file *filp, rhac_iocx_toggle_dup_flag_param_t *param)
{
	int err;
	struct rhac_ctx *ctx = rhac_ctx_from_filp(filp);
	err = rhac_nvidia_mm_toggle_dup_flag(param->vaddr, param->size, param->turnon_flag);
	if (err) return err;

	if (ctx->num_nodes <= 1)
	  return 0;

  if (param->turnon_flag) {
    err = rhac_pdsc_set_readonly(&ctx->page_table, param->vaddr, param->size);
  } else {
    err = rhac_pdsc_unset_readonly(&ctx->page_table, param->vaddr, param->size);
    if (err) return err;

    err = rhac_pdsc_inv(&ctx->page_table);
  }
	RHAC_LOG("IOCX_TOGGLE %llx by %llx, %s", param->vaddr, param->size, param->turnon_flag ? "ON" : "OFF");
	return err;
}

int rhac_iocx_prefetch_to_cpu(struct file *filp, rhac_iocx_prefetch_to_cpu_param_t *param)
{
	int err;
	err = rhac_nvidia_mm_prefetch_to_cpu(param->vaddr, param->size, param->device_id, true);
	RHAC_ASSERT(!err);
	//RHAC_LOG("IOCX_PREFETCH_TO_CPU %llx by %llu bytes to %u START (%s)", param->vaddr, param->size, param->device_id+1, param->is_async ? "async" : "sync");
	return err;
}

int rhac_iocx_prefetch_to_gpu(struct file *filp, rhac_iocx_prefetch_to_gpu_param_t *param)
{
	int err;
	err = rhac_nvidia_mm_prefetch_to_gpu(param->vaddr, &param->page_mask, param->device_id);
	RHAC_ASSERT(!err);
	//RHAC_LOG("IOCX_PREFETCH_TO_GPU %llx to %u", param->vaddr, param->device_id+1);
	return err;
}
