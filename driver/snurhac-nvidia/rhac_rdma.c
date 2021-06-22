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

#include <linux/completion.h>
      
#include <linux/inet.h>
#include <linux/dma-mapping.h>
#include <rdma/ib_verbs.h>
#include <rdma/rdma_cm.h>
#include <rdma/mr_pool.h>

#include "nvidia-uvm/uvm8_global.h"

#include "rhac_ctx.h"
#include "rhac_rdma.h"
#include "rhac_utils.h"

//#define RHAC_RDMA_DEBUG

static struct conn_data {
	uint32_t node_id;
	uint32_t rkey;
} conn_data;

static struct rdma_conn_param conn_param = {
	.private_data =  &conn_data,
	.private_data_len = sizeof(struct conn_data),
	.responder_resources = 16,
	.initiator_depth = 16,
	.retry_count = 7,
	.rnr_retry_count = 7,
};

static const char *node_ip_str[] = {
	#include "../snurhac_nodes.config"
};

static struct conn_ctx {
	struct completion done;
	int ret;
} conn_context;

static struct sockaddr node_addrs[RHAC_MAX_NODES];
static struct sockaddr server_addr;

static int CONN_TIMEOUT_MS = 5000;
module_param_named(conn_timout, CONN_TIMEOUT_MS, int, 0);

static int IB_CLIENT_PORT = 10021;

static struct rdma_cm_id *server_cm_id;
static struct rdma_cm_id *conn_cm_id[RHAC_MAX_NODES];
static struct ib_device *ib_device;

static struct ib_pd *pd = NULL;
static struct ib_cq *cq = NULL;
static struct ib_srq *srq = NULL;

static uint32_t global_rkeys[RHAC_MAX_NODES];


int rhac_rdma_map_mr_sg(
    struct scatterlist *sg,
    int sg_nents)
{
  /*
  struct ib_mr *mr = ib_get_dma_mr(pd, IB_ACCESS_REMOTE_READ);
  RHAC_LOG("mr: %llx\n", mr);
  RHAC_LOG("%lu %lu %llu %llu\n", mr->lkey, mr->rkey, mr->iova, mr->length);
  int err = ib_map_mr_sg(mr, sg, sg_nents, NULL, PAGE_SIZE);
  RHAC_LOG("err: %d\n", err);
  return err;
  */
  return 0;
}

static int rhac_rdma_post_send_common(
		uint64_t dma_addr,
		uint64_t size,
		uint32_t dst_id,
		int flag,
		struct ib_cqe *cqe
		)
{
	int err;
	struct ib_qp *qp = conn_cm_id[dst_id]->qp;
	struct ib_sge sge = {
		.addr = dma_addr, 
		.length = size,
		.lkey = qp->pd->local_dma_lkey,
	};

	const struct ib_send_wr *bad_send_wr;
	struct ib_send_wr send_wr = {
    .wr_cqe = cqe,
		.sg_list = &sge,
		.num_sge = 1,
		.opcode = IB_WR_SEND,
		.send_flags = flag,
	};

	err = ib_post_send(qp, &send_wr, &bad_send_wr);
	return err;
}

int rhac_rdma_post_send(
		uint64_t dma_addr,
		uint64_t size,
		uint32_t dst_id,
		struct ib_cqe *cqe,
		bool signal,
		bool fence
		)
{
	return rhac_rdma_post_send_common(
			dma_addr,
			size,
			dst_id,
#ifdef RHAC_RDMA_SEND_INLINE
			IB_SEND_INLINE | 
#endif
      (signal ? IB_SEND_SIGNALED : 0) | 
			(fence ?  (IB_SEND_FENCE | IB_SEND_SIGNALED) : 0)
			| 0,
			cqe
			);
}

int rhac_rdma_post_recv(
		uint64_t dma_addr,
		uint64_t size,
		struct ib_cqe *cqe
		)
{
	struct ib_sge sg_list = {
		.addr = dma_addr,
		.length = size,
		.lkey = srq->pd->local_dma_lkey,
	};

	const struct ib_recv_wr *bad_recv_wr;
	struct ib_recv_wr recv_wr = {
		.wr_cqe = cqe,
		.sg_list = &sg_list,
		.num_sge = 1,
	};

	return ib_post_srq_recv(srq, &recv_wr, &bad_recv_wr); 
}

int rhac_rdma_read(
	uint64_t dma_addr,
	uint64_t raddr,
	uint64_t length, 
	uint64_t dst_id,
	struct ib_cqe *cqe,
	bool signal
)
{
	struct ib_qp *qp = conn_cm_id[dst_id]->qp;

	struct ib_sge sge = {
		.addr = dma_addr,
		.length = length,
		.lkey = qp->pd->local_dma_lkey,
	};

	const struct ib_send_wr *bad_send_wr;
	struct ib_rdma_wr rdma_wr = {
		.wr = {
			.wr_cqe = cqe,
			.sg_list = &sge,
			.num_sge = 1,
			.opcode = IB_WR_RDMA_READ,
			.send_flags = (signal ? IB_SEND_SIGNALED : 0),
		},
		.remote_addr = raddr,
		.rkey = global_rkeys[dst_id],
	};

	int err;

retry:
	err = ib_post_send(qp, &rdma_wr.wr, &bad_send_wr);
  if (err) {
    rdma_wr.wr.send_flags |= IB_SEND_SIGNALED;
    RHAC_ASSERT(false);
    usleep_range(10, 20);
    goto retry;
	}

	return 0;
}

uint64_t rhac_rdma_map_single(void *cpu_addr, size_t size, enum dma_data_direction direction)
{
	return ib_dma_map_single(ib_device, cpu_addr, size, direction);
}

void rhac_rdma_unmap_single(uint64_t daddr, size_t size, enum dma_data_direction direction)
{
	ib_dma_unmap_single(ib_device, daddr, size, direction);
}

uint64_t rhac_rdma_map_page(struct page *page, size_t offset, enum dma_data_direction direction)
{
	return ib_dma_map_page(ib_device, page, offset, PAGE_SIZE, direction);
}

void rhac_rdma_unmap_page(uint64_t daddr, size_t size, enum dma_data_direction direction)
{
	ib_dma_unmap_page(ib_device, daddr, size, direction);
}

int rhac_rdma_map_sg(struct scatterlist *sg, int nents, enum dma_data_direction direction)
{
	return ib_dma_map_sg(ib_device, sg, nents, direction);
}

void rhac_rdma_unmap_sg(struct scatterlist *sg, int nents, enum dma_data_direction direction)
{
	ib_dma_unmap_sg(ib_device, sg, nents, direction);
}

int rhac_rdma_mapping_error(uint64_t addr)
{
	return ib_dma_mapping_error(ib_device, addr);
}

void rhac_rdma_sync_sg(struct scatterlist *sgl, int nents, enum dma_data_direction direction)
{
	int i;
	struct scatterlist *sg;
	for_each_sg(sgl, sg, nents, i) {
		ib_dma_sync_single_for_cpu(ib_device,
				sg_dma_address(sg),
				sg_dma_len(sg),
				direction);
	}
}

void rhac_rdma_sync(uint64_t dma_addr, int len, enum dma_data_direction direction)
{
	ib_dma_sync_single_for_cpu(ib_device, dma_addr, len, direction);
}

static void mock_event_handler(struct ib_event *event, void *ctx)
{
	switch (event->event) {
		case IB_EVENT_CQ_ERR:
			RHAC_LOG("IB_EVENT_CQ_ERR");
			break;
		case IB_EVENT_QP_FATAL:
			RHAC_LOG("IB_EVENT_QP_FATAL");
			break;
		case IB_EVENT_QP_REQ_ERR:
			RHAC_LOG("IB_EVENT_QP_REQ_ERR");
			break;
		case IB_EVENT_QP_ACCESS_ERR:
			RHAC_LOG("IB_EVENT_QP_ACCESS_ERR");
			break;
		case IB_EVENT_COMM_EST:
			RHAC_LOG("IB_EVENT_COMM_EST");
			break;
		case IB_EVENT_SQ_DRAINED:
			RHAC_LOG("IB_EVENT_SQ_DRAINED");
			break;
		case IB_EVENT_PATH_MIG:
			RHAC_LOG("IB_EVENT_PATH_MIG");
			break;
		case IB_EVENT_PATH_MIG_ERR:
			RHAC_LOG("IB_EVENT_PATH_MIG_ERR");
			break;
		case IB_EVENT_DEVICE_FATAL:
			RHAC_LOG("IB_EVENT_DEVICE_FATAL");
			break;
		case IB_EVENT_PORT_ACTIVE:
			RHAC_LOG("IB_EVENT_PORT_ACTIVE");
			break;
		case IB_EVENT_PORT_ERR:
			RHAC_LOG("IB_EVENT_PORT_ERR");
			break;
		case IB_EVENT_LID_CHANGE:
			RHAC_LOG("IB_EVENT_LID_CHANGE");
			break;
		case IB_EVENT_PKEY_CHANGE:
			RHAC_LOG("IB_EVENT_PKEY_CHANGE");
			break;
		case IB_EVENT_SM_CHANGE:
			RHAC_LOG("IB_EVENT_SM_CHANGE");
			break;
		case IB_EVENT_SRQ_ERR:
			RHAC_LOG("IB_EVENT_SRQ_ERR");
			break;
		case IB_EVENT_SRQ_LIMIT_REACHED:
			RHAC_LOG("IB_EVENT_SRQ_LIMIT_REACHED");
			break;
		case IB_EVENT_QP_LAST_WQE_REACHED:
			RHAC_LOG("IB_EVENT_QP_LAST_WQE_REACHED");
			break;
		case IB_EVENT_CLIENT_REREGISTER:
			RHAC_LOG("IB_EVENT_CLIENT_REREGISTER");
			break;
		case IB_EVENT_GID_CHANGE:
			RHAC_LOG("IB_EVENT_GID_CHANGE");
			break;
		case IB_EVENT_WQ_FATAL:
			RHAC_LOG("IB_EVENT_WQ_FATAL");
			break;
		default: RHAC_LOG("UNKNOWN CASE FAIL"); break;
	}
}

static int create_qp(struct rdma_cm_id *cm_id)
{
	int err = 0;
	struct ib_qp_init_attr qp_init_attr = {
		.event_handler = mock_event_handler,
		.qp_context = NULL,
		.send_cq = cq,
		.recv_cq = cq,
		.srq = srq,
		.cap = {
			.max_send_wr = RHAC_RDMA_MAX_SWR,
			.max_recv_wr = RHAC_RDMA_MAX_RWR, 
			.max_send_sge = RHAC_RDMA_MAX_SSGE,
			.max_recv_sge = RHAC_RDMA_MAX_RSGE,
			.max_rdma_ctxs = 16,
#ifdef RHAC_RDMA_SEND_INLINE
			.max_inline_data = 256,
#endif
		},
		.sq_sig_type = IB_SIGNAL_REQ_WR,
		.qp_type = IB_QPT_RC,
	};

	RHAC_ASSERT(cm_id->device == pd->device);

	err = rdma_create_qp(cm_id, pd, &qp_init_attr);
	RHAC_ASSERT(!err);
	if (err) return -EINVAL;

	return 0;
}

static int cm_handler(struct rdma_cm_id *cm_id, struct rdma_cm_event *event)
{
	int err = 0;
	const struct conn_data *recv_data;

	switch (event->event) {
		case RDMA_CM_EVENT_ADDR_RESOLVED:
			RHAC_LOG("RDMA_CM_EVENT_ADDR_RESOLVED");
			cm_id->device = ib_device;
			err = rdma_resolve_route(cm_id, CONN_TIMEOUT_MS);
			break;

		case RDMA_CM_EVENT_ADDR_ERROR:
			RHAC_LOG("RDMA_CM_EVENT_ADDR_ERROR");
			err = -EINVAL;
			break;

		case RDMA_CM_EVENT_ROUTE_RESOLVED:
			RHAC_LOG("RDMA_CM_EVENT_ROUTE_RESOLVED %llx", (uint64_t)cm_id);
			err = create_qp(cm_id);
			if (err) {
			  RHAC_LOG("ERR!\n");
			  break;
      }

			conn_data.rkey = pd->unsafe_global_rkey;
			err = rdma_connect(cm_id, &conn_param);
			RHAC_ASSERT(!err);
			break;

		case RDMA_CM_EVENT_ROUTE_ERROR:
			RHAC_LOG("RDMA_CM_EVENT_ROUTE_ERROR");
			err = -EINVAL;
			break;

		case RDMA_CM_EVENT_CONNECT_REQUEST:
      RHAC_LOG("RDMA_CM_EVENT_CONNECT_REQUEST");
			err = create_qp(cm_id);
			if (err) break;

			conn_data.rkey = pd->unsafe_global_rkey;

			recv_data = event->param.conn.private_data;
      conn_cm_id[recv_data->node_id] = cm_id;
			global_rkeys[recv_data->node_id] = recv_data->rkey;

			err = rdma_accept(cm_id, &conn_param);
			break;
			
		case RDMA_CM_EVENT_CONNECT_RESPONSE:
			RHAC_LOG("RDMA_CM_EVENT_CONNECT_RESPONSE");
			err = -EINVAL;
			break;

		case RDMA_CM_EVENT_CONNECT_ERROR:
			RHAC_LOG("RDMA_CM_EVENT_CONNECT_ERROR");
			err = -EINVAL;
			break;

		case RDMA_CM_EVENT_UNREACHABLE:
			RHAC_LOG("RDMA_CM_EVENT_UNREACHABLE");
			err = -EINVAL;
			break;

		case RDMA_CM_EVENT_REJECTED:
			RHAC_LOG("RDMA_CM_EVENT_REJECTED");
			err = -EINVAL;
			break;

		case RDMA_CM_EVENT_ESTABLISHED:
			RHAC_LOG("RDMA_CM_EVENT_ESTABLISHED: %p", cm_id->context);
			recv_data = event->param.conn.private_data;
			if (recv_data) { // client-side
        conn_cm_id[recv_data->node_id] = cm_id;
        global_rkeys[recv_data->node_id] = recv_data->rkey;
        complete(&conn_context.done);
			}
			break;

		case RDMA_CM_EVENT_DISCONNECTED:
			//err = -EINVAL;
			break;

		case RDMA_CM_EVENT_DEVICE_REMOVAL:
			err = -EINVAL;
			break;

		case RDMA_CM_EVENT_MULTICAST_JOIN:
			err = -EINVAL;
			break;

		case RDMA_CM_EVENT_MULTICAST_ERROR:
			err = -EINVAL;
			break;

		case RDMA_CM_EVENT_ADDR_CHANGE:
			err = -EINVAL;
			break;

		case RDMA_CM_EVENT_TIMEWAIT_EXIT:
			err = -EINVAL;
			break;
	}

	if (err) {
	  RHAC_LOG("ERRR!!\n");
		conn_context.ret = err;
		complete(&conn_context.done);
	}

	return 0;
}

static int address_setup_ipv4(struct sockaddr *addr, const char *ipv4, __be16 port)
{
	int err;
	struct sockaddr_in *saddr = (struct sockaddr_in*)addr;
	saddr->sin_family = AF_INET;
	saddr->sin_port = htons(port);

	err = in4_pton(ipv4, -1, (u8*)&saddr->sin_addr, -1, NULL);
	if (!err) return -EINVAL;

	return 0;
}

static struct rdma_cm_id* rdma_server_setup(struct sockaddr *addr)
{
	int err;
	struct rdma_cm_id *cm_id = rdma_create_id(
			&init_net,
			cm_handler,
			NULL,
			RDMA_PS_TCP,
			IB_QPT_RC
			);
	if (!cm_id) return NULL;

	err = rdma_set_reuseaddr(cm_id, 1);
	if (err) goto FAIL;

	err = rdma_bind_addr(cm_id, addr);
	if (err) goto FAIL;

	ib_device = cm_id->device;

	err = rdma_listen(cm_id, 5);
	if (err) goto FAIL;

	return cm_id;

FAIL:
	rdma_destroy_id(cm_id);
	return NULL;
}

static char* ib_device_atomic_cap_str(enum ib_atomic_cap cap)
{
	switch (cap) {
		case IB_ATOMIC_NONE: return "IB_ATOMIC_NONE";
		case IB_ATOMIC_HCA: return "IB_ATOMIC_HCA";
		case IB_ATOMIC_GLOB: return "IB_ATOMIC_GLOB";
	}
	return "";
}

static void dump_srq_attr(struct ib_srq_attr *attr)
{
#ifdef RHAC_RDMA_DEBUG
	RHAC_LOG("max_wr: %u", attr->max_wr);
	RHAC_LOG("max_sge: %u", attr->max_sge);
	RHAC_LOG("srq_limit: %u", attr->srq_limit);
#endif
}

static void dump_qp_attr(struct ib_qp_attr *attr)
{
#ifdef RHAC_RDMA_DEBUG
	RHAC_LOG("ib_qp_state: %u", attr->qp_state);
	RHAC_LOG("ib_qp_state: %u", attr->cur_qp_state);
	RHAC_LOG("ib_mtu: %u", attr->path_mtu);
	RHAC_LOG("ib_mig_state: %u", attr->path_mig_state);
	RHAC_LOG("qkey: %u", attr->qkey);
	RHAC_LOG("rq_psn: %u", attr->rq_psn);
	RHAC_LOG("sq_psn: %u", attr->sq_psn);
	RHAC_LOG("dest_qp_num: %u", attr->dest_qp_num);
	RHAC_LOG("qp_access_flags: %u", attr->qp_access_flags);
	RHAC_LOG("max_send_wr: %u", attr->cap.max_send_wr);
	RHAC_LOG("max_recv_wr: %u", attr->cap.max_recv_wr);
	RHAC_LOG("max_send_sge: %u", attr->cap.max_send_sge);
	RHAC_LOG("max_recv_sge: %u", attr->cap.max_recv_sge);
	RHAC_LOG("max_inline_data: %u", attr->cap.max_inline_data);
	RHAC_LOG("max_rdma_ctxs: %u", attr->cap.max_rdma_ctxs);
	RHAC_LOG("qpg_tss_mask_sz: %u", attr->cap.qpg_tss_mask_sz);
//	struct rdma_ah_attr	ah_attr;
//	struct rdma_ah_attr	alt_ah_attr;
	RHAC_LOG("pkey_index: %u", attr->pkey_index);
	RHAC_LOG("alt_pkey_index: %u", attr->alt_pkey_index);
	RHAC_LOG("en_sqd_async_notify: %u", attr->en_sqd_async_notify);
	RHAC_LOG("sq_draining: %u", attr->sq_draining);
	RHAC_LOG("max_rd_atomic: %u", attr->max_rd_atomic);
	RHAC_LOG("max_dest_rd_atomic: %u", attr->max_dest_rd_atomic);
	RHAC_LOG("min_rnr_timer: %u", attr->min_rnr_timer);
	RHAC_LOG("port_num: %u", attr->port_num);
	RHAC_LOG("timeout: %u", attr->timeout);
	RHAC_LOG("retry_cnt: %u", attr->retry_cnt);
	RHAC_LOG("rnr_retry: %u", attr->rnr_retry);
	RHAC_LOG("alt_port_num: %u", attr->alt_port_num);
	RHAC_LOG("alt_timeout: %u", attr->alt_timeout);
	RHAC_LOG("dct_key: %llu", attr->dct_key);
	RHAC_LOG("rate_limit: %u", attr->rate_limit);
	RHAC_LOG("flow_entropy: %u", attr->flow_entropy);
	RHAC_LOG("offload_type: %u", attr->offload_type);
	RHAC_LOG("burst_info.max_burst_sz: %u", attr->burst_info.max_burst_sz);
	RHAC_LOG("burst_info.typical_pkt_sz: %u", attr->burst_info.typical_pkt_sz);
#endif
}

static void dump_qp_init_attr(struct ib_qp_init_attr *attr)
{
}

static void dump_ib_device_attr(struct ib_device_attr *attr)
{
#ifdef RHAC_RDMA_DEBUG
	RHAC_LOG("fw_ver: %llu", attr->fw_ver);
	RHAC_LOG("sys_image_guid: %llu", attr->sys_image_guid);
	RHAC_LOG("max_mr_size: %llu", attr->max_mr_size);
	RHAC_LOG("page_size_cap: %llu", attr->page_size_cap);
	RHAC_LOG("vendor_id: %d", attr->vendor_id);
	RHAC_LOG("vendor_part_id: %d", attr->vendor_part_id);
	RHAC_LOG("hw_ver: %d", attr->hw_ver);
	RHAC_LOG("max_qp: %d", attr->max_qp);
	RHAC_LOG("max_qp_wr: %d", attr->max_qp_wr);
	RHAC_LOG("device_cap_flags: %llu", attr->device_cap_flags);
	RHAC_LOG("max_sge: %d", attr->max_sge);
	RHAC_LOG("max_sge_rd: %d", attr->max_sge_rd);
	RHAC_LOG("max_cq: %d", attr->max_cq);
	RHAC_LOG("max_cqe: %d", attr->max_cqe);
	RHAC_LOG("max_mr: %d", attr->max_mr);
	RHAC_LOG("max_pd: %d", attr->max_pd);
	RHAC_LOG("max_qp_rd_atom: %d", attr->max_qp_rd_atom);
	RHAC_LOG("max_ee_rd_atom: %d", attr->max_ee_rd_atom);
	RHAC_LOG("max_res_rd_atom: %d", attr->max_res_rd_atom);
	RHAC_LOG("max_qp_init_rd_atom: %d", attr->max_qp_init_rd_atom);
	RHAC_LOG("max_ee_init_rd_atom: %d", attr->max_ee_init_rd_atom);
	RHAC_LOG("atomic_cap: %s", ib_device_atomic_cap_str(attr->atomic_cap));
	RHAC_LOG("masked_atomic_cap: %s", ib_device_atomic_cap_str(attr->masked_atomic_cap));
	RHAC_LOG("max_ee: %d", attr->max_ee);
	RHAC_LOG("max_rdd: %d", attr->max_rdd);
	RHAC_LOG("max_mw: %d", attr->max_mw);
	RHAC_LOG("max_raw_ipv6_qp: %d", attr->max_raw_ipv6_qp);
	RHAC_LOG("max_raw_ethy_qp: %d", attr->max_raw_ethy_qp);
	RHAC_LOG("max_mcast_grp: %d", attr->max_mcast_grp);
	RHAC_LOG("max_mcast_qp_attach: %d", attr->max_mcast_qp_attach);
	RHAC_LOG("max_total_mcast_qp_attach: %d", attr->max_total_mcast_qp_attach);
	RHAC_LOG("max_ah: %d", attr->max_ah);
	RHAC_LOG("max_fmr: %d", attr->max_fmr);
	RHAC_LOG("max_map_per_fmr: %d", attr->max_map_per_fmr);
	RHAC_LOG("max_srq: %d", attr->max_srq);
	RHAC_LOG("max_srq_wr: %d", attr->max_srq_wr);
	RHAC_LOG("max_srq_sge: %d", attr->max_srq_sge);
	RHAC_LOG("max_fast_reg_page_list_len: %u",  attr->max_fast_reg_page_list_len);
	RHAC_LOG("max_pkeys: %u",  attr->max_pkeys);
	RHAC_LOG("local_ca_ack_delay: %u", attr->local_ca_ack_delay);
	RHAC_LOG("sig_prot_cap: %d", attr->sig_prot_cap);
	RHAC_LOG("sig_guard_cap: %d", attr->sig_guard_cap);
	RHAC_LOG("odp_caps.general_caps: %llu", attr->odp_caps.general_caps);
	RHAC_LOG("odp_caps.max_size: %llu", attr->odp_caps.max_size);
	RHAC_LOG("odp_caps.per_transport_caps.rc_odp_caps: %u", attr->odp_caps.per_transport_caps.rc_odp_caps);
	RHAC_LOG("odp_caps.per_transport_caps.uc_odp_caps: %u", attr->odp_caps.per_transport_caps.uc_odp_caps);
	RHAC_LOG("odp_caps.per_transport_caps.ud_odp_caps: %u", attr->odp_caps.per_transport_caps.ud_odp_caps);
	RHAC_LOG("odp_caps.per_transport_caps.dc_odp_caps: %u", attr->odp_caps.per_transport_caps.dc_odp_caps);
	RHAC_LOG("nvmf_caps.offload_type_dc: %u", attr->nvmf_caps.offload_type_dc);
	RHAC_LOG("nvmf_caps.offload_type_rc: %u", attr->nvmf_caps.offload_type_rc);
	RHAC_LOG("nvmf_caps.max_namespace: %u", attr->nvmf_caps.max_namespace);
	RHAC_LOG("nvmf_caps.max_staging_buffer_sz: %u", attr->nvmf_caps.max_staging_buffer_sz);
	RHAC_LOG("nvmf_caps.min_staging_buffer_sz: %u", attr->nvmf_caps.min_staging_buffer_sz);
	RHAC_LOG("nvmf_caps.max_io_sz: %u", attr->nvmf_caps.max_io_sz);
	RHAC_LOG("nvmf_caps.max_be_ctrl: %u", attr->nvmf_caps.max_be_ctrl);
	RHAC_LOG("nvmf_caps.max_queue_sz: %u", attr->nvmf_caps.max_queue_sz);
	RHAC_LOG("nvmf_caps.min_queue_sz: %u", attr->nvmf_caps.min_queue_sz);
	RHAC_LOG("nvmf_caps.min_cmd_size: %u", attr->nvmf_caps.min_cmd_size);
	RHAC_LOG("nvmf_caps.max_cmd_size: %u", attr->nvmf_caps.max_cmd_size);
	RHAC_LOG("nvmf_caps. max_data_offset: %u", attr->nvmf_caps. max_data_offset);
	RHAC_LOG("timestamp_mask: %llu", attr->timestamp_mask);
	RHAC_LOG("hca_core_clock: %llu", attr->hca_core_clock);
	RHAC_LOG("supported_qpts: %u", attr->rss_caps.supported_qpts);
	RHAC_LOG("max_rwq_indirection_tables: %u", attr->rss_caps.max_rwq_indirection_tables);
	RHAC_LOG("max_rwq_indirection_table_size: %u", attr->rss_caps.max_rwq_indirection_table_size);
	RHAC_LOG("max_wq_type_rq: %u", attr->max_wq_type_rq);
	RHAC_LOG("raw_packet_caps: %u", attr->raw_packet_caps);
	RHAC_LOG("tm_caps.max_rndv_hdr_size: %u", attr->tm_caps.max_rndv_hdr_size);
	RHAC_LOG("tm_caps.max_num_tags: %u", attr->tm_caps.max_num_tags);
	RHAC_LOG("tm_caps.flags: %u", attr->tm_caps.flags);
	RHAC_LOG("tm_caps.max_ops: %u", attr->tm_caps.max_ops);
	RHAC_LOG("tm_caps.max_sge: %u", attr->tm_caps.max_sge);
	RHAC_LOG("cq_caps.max_cq_moderation_count: %u", attr->cq_caps.max_cq_moderation_count);
	RHAC_LOG("cq_caps.max_cq_moderation_period: %u", attr->cq_caps.max_cq_moderation_period);
	RHAC_LOG("max_counter_sets: %u", attr->max_counter_sets);
#endif
}

int rhac_rdma_launch_server(const char *ipstr, uint16_t port)
{
	int err;
	err = address_setup_ipv4(
			&server_addr,
			ipstr,
			port
			);
	if (err) return -EINVAL;

	init_completion(&conn_context.done);
	conn_context.ret = 0;
	server_cm_id = rdma_server_setup(&server_addr);
	if (!server_cm_id) {
	  return -EINVAL;
  }

	dump_ib_device_attr(&ib_device->attrs);

	pd = ib_alloc_pd(ib_device, IB_PD_UNSAFE_GLOBAL_RKEY);
	if (!pd) {
		return -EINVAL;
	}

	RHAC_LOG("TCP server for RDMA connection launched (ib_addr: %s, ib_port: %d)"
			, ipstr, port);


	return 0;
}

void rhac_rdma_shutdown_server(void)
{
	//if (pd) ib_dealloc_pd(pd);
	pd = NULL;

	if (server_cm_id) rdma_destroy_id(server_cm_id);
	server_cm_id = NULL;

	RHAC_LOG("TCP server for RDMA has been shutdown");
}

static struct rdma_cm_id* rhac_rdma_connect_to(struct sockaddr *addr)
{
	struct rdma_cm_id *cm_id = rdma_create_id(
			&init_net,
			cm_handler,
			NULL,
			RDMA_PS_TCP,
			IB_QPT_RC
			);
	if (!cm_id) return NULL;

	int err;
	err = rdma_resolve_addr(cm_id, NULL, addr, CONN_TIMEOUT_MS);
	if (err) {
		RHAC_LOG("resolve failure");
		rdma_destroy_id(cm_id);
		return NULL;
	}

	err = wait_for_completion_timeout(&conn_context.done, 10 * HZ);
	if (err == 0 || conn_context.ret != 0)  { // timeout or err
		if (err == 0) 
			RHAC_LOG("Timeout");
		else
			RHAC_LOG("Connection failed");
		rdma_destroy_id(cm_id);
		return NULL;
	}
	RHAC_LOG("Connection Established");

	return cm_id;
}

int rhac_rdma_poll(int npoll, struct ib_wc *wc)
{
	return ib_poll_cq(cq, 1, wc);
}

struct device* rhac_rdma_device(void)
{
	if (!ib_device) return NULL;
	return ib_device->dma_device;
}

int rhac_rdma_init(uint32_t node_id, uint32_t num_nodes)
{

	int i, err;
	for (i = 0; i < sizeof(node_ip_str)/sizeof(char*); i++) {
		//RHAC_LOG("%d: %s", i, node_ip_str[i]);
		err = address_setup_ipv4(
				&node_addrs[i],
				node_ip_str[i],
				IB_CLIENT_PORT 
				);
		if (err) return -EINVAL;
	}

	cq = ib_alloc_cq(ib_device, NULL, RHAC_RDMA_MAX_CQE, 0, IB_POLL_WORKQUEUE);
	if (!cq) return -EINVAL;

	struct ib_srq_init_attr srq_init_attr = {
		.event_handler = mock_event_handler,
		.srq_context = NULL,
		.attr = {
			.max_wr = RHAC_RDMA_MAX_RWR,
			.max_sge = RHAC_RDMA_MAX_RSGE,
			.srq_limit = RHAC_RDMA_SRQ_LIMIT,
		},
		.srq_type = IB_SRQT_BASIC,
	};

	srq = ib_create_srq(pd, &srq_init_attr);
	if (!srq) {
		ib_free_cq(cq);
		cq = NULL;
		return -EINVAL;
	}

	conn_data.node_id = node_id;
	conn_data.rkey = pd->unsafe_global_rkey;

	for (i = 0; i < num_nodes; i++) {
		conn_cm_id[i] = NULL;
	}

	for (i = node_id+1; i < num_nodes; i++) {
		conn_cm_id[i] = rhac_rdma_connect_to(&node_addrs[i]);
		if (!conn_cm_id[i]) {
			rhac_rdma_deinit();
			return -EINVAL;
		}
	}

  /*
   struct ib_srq_attr srq_attr;
   err = ib_query_srq(srq, &srq_attr);
   RHAC_ASSERT(!err);
   dump_srq_attr(&srq_attr);

   struct ib_qp_attr qp_attr;
   struct ib_qp_init_attr qp_init_attr;
   for (i = node_id+1; i < num_nodes; i++) {
   err = ib_query_qp(conn_cm_id[i]->qp, &qp_attr, 0, &qp_init_attr);
   RHAC_ASSERT(!err);
   dump_qp_attr(&qp_attr);
   }
 */

	return 0;
}

void rhac_rdma_deinit(void)
{
	int i;

	init_completion(&conn_context.done);
	for (i = 0; i < RHAC_MAX_NODES; i++) {
		if (conn_cm_id[i]) {
			rdma_destroy_qp(conn_cm_id[i]);
			rdma_destroy_id(conn_cm_id[i]);
			conn_cm_id[i] = NULL;
		}
	}

	//if (cq) ib_free_cq(cq);
	//if (srq) ib_destroy_srq(srq);
	cq = NULL;
	srq = NULL;
}
