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

#ifndef __RHAC_RDMA_H__
#define __RHAC_RDMA_H__

#include <linux/dma-mapping.h>
#include "rhac_config.h"

struct ib_cqe;
struct device;
struct ib_wc;
struct sg_table;

//#define RHAC_RDMA_MAX_SWR 	(4096 + 512)
#define RHAC_RDMA_MAX_SWR 	(4096)
#define RHAC_RDMA_MAX_RWR 	(4096)
#define RHAC_RDMA_MAX_CQE   	(8192)
#define RHAC_RDMA_MAX_SSGE	16
#define RHAC_RDMA_MAX_RSGE	16
#define RHAC_RDMA_MAX_QP_RD_ATOM 1
#define RHAC_RDMA_SRQ_LIMIT (1024)


int rhac_rdma_launch_server(const char *ipstr, uint16_t port);
void rhac_rdma_shutdown_server(void);

int rhac_rdma_init(uint32_t node_id, uint32_t num_nodes);
void rhac_rdma_deinit(void);

int rhac_rdma_map_mr_sg(struct scatterlist *sg, int sg_nents);

int rhac_rdma_post_send(uint64_t dma_addr, uint64_t size, uint32_t dst_id, struct ib_cqe *cqe, bool signal, bool fence);
int rhac_rdma_post_recv(uint64_t dma_addr, uint64_t size, struct ib_cqe *cqe);

int rhac_rdma_read(uint64_t dma_addr, uint64_t raddr, uint64_t length, uint64_t dst_id, struct ib_cqe *cqe, bool signal);

uint64_t rhac_rdma_map_single(void *cpu_addr, size_t size, enum dma_data_direction direction);
void rhac_rdma_unmap_single(uint64_t daddr, size_t size, enum dma_data_direction direction);

int rhac_rdma_map_sg(struct scatterlist *sg, int nents, enum dma_data_direction direction);
void rhac_rdma_unmap_sg(struct scatterlist *sg, int nents, enum dma_data_direction direction);

uint64_t rhac_rdma_map_page(struct page *page, size_t offset, enum dma_data_direction direction);
void rhac_rdma_unmap_page(uint64_t daddr, size_t size, enum dma_data_direction direction);

void rhac_rdma_sync_sg(struct scatterlist *sgl, int nents, enum dma_data_direction direction);
void rhac_rdma_sync(uint64_t dma_addr, int len, enum dma_data_direction direction);
int rhac_rdma_mapping_error(uint64_t addr);
int rhac_rdma_poll(int npoll, struct ib_wc *wc);

struct device* rhac_rdma_device(void);

#endif //__RHAC_RDMA_H__
