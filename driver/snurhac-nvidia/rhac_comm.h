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

#ifndef __RHAC_COMM_H__
#define __RHAC_COMM_H__

#include "rhac_types.h"
#include "rhac_config.h"

struct rhac_nvidia_isr_ctx;
struct rhac_comm {
	atomic_t posted;
	atomic_t cnt;
	atomic_t req_unlock;

	int err;
	int processing;
	wait_queue_head_t wait_queue;

	struct rhac_comm *parent;
	struct rhac_nvidia_isr_ctx *isr_ctx;
	DECLARE_BITMAP(copy_mask, RHAC_PDSC_PER_PBLK);

	int cur;
	int next;

	int type;

	struct spinlock lock;
	struct list_head list;
	struct list_head local_list;


	int number;
};

int rhac_comm_init(void);
void rhac_comm_deinit(void);
struct rhac_comm* rhac_comm_alloc(void);
struct rhac_comm* rhac_comm_spawn(struct rhac_comm *comm);
void rhac_comm_free(struct rhac_comm *comm);
void rhac_comm_post(struct rhac_comm *comm);
unsigned int rhac_comm_unpost(struct rhac_comm *comm);
void rhac_comm_fail(struct rhac_comm *ch, int err);
int rhac_comm_wait(struct rhac_comm *comm);
void rhac_comm_post(struct rhac_comm *comm);
unsigned int rhac_comm_unpost(struct rhac_comm *comm);

#endif //__RHAC_COMM_H__
