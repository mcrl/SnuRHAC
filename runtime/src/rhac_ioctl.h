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

#ifndef __RHAC_IOCTL_H__
#define __RHAC_IOCTL_H__

#ifdef __cplusplus
extern "C" {
#endif


//
// INIT
//
typedef struct {
  uint64_t vaddr_base;
  uint32_t num_nodes;
  uint32_t node_id;
  uint32_t num_local_gpus;
} rhac_iocx_init_param_t;
#define RHAC_IOCX_INIT					_IOW(0xff, 0, rhac_iocx_init_param_t)

//
// RESERVE
//
typedef struct {
  uint64_t capacity;
} rhac_iocx_reserve_param_t;
#define RHAC_IOCX_RESERVE				_IOW(0xff, 1, rhac_iocx_reserve_param_t)

//
// SYNC
//
typedef struct {
} rhac_iocx_sync_param_t;
#define RHAC_IOCX_SYNC					_IO (0xff, 2)

//
// SPLIT_VA_RANGE
//
typedef struct {
  uint64_t vaddr;
  uint64_t length;
} rhac_iocx_split_va_range_param_t;
#define RHAC_IOCX_SPLIT_VA_RANGE      _IOW(0xff, 10, rhac_iocx_split_va_range_param_t)

//
// TOGGLE_DUP_FLAG
//
typedef struct {
  uint64_t vaddr;
  uint64_t size;
  uint32_t turnon_flag;
} rhac_iocx_toggle_dup_flag_param_t;
#define RHAC_IOCX_TOGGLE_DUP_FLAG     _IOW(0xff, 11, rhac_iocx_toggle_dup_flag_param_t)

//
// PREFETCH_TO_CPU
//
typedef struct {
  uint64_t vaddr;
  uint64_t size;
  uint32_t device_id;
  bool is_async;
} rhac_iocx_prefetch_to_cpu_param_t;
#define RHAC_IOCX_PREFETCH_TO_CPU     _IOW(0xff, 12, rhac_iocx_prefetch_to_cpu_param_t)

//
// PREFETCH_TO_GPU
//
typedef struct {
  uint64_t vaddr;
  uvm_page_mask_t page_mask;
  uint32_t device_id;
} rhac_iocx_prefetch_to_gpu_param_t;
#define RHAC_IOCX_PREFETCH_TO_GPU     _IOW(0xff, 13, rhac_iocx_prefetch_to_gpu_param_t)

#ifdef __cplusplus
}
#endif

#endif //__RHAC_IOCTL_H__
