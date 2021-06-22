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

#ifndef __RHAC_NVIDIA_MM_H__
#define __RHAC_NVIDIA_MM_H__

struct page;

void rhac_nvidia_mm_lock_blk(uint64_t blk_vaddr);
void rhac_nvidia_mm_unlock_blk(uint64_t blk_vaddr);

struct page** rhac_nvidia_mm_get_pages(uint64_t blk_vaddr, const unsigned long *mask);

int rhac_nvidia_mm_disable_write_async(uint64_t blk_vaddr, const unsigned long *prot_mask);

int rhac_nvidia_mm_stage_to_cpu(uint64_t blk_vaddr, const unsigned long *page_mask);
int rhac_nvidia_mm_stage_to_cpu_async(uint64_t blk_vaddr, const unsigned long *page_mask);

int rhac_nvidia_mm_pin_sg(uint64_t blk_vaddr, const unsigned long *page_mask, const unsigned long *prot_mask);
int rhac_nvidia_mm_unpin_sg(uint64_t blk_vaddr, const unsigned long *page_mask, const unsigned long *prot_mask);

int rhac_nvidia_mm_inv(uint64_t blk_vaddr, const unsigned long *invmask);

int rhac_nvidia_mm_split_va_range(uint64_t blk_vaddr, uint64_t len);

int rhac_nvidia_mm_toggle_dup_flag(uint64_t blk_vaddr, uint64_t size, uint32_t on);

int rhac_nvidia_mm_prefetch_to_cpu(uint64_t blk_vaddr, uint64_t size, uint32_t device_id, bool is_async);
int rhac_nvidia_mm_prefetch_to_gpu(uint64_t blk_vaddr, uvm_page_mask_t *page_mask, uint32_t device_id);

int rhac_nvidia_mm_make_resident_cpu(uint64_t blk_vaddr, uvm_processor_id_t dst_id, const unsigned long *mask);
int rhac_nvidia_mm_copy_to_buf(uint64_t blk_vaddr, const unsigned long *mask, void *buf);
int rhac_nvidia_mm_copy_from_buf(uint64_t blk_vaddr, const unsigned long *mask, void *buf);

int rhac_nvidia_mm_populate(uint64_t blk_vaddr, const unsigned long *page_mask);

#endif //__RHAC_NVIDIA_MM_H__
