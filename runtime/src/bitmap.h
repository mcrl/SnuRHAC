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

#ifndef __BITMAP_H__
#define __BITMAP_H__

#ifndef __KERNEL__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

#ifndef PAGE_SIZE
#define PAGE_SIZE	4096
#endif

#ifndef __WORDSIZE
#define __WORDSIZE (__SIZEOF_LONG__ * 8)
#endif

#ifndef BITS_PER_LONG
#define BITS_PER_LONG __WORDSIZE
#endif

#define BIT(nr)												(1UL << (nr))
#define BIT_ULL(nr)										(1ULL << (nr))
#define BIT_MASK(nr)									(1UL << ((nr) % BITS_PER_LONG))
#define BIT_WORD(nr)									((nr) / BITS_PER_LONG)
#define BIT_ULL_MASK(nr)							(1ULL << ((nr) % BITS_PER_LONG_LONG))
#define BIT_ULL_WORD(nr)							((nr) / BITS_PER_LONG_LONG)
#define BITS_PER_BYTE									8
#define DIV_ROUND_UP(n,d)							(((n) + (d) - 1) / (d))
#define BITS_TO_LONGS(nr)							DIV_ROUND_UP(nr, BITS_PER_BYTE * sizeof(long))
#define BITMAP_FIRST_WORD_MASK(start)	(~0UL << ((start) & (BITS_PER_LONG - 1)))
#define BITMAP_LAST_WORD_MASK(nbits)	(~0UL >> (-(nbits) & (BITS_PER_LONG - 1)))

#define DECLARE_BITMAP(name,bits) \
      unsigned long name[BITS_TO_LONGS(bits)]

typedef struct {
    DECLARE_BITMAP(bitmap, PAGES_PER_UVM_VA_BLOCK);
} uvm_page_mask_t;

static inline void bitmap_zero(unsigned long *dst, unsigned int nbits) {
  unsigned int len = BITS_TO_LONGS(nbits) * sizeof(unsigned long);
  memset(dst, 0, len);
}

void bitmap_set(unsigned long *map, unsigned int start, int len);

unsigned int get_page_index_in_block(uint64_t addr);
void zero_page_mask(uvm_page_mask_t *page_mask);
void set_page_mask(uvm_page_mask_t *page_mask, uint64_t addr, size_t length);
void merge_page_mask(uvm_page_mask_t *dst, uvm_page_mask_t *src);    
void fill_page_mask(uvm_page_mask_t *page_mask);

#endif // __KERNEL__

#endif // __BITMAP_H__
