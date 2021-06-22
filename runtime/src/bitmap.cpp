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

#include "bitmap.h"

void bitmap_set(unsigned long *map, unsigned int start, int len) {
	unsigned long *p = map + BIT_WORD(start);
	const unsigned int size = start + len;
	int bits_to_set = BITS_PER_LONG - (start % BITS_PER_LONG);
	unsigned long mask_to_set = BITMAP_FIRST_WORD_MASK(start);

	while (len - bits_to_set >= 0) {
		*p |= mask_to_set;
		len -= bits_to_set;
		bits_to_set = BITS_PER_LONG;
		mask_to_set = ~0UL;
		p++;
	}
	if (len) {
		mask_to_set &= BITMAP_LAST_WORD_MASK(size);
		*p |= mask_to_set;
	}
}

void __bitmap_or(unsigned long *dst, const unsigned long *bitmap1,
    const unsigned long *bitmap2, unsigned int bits) {
  unsigned int k;
  unsigned int nr = BITS_TO_LONGS(bits);

  for (k = 0; k < nr; k++)
    dst[k] = bitmap1[k] | bitmap2[k];
}

unsigned int get_page_index_in_block(uint64_t addr) {
  return ((addr & UVM_VA_BLOCK_SIZE_MASK) / PAGE_SIZE);
}

void zero_page_mask(uvm_page_mask_t *page_mask) {
  bitmap_zero(page_mask->bitmap, PAGES_PER_UVM_VA_BLOCK);
}

void set_page_mask(uvm_page_mask_t *page_mask,
    uint64_t addr, size_t length) {
  unsigned int start_index = get_page_index_in_block(addr);
  unsigned int end_index = get_page_index_in_block(addr + length - 1);
  bitmap_set(page_mask->bitmap, start_index, end_index - start_index + 1);
}

void merge_page_mask(uvm_page_mask_t *dst, uvm_page_mask_t *src) {
  __bitmap_or(dst->bitmap, dst->bitmap, src->bitmap, PAGES_PER_UVM_VA_BLOCK);
}

void fill_page_mask(uvm_page_mask_t *page_mask) {
  bitmap_set(page_mask->bitmap, 0, PAGES_PER_UVM_VA_BLOCK);
}
