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

#ifndef __RHAC_LIBMAPA_H__
#define __RHAC_LIBMAPA_H__

#include "utils.h"
#include <pthread.h>
#include <stdio.h>

#define CHECK_MAPA_SYMBOL(x)                        \
  do {                                              \
    if (x == NULL) {                                \
      printf("Failed to load symbol " #x "\n");     \
      assert(0);                                    \
      return false;                                 \
    }                                               \
  } while (0)

class LibMAPA {
public:
  bool OpenMAPA();
  void CloseMAPA();

  int32_t MAPA_get_kernel_id(const char*);

  uint32_t (*MAPA_get_num_readonly_buffers)(int32_t);
  uint32_t (*MAPA_get_num_non_readonly_buffers)(int32_t);
  void (*MAPA_get_readonly_buffers)(int32_t, uint32_t *);
  void (*MAPA_get_non_readonly_buffers)(int32_t, uint32_t *);

  uint32_t (*MAPA_get_num_expressions)(int32_t);
  size_t (*MAPA_get_kernel_arg_index)(int32_t, uint32_t);

  int64_t (*MAPA_get_gx_coeff)(int32_t, uint32_t, void**, dim3, dim3);
  int64_t (*MAPA_get_gy_coeff)(int32_t, uint32_t, void**, dim3, dim3);
  int64_t (*MAPA_get_gz_coeff)(int32_t, uint32_t, void**, dim3, dim3);
  int64_t (*MAPA_get_lx_coeff)(int32_t, uint32_t, void**, dim3, dim3);
  int64_t (*MAPA_get_ly_coeff)(int32_t, uint32_t, void**, dim3, dim3);
  int64_t (*MAPA_get_lz_coeff)(int32_t, uint32_t, void**, dim3, dim3);
  int64_t (*MAPA_get_i0_bound)(int32_t, uint32_t, void**, dim3, dim3);
  int64_t (*MAPA_get_i0_step)(int32_t, uint32_t, void**, dim3, dim3);
  int64_t (*MAPA_get_i1_bound)(int32_t, uint32_t, void**, dim3, dim3);
  int64_t (*MAPA_get_i1_step)(int32_t, uint32_t, void**, dim3, dim3);
  int64_t (*MAPA_get_const)(int32_t, uint32_t, void**, dim3, dim3);
  size_t (*MAPA_get_fetch_size)(int32_t, uint32_t);
  bool (*MAPA_is_readonly_buffer)(int32_t, uint32_t);
  bool (*MAPA_is_one_thread_expression)(int32_t, uint32_t);

private:
  int32_t (*MAPA_get_kernel_id_)(const char*);

  void *mapa_handle_;
  bool mapa_opened_;

public:
  static LibMAPA* GetLibMAPA();

private:
  LibMAPA();
  ~LibMAPA();

  static LibMAPA* singleton_;
};

#endif // __RHAC_LIBMAPA_H__
