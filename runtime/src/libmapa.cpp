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

#include "libmapa.h"
#include <dlfcn.h>
#include <assert.h>
#include <unistd.h>
#include <libgen.h>
#include <string.h>

LibMAPA* LibMAPA::singleton_ = NULL;

LibMAPA* LibMAPA::GetLibMAPA() {
  if (singleton_ == NULL)
    singleton_ = new LibMAPA();
  return singleton_;
}

LibMAPA::LibMAPA() {
  if (OpenMAPA())
    mapa_opened_ = true;
  else
    mapa_opened_ = false;
}

LibMAPA::~LibMAPA() {
  CloseMAPA();
}

int32_t LibMAPA::MAPA_get_kernel_id(const char* kernel_name) {
  if (mapa_opened_ == true)
    return MAPA_get_kernel_id_(kernel_name);
  else
    return -1;
}

bool LibMAPA::OpenMAPA() {
  char buf[255];

  // find from executable directory
  readlink("/proc/self/exe", buf, 255);
  dirname(buf);
  strcat(buf, "/kernel.mapa");
  RHAC_LOG("Trying to open mapa library from \"%s\"\n", buf);
  mapa_handle_ = dlopen(buf, RTLD_NOW);

  if (!mapa_handle_) {
    // find from current directory
    getcwd(buf, 255);
    strcat(buf, "/kernel.mapa");
    RHAC_LOG("Trying to open mapa library from \"%s\"\n", buf);
    mapa_handle_ = dlopen(buf, RTLD_NOW);
    if (!mapa_handle_) {
      fprintf(stderr, "Failed to open mapa library\n");
      return false;
    }
  }

  MAPA_get_kernel_id_ = (int32_t(*)(const char*))
    dlsym(mapa_handle_, "MAPA_get_kernel_id");
  CHECK_MAPA_SYMBOL(MAPA_get_kernel_id_);

  MAPA_get_num_readonly_buffers = (uint32_t(*)(int32_t))
    dlsym(mapa_handle_, "MAPA_get_num_readonly_buffers");
  CHECK_MAPA_SYMBOL(MAPA_get_num_readonly_buffers);

  MAPA_get_num_non_readonly_buffers = (uint32_t(*)(int32_t))
    dlsym(mapa_handle_, "MAPA_get_num_non_readonly_buffers");
  CHECK_MAPA_SYMBOL(MAPA_get_num_non_readonly_buffers);

  MAPA_get_readonly_buffers = (void(*)(int32_t, uint32_t*))
    dlsym(mapa_handle_, "MAPA_get_readonly_buffers");
  CHECK_MAPA_SYMBOL(MAPA_get_readonly_buffers);

  MAPA_get_non_readonly_buffers = (void(*)(int32_t, uint32_t*))
    dlsym(mapa_handle_, "MAPA_get_non_readonly_buffers");
  CHECK_MAPA_SYMBOL(MAPA_get_non_readonly_buffers);

  MAPA_get_num_expressions = (uint32_t(*)(int32_t))
    dlsym(mapa_handle_, "MAPA_get_num_expressions");
  CHECK_MAPA_SYMBOL(MAPA_get_num_expressions);

  MAPA_get_kernel_arg_index = (size_t(*)(int32_t, uint32_t))
    dlsym(mapa_handle_, "MAPA_get_kernel_arg_index");
  CHECK_MAPA_SYMBOL(MAPA_get_kernel_arg_index);

  MAPA_get_gx_coeff = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_gx_coeff");
  CHECK_MAPA_SYMBOL(MAPA_get_gx_coeff);

  MAPA_get_gy_coeff = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_gy_coeff");
  CHECK_MAPA_SYMBOL(MAPA_get_gy_coeff);

  MAPA_get_gz_coeff = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_gz_coeff");
  CHECK_MAPA_SYMBOL(MAPA_get_gz_coeff);

  MAPA_get_lx_coeff = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_lx_coeff");
  CHECK_MAPA_SYMBOL(MAPA_get_lx_coeff);

  MAPA_get_ly_coeff = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_ly_coeff");
  CHECK_MAPA_SYMBOL(MAPA_get_ly_coeff);

  MAPA_get_lz_coeff = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_lz_coeff");
  CHECK_MAPA_SYMBOL(MAPA_get_lz_coeff);

  MAPA_get_i0_bound = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_i0_bound");
  CHECK_MAPA_SYMBOL(MAPA_get_i0_bound);

  MAPA_get_i0_step = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_i0_step");
  CHECK_MAPA_SYMBOL(MAPA_get_i0_step);

  MAPA_get_i1_bound = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_i1_bound");
  CHECK_MAPA_SYMBOL(MAPA_get_i1_bound);

  MAPA_get_i1_step = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_i1_step");
  CHECK_MAPA_SYMBOL(MAPA_get_i1_step);

  MAPA_get_const = (int64_t(*)(int32_t, uint32_t, void**, dim3, dim3))
    dlsym(mapa_handle_, "MAPA_get_const");
  CHECK_MAPA_SYMBOL(MAPA_get_const);

  MAPA_get_fetch_size = (size_t(*)(int32_t, uint32_t))
    dlsym(mapa_handle_, "MAPA_get_fetch_size");
  CHECK_MAPA_SYMBOL(MAPA_get_fetch_size);

  MAPA_is_readonly_buffer = (bool(*)(int32_t, uint32_t))
    dlsym(mapa_handle_, "MAPA_is_readonly_buffer");
  CHECK_MAPA_SYMBOL(MAPA_is_readonly_buffer);

  MAPA_is_one_thread_expression = (bool(*)(int32_t, uint32_t))
    dlsym(mapa_handle_, "MAPA_is_one_thread_expression");
  CHECK_MAPA_SYMBOL(MAPA_is_one_thread_expression);

  RHAC_LOG("Opened mapa library from \"%s\"\n", buf);

  RHAC_LOG("Total number of kernels in kernel.mapa is %d\n",
      MAPA_get_kernel_id_("_GET_NUM_KERNELS_"));

  return true;
}

void LibMAPA::CloseMAPA() {
  if (mapa_handle_)
    dlclose(mapa_handle_);
}
