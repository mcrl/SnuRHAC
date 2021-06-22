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

#ifndef __RHAC_H__
#define __RHAC_H__

#include <stdint.h>
#include <vector>
#include <stdlib.h>
#include <string>

#define KB (1024L)
#define MB (1024L*KB)
#define GB (1024L*MB)

#define DEFAULT_MALLOC_MANAGED_SIZE       (16L*1024L*GB)
#define DEFAULT_CLUSTERSVM_ALLOC_SIZE     (4L*1024L*GB)
#define DEFAULT_CLUSTERSVM_RESERVE_SIZE   (16L*GB)
//#define DEFAULT_CLUSTERSVM_RESERVE_SIZE   (1L*KB)
#define SVM_ALIGNMENT                     (2L*MB)

#define SVM_TYPE_DEVICE   0
#define SVM_TYPE_HOST     1
#define SVM_TYPE_MANAGED  2

class Platform;
class LibCUDA;
class RHACCommand;
class RHACResponse;
class Thread;
class Executor;
class NodeExecutor;
class DeviceExecutor;
class Transmitter;
class Receiver;
class Communicator;
class ClusterSVMObject;
class RHACDriver;
class RHACEvent;
class RHACBarrier;

extern Platform rhac_platform;

// FIXME
typedef uint64_t rhac_command_id_t;

//FIXME
typedef int clusterSVMObject_t;

typedef struct FunctionInfo_t {
  int has_global_atomics;
  int num_args;
  std::vector<int> arg_sizes;
} FunctionInfo;

typedef struct VarInfo_t {
  std::string var_name;
  size_t type_bitwidth;
  size_t array_width;
  void* dev_ptr;
} VarInfo;

#endif // __RHAC_H__
