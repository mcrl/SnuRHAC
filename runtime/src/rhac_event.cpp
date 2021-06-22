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

#include "rhac_event.h"
#include "platform.h"
#include "libcuda.h"

RHACEvent::RHACEvent(cudaEvent_t *event) 
{
  int nNodes, nDevs;

  nNodes = rhac_platform.GetClusterNumNodes();

  node_wait_ids_ = new rhac_command_id_t[nNodes];
  device_wait_ids_ = new rhac_command_id_t*[nNodes];

  for (int n = 0; n < nNodes; n++) {
    nDevs = rhac_platform.GetNumDevicesIn(n);
    device_wait_ids_[n] = new rhac_command_id_t[nDevs];
  }

  // create CUDA event
  LibCUDA *libcuda = LibCUDA::GetLibCUDA();
  cudaError_t cuda_err;
  cuda_err = libcuda->cudaEventCreate(event);
  CHECK_CUDART_ERROR(cuda_err);
  event_ = *event;
}

RHACEvent::~RHACEvent()
{
  int nNodes = rhac_platform.GetClusterNumNodes();

  delete[] node_wait_ids_;

  for (int n = 0; n < nNodes; n++) {
    delete[] device_wait_ids_[n];
  }
  delete[] device_wait_ids_;
}

void RHACEvent::RegisterEventIDs(rhac_command_id_t id, int node)
{
  node_wait_ids_[node] = id;
}

void RHACEvent::RegisterEventIDs(rhac_command_id_t id, int node, int dev)
{
  device_wait_ids_[node][dev] = id;
}

bool RHACEvent::QueryEvent()
{
  int nNodes = rhac_platform.GetClusterNumNodes();
  int nDevs;
  bool ret = true;

  for (int n = 0; n < nNodes; n++) {
    ret &= rhac_platform.QueryResponse(node_wait_ids_[n], n);

    if (!ret)
      break;

    nDevs = rhac_platform.GetNumDevicesIn(n);
    for (int d = 0; d < nDevs; d++) {
      ret &= rhac_platform.QueryResponse(device_wait_ids_[n][d], n, d);

      if (!ret)
        break;
    }

    if (!ret)
      break;
  }

  return ret;
}

void RHACEvent::WaitEvent()
{
  int nNodes = rhac_platform.GetClusterNumNodes();
  int nDevs;

  for (int n = 0; n < nNodes; n++) {
    rhac_platform.WaitResponse(node_wait_ids_[n], n);

    nDevs = rhac_platform.GetNumDevicesIn(n);

    for (int d = 0; d < nDevs; d++) {
      rhac_platform.WaitResponse(device_wait_ids_[n][d], n, d);
    }
  }
}

cudaEvent_t RHACEvent::GetEvent()
{
  return event_;
}

