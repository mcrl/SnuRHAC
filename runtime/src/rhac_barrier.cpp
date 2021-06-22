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

#include "rhac_barrier.h"
#include "platform.h"
#include <map>
#include <assert.h>

RHACBarrier::RHACBarrier()
{
  assert(rhac_platform.IsHost());
}

RHACBarrier::~RHACBarrier()
{
}

void RHACBarrier::RegisterBarrierCommandID(rhac_command_id_t cmd_id, int node, int dev)
{
  assert(rhac_platform.IsHost());

  std::pair<int, int> k;
  std::pair<rhac_command_id_t, bool> v;

  k = std::make_pair(node, dev);
  v = std::make_pair(cmd_id, false);

  assert(pending_map_.find(k) == pending_map_.end());

  pending_map_.insert(std::make_pair(k, v));
}

void RHACBarrier::RegisterBarrierCommandID(rhac_command_id_t cmd_id, int node)
{
  assert(rhac_platform.IsHost());

  RHACBarrier::RegisterBarrierCommandID(cmd_id, node, -1);
}

bool RHACBarrier::CheckFinish()
{
  assert(rhac_platform.IsHost());

  std::map<std::pair<int, int>,
           std::pair<rhac_command_id_t, bool>>::iterator mi;

  bool finish = true;

  for (mi = pending_map_.begin();
       mi != pending_map_.end();
       mi++) 
  {
    if (((*mi).second).second == false) {
      int n, d;
      bool result = false;
      rhac_command_id_t cmd_id;

      n = (mi->first).first;
      d = (mi->first).second;

      cmd_id = ((*mi).second).first;

      // update status 
      if (d == -1) { 
        // node thread command
        result = rhac_platform.QueryResponse(cmd_id, n);
      }
      else {
        // device thread command
        result = rhac_platform.QueryResponse(cmd_id, n, d);
      }

      if (result) {
        (mi->second).second = result;
      }
      else {
        finish = false;
        break;
      }
    }
  }

  return finish;
}

void RHACBarrier::PrintStatus()
{
  assert(rhac_platform.IsHost());

  std::map<std::pair<int, int>,
           std::pair<rhac_command_id_t, bool>>::iterator mi;
  bool result;

  RHAC_LOG("Print Barrier Status Start");

  for (mi = pending_map_.begin();
       mi != pending_map_.end();
       mi++)
  {
    int n, d;
    rhac_command_id_t cmd_id;
    
    n = (mi->first).first;
    d = (mi->first).second;

    cmd_id = ((*mi).second).first;

    if (d == -1) {
      result = rhac_platform.QueryResponse(cmd_id, n);
      RHAC_LOG("Node %d Dev %d : Status %d(expect %lu but %lu)", n, d, result, cmd_id, *rhac_platform.GetResponseQueue(n));
    }
    else {
      result = rhac_platform.QueryResponse(cmd_id, n, d);
      RHAC_LOG("Node %d Dev %d : Status %d(expect %lu but %lu)", n, d, result, cmd_id, *rhac_platform.GetResponseQueue(n, d));
    }
  }

  RHAC_LOG("Print Barrier Status End");


}
