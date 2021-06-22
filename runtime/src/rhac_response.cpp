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

#include "rhac_response.h"
#include "utils.h"
#include "platform.h"

#include <cstring>
#include <assert.h>

RHACResponse::RHACResponse()
{
}

RHACResponse::~RHACResponse()
{
}

void RHACResponse::SetDefaultInfo(rhac_command_id_t command_id,
                                  int node, int device)
{
  assert(WriteData((char *)&command_id, sizeof(rhac_command_id_t), OFFSET_RESPONSE_ID));
  assert(WriteData((char *)&node, sizeof(int), OFFSET_RESPONSE_NODE));
  assert(WriteData((char *)&device, sizeof(int), OFFSET_RESPONSE_DEVICE));
}

bool RHACResponse::WriteData(const char *src,
                             const size_t size,
                             const size_t offset)
{
  if (offset + size > RESPONSE_SIZE)
    return false;

  memcpy(response_ + offset, src, size);

  return true;
}

void RHACResponse::ReadData(char *dst,
                            const size_t size,
                            const size_t offset)
{
  assert(offset + size <= RESPONSE_SIZE);

  memcpy(dst, response_ + offset, size);
}

rhac_command_id_t RHACResponse::GetCommandID()
{
  rhac_command_id_t ret;
  ReadData((char *)&ret, sizeof(rhac_command_id_t), OFFSET_RESPONSE_ID);
  return ret;
}

int RHACResponse::GetTargetNode()
{
  int ret;
  ReadData((char *)&ret, sizeof(int), OFFSET_RESPONSE_NODE);
  return ret;
}

int RHACResponse::GetTargetDevice()
{
  int ret;
  ReadData((char *)&ret, sizeof(int), OFFSET_RESPONSE_DEVICE);
  return ret;
}

char * RHACResponse::GetDataPtr()
{
  return response_;
}

void RHACResponse::PrintInfo(const char *prefix)
{
  RHAC_LOG("Rank %d (%s) Response Info - ID %zu, Node %d, Device %d",
           rhac_platform.GetRank(),
           prefix,
           GetCommandID(),
           GetTargetNode(),
           GetTargetDevice());
}
