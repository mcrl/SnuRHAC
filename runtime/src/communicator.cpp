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

#include "communicator.h"
#include "rhac_command.h"
#include "rhac_response.h"
#include "platform.h"

#include <mpi.h>

Communicator* Communicator::singleton_ = NULL;
mutex_t Communicator::mutex_;

Communicator* Communicator::GetCommunicator()
{
  mutex_.lock();
  if (singleton_ == NULL)
    singleton_ = new Communicator();
  mutex_.unlock();

  return singleton_;
}

Communicator::Communicator()
{
}

Communicator::~Communicator()
{
}

void Communicator::SendCommand(RHACCommand *cmd)
{
  int node, err;
  char *header;

  node = cmd->GetTargetNode();
  header = cmd->GetHeaderPtr();

  err = MPI_Send((const void *)header, COMMAND_HEADER_SIZE,
                 MPI_CHAR, node, RHAC_MPI_TAG,
                 MPI_COMM_WORLD);
  CHECK_MPI_ERROR(err);
}

void Communicator::SendPayload(RHACCommand *cmd)
{
  /* not tested */
  int node, err;
  uint32_t payload_size;
  char *payload_ptr;

  assert(cmd->HasPayload());

  node = cmd->GetTargetNode();
  payload_size = cmd->GetPayloadSize();
  payload_ptr = cmd->GetPayloadPtr();

  err = MPI_Send((const void *)payload_ptr, payload_size,
                 MPI_CHAR, node, RHAC_MPI_TAG,
                 MPI_COMM_WORLD);
  CHECK_MPI_ERROR(err);
}

void Communicator::SendCommandWithPayload(RHACCommand *cmd)
{
  SendCommand(cmd);

  if (cmd->HasPayload())
    SendPayload(cmd);
}

void Communicator::IsendCommand(RHACCommand *cmd,
                                MPI_Request *request)
{
  int node, err;
  char *header;

  node = cmd->GetTargetNode();
  header = cmd->GetHeaderPtr();

  err = MPI_Isend((const void *)header, COMMAND_HEADER_SIZE,
                  MPI_CHAR, node, RHAC_MPI_TAG,
                  MPI_COMM_WORLD, request);
  CHECK_MPI_ERROR(err);
}

void Communicator::IsendPayload(RHACCommand *cmd,
                                MPI_Request *request)
{
  int node, err;
  uint32_t payload_size;
  char *payload_ptr;

  assert(cmd->HasPayload());

  node = cmd->GetTargetNode();
  payload_size = cmd->GetPayloadSize();
  payload_ptr = cmd->GetPayloadPtr();

  err = MPI_Isend((const void *)payload_ptr, payload_size,
                  MPI_CHAR, node, RHAC_MPI_TAG,
                  MPI_COMM_WORLD, request);
  CHECK_MPI_ERROR(err);  
}

void Communicator::IsendCommandWithPayload(RHACCommand *cmd,
                                           MPI_Request *request)
{
  if (!cmd->HasPayload()) {
    IsendCommand(cmd, request);
  }
  else {
    MPI_Request temp_request;

    IsendCommand(cmd, &temp_request);

    IsendPayload(cmd, request);
  }
}

void Communicator::IsendResponse(RHACResponse *response,
                                 MPI_Request *request)
{
  int err;
  char *data;

  data = response->GetDataPtr();

  err = MPI_Isend((const void *)data, RESPONSE_SIZE,
                  MPI_CHAR, HOST_NODE, RHAC_MPI_TAG,
                  MPI_COMM_WORLD, request);
  CHECK_MPI_ERROR(err);
}

void Communicator::RecvCommand(RHACCommand *cmd)
{
  int err;
  char *header;

  header = cmd->GetHeaderPtr();

  err = MPI_Recv((void *)header, COMMAND_HEADER_SIZE,
                 MPI_CHAR, MPI_ANY_SOURCE,
                 RHAC_MPI_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  CHECK_MPI_ERROR(err);
}

void Communicator::RecvPayload(RHACCommand *cmd)
{
  /* not tested */
  int err;
  uint32_t payload_size;
  char *payload_ptr;

  assert(cmd->HasPayload());

  payload_size = cmd->GetPayloadSize();
  cmd->AllocPayload(payload_size);
  payload_ptr = cmd->GetPayloadPtr();

  err = MPI_Recv((void *)payload_ptr, payload_size,
                 MPI_CHAR, MPI_ANY_SOURCE,
                 RHAC_MPI_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  CHECK_MPI_ERROR(err);
}

void Communicator::IrecvCommand(RHACCommand *cmd, MPI_Request *request)
{
  int err;
  char *header;
  
  header = cmd->GetHeaderPtr();

  err = MPI_Irecv((void *)header, COMMAND_HEADER_SIZE,
                  MPI_CHAR, MPI_ANY_SOURCE,
                  RHAC_MPI_TAG,
                  MPI_COMM_WORLD, request);
  CHECK_MPI_ERROR(err);
}

void Communicator::IrecvPayload(RHACCommand *cmd,
                                MPI_Request *request)
{
  int err;
  uint32_t payload_size;
  char *payload_ptr;

  assert(cmd->HasPayload());

  payload_size = cmd->GetPayloadSize();
  cmd->AllocPayload(payload_size);
  payload_ptr = cmd->GetPayloadPtr();

  err = MPI_Irecv((void *)payload_ptr, payload_size,
                  MPI_CHAR, MPI_ANY_SOURCE,
                  RHAC_MPI_TAG,
                  MPI_COMM_WORLD, request);
  CHECK_MPI_ERROR(err);
}

void Communicator::IrecvResponse(RHACResponse *response, MPI_Request *request)
{
  int err;
  char *data;

  data = response->GetDataPtr();

  err = MPI_Irecv((void *)data, RESPONSE_SIZE,
                  MPI_CHAR, MPI_ANY_SOURCE,
                  RHAC_MPI_TAG,
                  MPI_COMM_WORLD, request);
  CHECK_MPI_ERROR(err);
}

bool Communicator::CheckFinish(MPI_Request *request)
{
  int err;
  int flag;

  err = MPI_Test(request, &flag, MPI_STATUS_IGNORE);
  CHECK_MPI_ERROR(err);

  return flag; 
}
