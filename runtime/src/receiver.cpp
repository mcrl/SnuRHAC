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

#include "receiver.h"
#include "platform.h"
#include "communicator.h"
#include "rhac_command.h"
#include "rhac_response.h"

#define RECEIVER_GC_WINDOW 100

Receiver::Receiver()
  : Thread(AFFINITY_TYPE_0) {
  int d, n_d;
  communicator_ = Communicator::GetCommunicator();
  n_d = rhac_platform.GetNodeNumDevices();

  node_sent_response_ = 0;
  device_sent_response_.assign(n_d, 0); 
  node_quit_id_ = 0;
  device_quit_id_.assign(n_d, 0);

  for (d = 0; d < n_d; d++) {
    device_sent_response_.push_back(0);
  }
}

Receiver::~Receiver()
{
}

void Receiver::run_()
{
  bool quit_signal = false;
  RHACCommand *cmd = NULL;
  MPI_Request recv_request = MPI_REQUEST_NULL;
  MPI_Request *send_request_ptr;
  RHACResponse *response;
  rhac_command_id_t responsed_id;
  int d, n_d = rhac_platform.GetNodeNumDevices();
  int my_rank = rhac_platform.GetRank();

  RHAC_LOG("Rank %d run Receiver ", my_rank);

  /* for first case */
  cmd = new RHACCommand;
  communicator_->IrecvCommand(cmd, &recv_request);

  while (!quit_signal) {

    // Get request
    if (communicator_->CheckFinish(&recv_request)) {

      if (cmd->HasPayload() && cmd->GetPayloadPtr() == NULL) {
        /* payload data is NOT received yet*/
        communicator_->IrecvPayload(cmd, &recv_request);
      }
      else {
        quit_signal = CheckQuit(cmd);

        // CAUTION - Handle for Special Case 
        HandleSpecialCase(cmd);

        if (cmd->GetCommandKind() != GlobalBarrierEnd
            && cmd->GetCommandKind() != ADMemcpyToSymbol
            && cmd->GetCommandKind() != ADMemcpy2DToArray
            && cmd->GetCommandKind() != AGlobalBarrier)
          rhac_platform.EnqueueCommand(cmd);

        cmd = new RHACCommand;
        communicator_->IrecvCommand(cmd, &recv_request);
      }
    }

    if (quit_signal) {
      WaitQuitExecutors();
    }

    // Send Response
    
    /* Check node executor response */
    responsed_id = *(rhac_platform.GetResponseQueue(0));

    if (node_sent_response_ != responsed_id) {
      response = new RHACResponse;
      send_request_ptr = new MPI_Request;

      response->SetDefaultInfo(responsed_id, my_rank, -1);
      
      // update sent_response_
      node_sent_response_ = responsed_id;

      communicator_->IsendResponse(response, send_request_ptr);
      send_responses_.push(std::make_pair(response, send_request_ptr));
    }

    /* Check device executor response */
    for (d = 0; d < n_d; d++) {
      responsed_id = *(rhac_platform.GetResponseQueue(0, d));

      if (device_sent_response_[d] != responsed_id) {
        response = new RHACResponse;
        send_request_ptr = new MPI_Request;

        response->SetDefaultInfo(responsed_id, my_rank, d);

        // update sent_response_
        device_sent_response_[d] = responsed_id; 

        communicator_->IsendResponse(response, send_request_ptr);
        send_responses_.push(std::make_pair(response, send_request_ptr));
      }
    }

    GarbageCollect();
  }

  delete cmd;

  /* memory deallocation */
  std::pair<RHACResponse*, MPI_Request*> element;
  while (!send_responses_.empty()) {

    element = send_responses_.front();

    response = element.first;
    send_request_ptr = element.second;

    while (!communicator_->CheckFinish(send_request_ptr));

    send_responses_.pop();

    // FIXME
    response->PrintInfo("Receiver GC End Delete this");

    delete response;
    delete send_request_ptr;

  }


  RHAC_LOG("Rank %d receiver done", my_rank);
}

bool Receiver::CheckQuit(RHACCommand *cmd)
{
  static int quit_node = 0;
  static int quit_devices = 0;

  if (cmd->GetCommandKind() == NExit) {
    quit_node++;
    node_quit_id_ = cmd->GetCommandID();
  }
  else if (cmd->GetCommandKind() == DExit) {
    quit_devices++;
    device_quit_id_[cmd->GetTargetDevice()] = cmd->GetCommandID();
  }

  if (quit_node >= 1 
      && quit_devices >= rhac_platform.GetNodeNumDevices())
    return true;

  return false;
}

void Receiver::WaitQuitExecutors()
{
  int d, n_d = rhac_platform.GetNodeNumDevices();
  rhac_platform.WaitResponse(node_quit_id_, 0);

  for (d = 0; d < n_d; d++) {
    rhac_platform.WaitResponse(device_quit_id_[d], 0, d);
  }
}

void Receiver::HandleSpecialCase(RHACCommand *cmd)
{
  CommandKind cmd_kind;
  cmd_kind = cmd->GetCommandKind();
 
  if (cmd_kind == ADMemcpyToSymbol ||
      cmd_kind == ADMemcpy2DToArray) 
  {
    RHACCommandAllDevice *ad_cmd;
    ad_cmd = reinterpret_cast<RHACCommandAllDevice *>(cmd);
    ad_cmd->SetReferenceCount(rhac_platform.GetNodeNumDevices());

    for (int d = 0; d < rhac_platform.GetNodeNumDevices(); d++) {
      rhac_platform.EnqueueCommand(cmd, 0, d);
    }
  }
  else if (cmd_kind == AGlobalBarrier) {
    RHAC_LOG("Receiver Get All Command - AGlobalBarrier");
    RHACCommandAll *a_cmd;
    a_cmd = reinterpret_cast<RHACCommandAll *>(cmd);
    a_cmd->SetReferenceCount(rhac_platform.GetNodeNumDevices() + 1); // 1 for node executor

    rhac_platform.EnqueueCommand(cmd, 0);
    for (int d = 0; d < rhac_platform.GetNodeNumDevices(); d++) {
      rhac_platform.EnqueueCommand(cmd, 0, d);
    }  
  }
  else if (cmd_kind == GlobalBarrierEnd) {
    // notify Barrier End to intra node threads
    RHAC_LOG("Recevicer Start barrier Wait (node done id : %lu, dev 0 done id %lu)",
        *(rhac_platform.GetResponseQueue(0)), *(rhac_platform.GetResponseQueue(0,0)));
    rhac_platform.RhacBarrierWait();
    RHAC_LOG("Recevicer End barrier Wait");
  }

  return; 
}

void Receiver::GarbageCollect()
{
  /* memory deallocation */

  std::pair<RHACResponse*, MPI_Request*> element;

  if (send_responses_.size() < RECEIVER_GC_WINDOW)
    return;

  // FIXME
  RHAC_LOG("Rank %d Receiver GC trigger", rhac_platform.GetRank());

  element = send_responses_.front();

  while (!send_responses_.empty()
      && communicator_->CheckFinish(element.second))
  {
    RHACResponse *response;
    MPI_Request *req;

    response = element.first;
    req = element.second;

    send_responses_.pop();

    // FIXME
    response->PrintInfo("Receiver GC Delete this");

    delete response;
    delete req;

    if (!send_responses_.empty())
      element = send_responses_.front();
  }
}
