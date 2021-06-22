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

#include "transmitter.h"
#include "platform.h"
#include "communicator.h"
#include "rhac_command.h"
#include "rhac_response.h"
#include "rhac_barrier.h"

#define TRANSMITTER_GC_WINDOW 100

Transmitter::Transmitter(RequestQueue *request_queue)
  : Thread(AFFINITY_TYPE_0) {
  communicator_ = Communicator::GetCommunicator();
  request_queue_ = request_queue;
}

Transmitter::~Transmitter()
{
}

void Transmitter::run_()
{
  RHACCommand *cmd;
  RHACResponse *response = NULL;
  MPI_Request recv_request = MPI_REQUEST_NULL;
  MPI_Request *send_request_ptr;

  RHAC_LOG("Rank %d run transmitter", rhac_platform.GetRank());

  while (!quit_signal_) {

    /* Send Command */
    if (request_queue_->Dequeue(&cmd)) {

      // FIXME
      /* 
      RHAC_LOG("SendCommand Rank %d transmitter get cmd id %zu, cmd kind %d, node %d, dev %d", 
          rhac_platform.GetRank(),
          cmd->GetCommandID(),
          cmd->GetCommandKind(),
          cmd->GetTargetNode(),
          cmd->GetTargetDevice());
      */
      send_request_ptr = new MPI_Request;
      communicator_->IsendCommandWithPayload(cmd, send_request_ptr);
      send_commands_.push(std::make_pair(cmd, send_request_ptr));
    }

    /* Receive Response */
    if (communicator_->CheckFinish(&recv_request)) {
      // for first case
      if (response != NULL) {
        // FIXME
        /*
        RHAC_LOG("Rank %d transmitter get resposne id %zu, node %d, dev %d", 
            rhac_platform.GetRank(),
            response->GetCommandID(),
            response->GetTargetNode(),
            response->GetTargetDevice());
        */
        UpdateResponse(response);
        delete response;


      }
      response = new RHACResponse;
      communicator_->IrecvResponse(response, &recv_request);
    }

    // Special case for Barrier
    CheckBarrier();        


    GarbageCollect();
  }

  delete response;

  /* memory deallocation */
  std::pair<RHACCommand *, MPI_Request *> element;
  while (!send_commands_.empty()) {
    element = send_commands_.front();

    cmd = element.first;
    send_request_ptr = element.second;

    while (!communicator_->CheckFinish(send_request_ptr));

    send_commands_.pop();

    // FIXME
    cmd->PrintInfo("Transmitter GC End Delete this");

    delete cmd;
    delete send_request_ptr;
  }


  RHAC_LOG("Rank %d transmitter done", rhac_platform.GetRank());
}

bool Transmitter::CheckQuit(RHACCommand *cmd)
{
  assert(rhac_platform.IsHost());

  bool ret = false;
  static int quit_node = 0;
  static int quit_device = 0;

  if (cmd->GetCommandKind() == NExit)
    quit_node++;
  else if (cmd->GetCommandKind() == DExit)
    quit_device++;

  int cluster_nodes = rhac_platform.GetClusterNumNodes();
  int cluster_devices = rhac_platform.GetClusterNumDevices();
  int my_devices = rhac_platform.GetNodeNumDevices();

  if (quit_node >= cluster_nodes - 1
      && quit_device >= cluster_devices - my_devices)
    ret = true;

  return ret;
}

void Transmitter::Kill()
{
  quit_signal_ = true;
}

void Transmitter::UpdateResponse(RHACResponse *response)
{
  rhac_command_id_t command_id = response->GetCommandID();
  int node = response->GetTargetNode();
  int dev = response->GetTargetDevice();

  if (dev == -1) {
    /* node executor's response */
    *(rhac_platform.GetResponseQueue(node)) = command_id;
  }
  else {
    *(rhac_platform.GetResponseQueue(node, dev)) = command_id;
  }

}

void Transmitter::CheckBarrier()
{
  RHACCommand *cmd;
  RHACBarrier *barrier;
  MPI_Request *send_request_ptr;

  if (rhac_platform.RhacBarrierGet(&barrier)) {

    if (barrier->CheckFinish()) {
      for (int n = 1; n < rhac_platform.GetClusterNumNodes(); n++) {
        cmd = new RHACCommand;
        // FIXME cmd_id
        cmd->SetDefaultInfo(0, 
            GlobalBarrierEnd,
            n, -1);
        send_request_ptr = new MPI_Request;
        communicator_->IsendCommandWithPayload(cmd, send_request_ptr);
        send_commands_.push(std::make_pair(cmd, send_request_ptr));
      }

      // notify Barrier End to intra node threads
      rhac_platform.RhacBarrierWait();

      rhac_platform.RhacBarrierDeleteFront();
    }
  }
}

void Transmitter::GarbageCollect()
{
  /* memory deallocation */
  std::pair<RHACCommand *, MPI_Request *> element;
  
  if (send_commands_.size() < TRANSMITTER_GC_WINDOW)
    return;

  // FIXME
  RHAC_LOG("Transmiiter GC triggered");

  element = send_commands_.front();

  while (!send_commands_.empty() 
      && communicator_->CheckFinish(element.second))
  {
    RHACCommand *cmd;
    MPI_Request *req;

    cmd = element.first;
    req = element.second;

    send_commands_.pop();

    // FIXME
    cmd->PrintInfo("Transmitter GC Delete this");

    delete cmd;
    delete req;

    if (!send_commands_.empty())
      element = send_commands_.front();
  }
}
