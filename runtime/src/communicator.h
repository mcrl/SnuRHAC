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

#ifndef __RHAC_COMMNUNICATOR_H__
#define __RHAC_COMMNUNICATOR_H__

#include "rhac.h"
#include "utils.h"

#include <mpi.h>

class Communicator {
  public:
    void SendCommand(RHACCommand *cmd);
    void SendPayload(RHACCommand *cmd);
    void SendCommandWithPayload(RHACCommand *cmd);

    void IsendCommand(RHACCommand *cmd, MPI_Request *request);
    void IsendPayload(RHACCommand *cmd, MPI_Request *request);
    void IsendCommandWithPayload(RHACCommand *cmd, MPI_Request *request);

    void IsendResponse(RHACResponse *response, MPI_Request *request);

    void RecvCommand(RHACCommand *cmd);
    void RecvPayload(RHACCommand *cmd);

    void IrecvCommand(RHACCommand *cmd, MPI_Request *request);
    void IrecvPayload(RHACCommand *cmd, MPI_Request *request);

    void IrecvResponse(RHACResponse *response, MPI_Request *request);

    bool CheckFinish(MPI_Request *request);

  // For singleton
  public:
    static Communicator*  GetCommunicator();
  private:
    Communicator();
    ~Communicator();

    static Communicator*  singleton_;
    static mutex_t        mutex_;
};

#endif // __RHAC_COMMNUNICATOR_H__
