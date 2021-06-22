/*******************************************************************************
    Copyright (c) 2017-2019 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef __UVM8_HAL_TURING_FAULT_BUFFER_H__
#define __UVM8_HAL_TURING_FAULT_BUFFER_H__

#include "nvtypes.h"
#include "uvm_common.h"
#include "uvm8_gpu.h"

// There are up to 8 TPCs per GPC in Turing, and there is 1 LTP uTLB per TPC. Besides, there is one RGG uTLB per GPC.
// Each TPC has a number of clients that can make requests to its uTLB: 1xTPCCS, 1xPE, 2xT1. The client ids are local
// to their GPC and the id mapping is linear across TPCs:
// TPC_n has TPCCS_n, PE_n, T1_p, and T1_q, where p=2*n and q=p+1.
//
// NV_PFAULT_CLIENT_GPC_LTP_UTLB_n and NV_PFAULT_CLIENT_GPC_RGG_UTLB enums can be ignored. These will never be reported
// in a fault message, and should never be used in an invalidate. Therefore, we define our own values.
typedef enum {
    UVM_TURING_GPC_UTLB_ID_RGG = 0,
    UVM_TURING_GPC_UTLB_ID_LTP0 = 1,
    UVM_TURING_GPC_UTLB_ID_LTP1 = 2,
    UVM_TURING_GPC_UTLB_ID_LTP2 = 3,
    UVM_TURING_GPC_UTLB_ID_LTP3 = 4,
    UVM_TURING_GPC_UTLB_ID_LTP4 = 5,
    UVM_TURING_GPC_UTLB_ID_LTP5 = 6,
    UVM_TURING_GPC_UTLB_ID_LTP6 = 7,
    UVM_TURING_GPC_UTLB_ID_LTP7 = 8,

    UVM_TURING_GPC_UTLB_COUNT,
} uvm_turing_gpc_utlb_id_t;

static NvU32 uvm_turing_get_utlbs_per_gpc(uvm_gpu_t *gpu)
{
    UVM_ASSERT(gpu->rm_info.maxTpcPerGpc + 1 <= UVM_TURING_GPC_UTLB_COUNT);
    return gpu->rm_info.maxTpcPerGpc + 1;
}

#endif
