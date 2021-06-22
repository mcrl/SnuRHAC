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

#include <linux/module.h>

#include "rhac_chardev.h"
#include "rhac_rdma.h"
#include "rhac_config.h"
#include "rhac_utils.h"


MODULE_AUTHOR("Center for Manycore Programming, Seoul National University");
MODULE_DESCRIPTION("RHAC Device Driver for NVIDIA GPUs");
MODULE_VERSION("1.0");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_SOFTDEP("pre: nvidia-uvm");

static char* cdev_name = RHAC_CHARDEV_DEFAULT_NAME;
module_param_named(cdev_name, cdev_name, charp, 0);

static char* rdma_server_ipstr = RHAC_RDMA_DEFAULT_IPSTR;
module_param_named(ip_addr, rdma_server_ipstr, charp, 0);

static int rdma_server_port = RHAC_RDMA_DEFAULT_PORT;
module_param_named(ip_port, rdma_server_port, int, 0);



static int __init rhacdrv_init(void) 
{
	int err = 0;

	err = rhac_chardev_create(cdev_name);
	if (err) goto fail;

	err = rhac_rdma_launch_server(rdma_server_ipstr, rdma_server_port);
	if (err) goto fail0;

	RHAC_LOG("rhacdrv module loaded");

	return 0;

fail0:
	rhac_chardev_destroy();

fail:
	RHAC_LOG("rhacdrv module load failed");
	return err;
}

static void __exit rhacdrv_exit(void) 
{
	rhac_rdma_shutdown_server();
	rhac_chardev_destroy();
	RHAC_LOG("rhacdrv module unloaded");
}


module_init(rhacdrv_init);
module_exit(rhacdrv_exit);
