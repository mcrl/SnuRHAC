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

#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/moduleparam.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>


#include "rhac_chardev.h"
#include "rhac_ctx.h"
#include "rhac_iocx.h"
#include "rhac_ioctl.h"
#include "rhac_utils.h"


static int rhac_open(struct inode *inode, struct file *file);
static int rhac_release(struct inode *inode, struct file *file);
static long rhac_ioctl(struct file *file, u32 cmd, unsigned long arg);

static dev_t rhac_devno;
static struct device *rhac_dev;
static struct cdev rhac_cdev;

static char* devnode(struct device* dev, umode_t* mode)
{
	if (mode) *mode = 0666;
	return NULL;
}

static struct class rhac_class  = {
	.name = "rhac",
	.owner = THIS_MODULE,
	.devnode = devnode,
};

struct file_operations rhac_chardev_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = rhac_ioctl,
	.open = rhac_open,
	.release = rhac_release
};

static int rhac_open(struct inode *inode, struct file *file) 
{
	struct rhac_ctx *ctx = rhac_ctx_get_global();
	if (!ctx) return -ENOMEM;

	ctx->mm = current->mm;
	file->private_data = ctx;

	return 0;
}

static int rhac_release(struct inode *inode, struct file *file) 
{
	if (file->private_data)
		rhac_ctx_destroy((struct rhac_ctx*)file->private_data);

	file->private_data = NULL;
	return 0;
}

#define IOCX_PARAM_T(func) func ## _param_t
#define IOCX_FORWARD(func)								\
	{										\
		int err = 0;								\
		IOCX_PARAM_T(func) param;						\
		err = copy_from_user(&param, (typeof(&param))arg, sizeof(param));	\
		if (err) return -EINVAL;						\
		return func(file, &param);						\
	}

static long rhac_ioctl(struct file *file, unsigned int cmd, unsigned long arg) 
{
	switch (cmd) {
		case RHAC_IOCX_INIT: IOCX_FORWARD(rhac_iocx_init);
		case RHAC_IOCX_RESERVE: IOCX_FORWARD(rhac_iocx_reserve);
		case RHAC_IOCX_SYNC: IOCX_FORWARD(rhac_iocx_sync);

		case RHAC_IOCX_SPLIT_VA_RANGE: IOCX_FORWARD(rhac_iocx_split_va_range);
		case RHAC_IOCX_TOGGLE_DUP_FLAG: IOCX_FORWARD(rhac_iocx_toggle_dup_flag);
		case RHAC_IOCX_PREFETCH_TO_CPU: IOCX_FORWARD(rhac_iocx_prefetch_to_cpu);
		case RHAC_IOCX_PREFETCH_TO_GPU: IOCX_FORWARD(rhac_iocx_prefetch_to_gpu);

		default:
			RHAC_ERR("Unknown IOCTL command %u", cmd);
			return -EINVAL;
	} 

	return 0;
}

#undef IOCX_PARAM_T
#undef IOCX_FORWARD

int rhac_chardev_create(const char *dev_name)
{

	int err;

	err = alloc_chrdev_region(&rhac_devno, 0, 1, dev_name);
	if (err) goto L1;

	err = class_register(&rhac_class);
	if (err) goto L2;

	err = -ENOMEM;
	rhac_dev = device_create(&rhac_class, NULL, rhac_devno, NULL, dev_name);
	if (!rhac_dev) goto L3;

	cdev_init(&rhac_cdev, &rhac_chardev_fops);
	rhac_cdev.owner = THIS_MODULE;

	err = cdev_add(&rhac_cdev, rhac_devno, 1);
	if (err) goto L4;

	return 0;

L4: device_destroy(&rhac_class, rhac_devno);
L3: class_unregister(&rhac_class);
L2: unregister_chrdev_region(rhac_devno, 1);
L1:
	return err;
}

void rhac_chardev_destroy(void)
{
	device_destroy(&rhac_class, rhac_devno);
	class_unregister(&rhac_class);
	cdev_del(&rhac_cdev);
	unregister_chrdev_region(rhac_devno, 1);
}
