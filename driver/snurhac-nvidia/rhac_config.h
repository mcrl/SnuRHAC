#ifndef __RHAC_CONFIG_H__
#define __RHAC_CONFIG_H__

#define RHAC_CHARDEV_DEFAULT_NAME "snurhac-nvidia"
#define RHAC_RDMA_DEFAULT_IPSTR "0.0.0.0"
#define RHAC_RDMA_DEFAULT_PORT 10021

//
// rhac_node
//

#define RHAC_MAX_NODES 8

//
// rhac_pdsc
//

/*
 * table -> dir -> block -> pdsc
 * block: 512 pdscs
 * dir: 32 blks <=> 16384 (16K) pdscs
 * table: at most 16K dirs <=> 4G pdscs: supports up to 1TB
 */

#define RHAC_MAX_PDIR_PER_PTABLE_SHIFT (14)
#define RHAC_PDSC_PER_PBLK_SHIFT (9)				
#define RHAC_PBLK_PER_PDIR_SHIFT (5)			
#define RHAC_PDSC_PER_PDIR_SHIFT (RHAC_PDSC_PER_PBLK_SHIFT + RHAC_PBLK_PER_PDIR_SHIFT)
#define RHAC_PBLK_SIZE_SHIFT (RHAC_PDSC_PER_PBLK_SHIFT + PAGE_SHIFT)
#define RHAC_PDIR_SIZE_SHIFT (RHAC_PDSC_PER_PDIR_SHIFT + PAGE_SHIFT)

#define RHAC_MAX_PDIR_PER_PTABLE (1 << RHAC_MAX_PDIR_PER_PTABLE_SHIFT)
#define RHAC_PDSC_PER_PBLK (1 << RHAC_PDSC_PER_PBLK_SHIFT)
#define RHAC_PBLK_PER_PDIR (1 << RHAC_PBLK_PER_PDIR_SHIFT)
#define RHAC_PDSC_PER_PDIR (1 << RHAC_PDSC_PER_PDIR_SHIFT) 		             
#define RHAC_PBLK_SIZE (1ULL << RHAC_PBLK_SIZE_SHIFT)
#define RHAC_PDIR_SIZE (1ULL << RHAC_PDIR_SIZE_SHIFT)


// rhac_protocol
#define RHAC_PROT_NUM_POSTED_RECV_MSGS 2048
#define RHAC_MSG_CACHE_MSGS 2048
#define RHAC_MSG_CACHE_MAX_MSGS 2048

#define RHAC_BIDX(vaddr, base) (((vaddr) - (base)) >> RHAC_PBLK_SIZE_SHIFT)


//
// rhac_isr
//
//#define RHAC_ISR_GPU_POLICY_BLOCK

//
// rhac_rdma
//
#define RHAC_RDMA_SEND_INLINE


//
// rhac_msg_handler 
//
//#define RHAC_DUMP_TIME

// rhac_correlator
//
#define RHAC_DYNAMIC_PREFETCH_READONLY
#define RHAC_DYNAMIC_PREFETCH_NON_READONLY

#define RHAC_CORRELATOR_ASSOC       4
#define RHAC_CORRELATOR_LEVEL       3
#define RHAC_CORRELATOR_SUCCS       4

#define RHAC_CORRELATOR_ROWS        65536
#define RHAC_CORRELATOR_ROWS_BITS   16
#define RHAC_CORRELATOR_ROWS_MASK   0xFFFF

#define RHAC_CORRELATOR_QUEUE       131072
#define RHAC_CORRELATOR_QUEUE_MASK  0x1FFFF

#endif //__RHAC_CONFIG_H__
