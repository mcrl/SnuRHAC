#ifndef __RHAC_CONFIG_H__
#define __RHAC_CONFIG_H__ 

// baremetal library paths
#define RHAC_LIB_CUDART "/lib64/libcudart.so"
#define RHAC_LIB_CUDADRV "/usr/lib/x86_64-linux-gnu/libcuda.so"
#define RHAC_DRIVER_FILE "/dev/snurhac-nvidia"
#define NV_UVM_DRIVER_FILE "/dev/nvidia-uvm"

// flag for generating partitioned kernel
#define NVCC_ARCH_OPTION "-gencode arch=compute_70,code=sm_70 "
#define WRAPPED_FATBIN_NAME "snurhac_wrapped.fatbin"
#define KERNEL_INFO_FILE_NAME "kernel_info.snurhac"
#define VAR_INFO_FILE_NAME "var_info.snurhac"

// host node rank
#define HOST_NODE 0

// flag for debugging
//#define RHAC_LOGGING

// flag for debugging NVIDIA UVM Driver
//#define DEBUG_NV_UVM_DRIVER

// flag for using prefetching
#define RHAC_PREFETCH
#define READDUP_FLAG_CACHING

#define MAX_NUM_EXPRESSIONS 64
#define MAX_PREFETCH_SIZE (2*MB)
#define ACCESS_DENSITY_THRESHOLD 0.01f
#define ONE_GPU_OVERLAPPING_THRESHOLD 0.50f
#define PREFETCH_SCHEDULING_QUEUE_SIZE 16384

// flag for additional memcpy thread
#define RHAC_MEMCPY_HELPER
#define NUM_MEMCPY_THREAD 8
#define MEMCPY_CHUNK_SIZE (2*1024*1024)

// internal data
#define QUEUE_SIZE 100
#define RHAC_MPI_TAG 10

// NVIDIA variables
#define UVM_VA_BLOCK_SIZE (2*MB)
#define UVM_VA_BLOCK_SIZE_BITS 21
#define UVM_VA_BLOCK_SIZE_MASK 0x1FFFFF
#define PAGES_PER_UVM_VA_BLOCK 512

#endif // __RHAC_CONFIG_H__
