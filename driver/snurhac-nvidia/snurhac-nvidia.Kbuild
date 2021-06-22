###########################################################################
# Kbuild fragment for snurhac-nvidia.ko
###########################################################################

#
# Define SNURHAC_NVIDIA_{SOURCES,OBJECTS}
#

include $(src)/snurhac-nvidia/snurhac-nvidia-sources.Kbuild
SNURHAC_NVIDIA_OBJECTS = $(patsubst %.c,%.o,$(SNURHAC_NVIDIA_SOURCES))

obj-m += snurhac-nvidia.o
snurhac-nvidia-y := $(SNURHAC_NVIDIA_OBJECTS)

SNURHAC_NVIDIA_KO = snurhac-nvidia/snurhac-nvidia.ko

SNURHAC_NVIDIA_CFLAGS += -DNVIDIA_UVM_ENABLED
SNURHAC_NVIDIA_CFLAGS += -DNVIDIA_UNDEF_LEGACY_BIT_MACROS

SNURHAC_NVIDIA_CFLAGS += -DLinux
SNURHAC_NVIDIA_CFLAGS += -D__linux__
SNURHAC_NVIDIA_CFLAGS += -I$(src)/nvidia-uvm
SNURHAC_NVIDIA_CFLAGS += -I$(src)/common/inc
SNURHAC_NVIDIA_CFLAGS += -I$(SNURHACROOT)/runtime/src

SNURHAC_NVIDIA_CFLAGS += -Wno-declaration-after-statement

$(call ASSIGN_PER_OBJ_CFLAGS, $(SNURHAC_NVIDIA_OBJECTS), $(SNURHAC_NVIDIA_CFLAGS))

#
# Register the conftests needed by snurhac-nvidia.ko
#

NV_OBJECTS_DEPEND_ON_CONFTEST += $(SNURHAC_NVIDIA_OBJECTS)

NV_CONFTEST_TYPE_COMPILE_TESTS += file_inode
NV_CONFTEST_TYPE_COMPILE_TESTS += file_operations
NV_CONFTEST_TYPE_COMPILE_TESTS += node_states_n_memory
NV_CONFTEST_TYPE_COMPILE_TESTS += timespec64
NV_CONFTEST_TYPE_COMPILE_TESTS += proc_ops
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pde_data
NV_CONFTEST_FUNCTION_COMPILE_TESTS += proc_remove
NV_CONFTEST_FUNCTION_COMPILE_TESTS += timer_setup
NV_CONFTEST_FUNCTION_COMPILE_TESTS += kthread_create_on_node
NV_CONFTEST_FUNCTION_COMPILE_TESTS += list_is_first
NV_CONFTEST_FUNCTION_COMPILE_TESTS += ktime_get_real_ts64
NV_CONFTEST_FUNCTION_COMPILE_TESTS += ktime_get_raw_ts64
NV_CONFTEST_SYMBOL_COMPILE_TESTS += is_export_symbol_present_kthread_create_on_node
