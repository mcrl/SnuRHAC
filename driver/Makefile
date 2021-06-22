###########################################################################
# Makefile for SnuRHAC kernel modules
###########################################################################

ifneq ($(KERNELRELEASE),)
  include $(src)/Kbuild
else
  ifdef SYSSRC
    KERNEL_SOURCES := $(SYSSRC)
  else
    KERNEL_UNAME ?= $(shell uname -r)
    KERNEL_MODLIB := /lib/modules/$(KERNEL_UNAME)
    KERNEL_SOURCES := $(shell test -d $(KERNEL_MODLIB)/source && echo $(KERNEL_MODLIB)/source || echo $(KERNEL_MODLIB)/build)
  endif

  KERNEL_OUTPUT := $(KERNEL_SOURCES)
  KBUILD_PARAMS :=

  ifdef SYSOUT
    ifneq ($(SYSOUT), $(KERNEL_SOURCES))
      KERNEL_OUTPUT := $(SYSOUT)
      KBUILD_PARAMS := KBUILD_OUTPUT=$(KERNEL_OUTPUT)
    endif
  else
    KERNEL_UNAME ?= $(shell uname -r)
    KERNEL_MODLIB := /lib/modules/$(KERNEL_UNAME)
    ifeq ($(KERNEL_SOURCES), $(KERNEL_MODLIB)/source)
      KERNEL_OUTPUT := $(KERNEL_MODLIB)/build
      KBUILD_PARAMS := KBUILD_OUTPUT=$(KERNEL_OUTPUT)
    endif
  endif

  CC ?= cc

  ifndef ARCH
    ARCH := $(shell uname -m | sed -e 's/i.86/i386/' \
      -e 's/armv[0-7]\w\+/arm/' \
      -e 's/aarch64/arm64/' \
      -e 's/ppc64le/powerpc/' \
    )
  endif

  NV_KERNEL_MODULES ?= $(wildcard nvidia nvidia-uvm snurhac-nvidia)
  NV_KERNEL_MODULES := $(filter-out $(NV_EXCLUDE_KERNEL_MODULES), \
                                    $(NV_KERNEL_MODULES))
  NV_VERBOSE ?=

  ifeq ($(NV_VERBOSE),1)
    KBUILD_PARAMS += V=1
  endif
  KBUILD_PARAMS += -C $(KERNEL_SOURCES) M=$(CURDIR)
  KBUILD_PARAMS += ARCH=$(ARCH)
  KBUILD_PARAMS += NV_KERNEL_SOURCES=$(KERNEL_SOURCES)
  KBUILD_PARAMS += NV_KERNEL_OUTPUT=$(KERNEL_OUTPUT)
  KBUILD_PARAMS += NV_KERNEL_MODULES="$(NV_KERNEL_MODULES)"
  KBUILD_PARAMS += INSTALL_MOD_DIR=kernel/drivers/video

  .PHONY: modules module clean clean_conftest modules_install
  modules clean modules_install:
	@$(MAKE) "CC=$(CC)" $(KBUILD_PARAMS) $@
	@if [ "$@" = "modules" ]; then \
	  for module in $(NV_KERNEL_MODULES); do \
	    if [ -x split-object-file.sh ]; then \
	      ./split-object-file.sh $$module.ko; \
	    fi; \
	  done; \
	fi

  # Compatibility target for scripts that may be directly calling the
  # "module" target from the old build system.

  module: modules

  # Kbuild's "clean" rule won't clean up the conftest headers on its own, and
  # clean-dirs doesn't appear to work as advertised.
  clean_conftest:
	$(RM) -r conftest
	$(RM) conftest*.c
  clean: clean_conftest

endif # KERNELRELEASE
