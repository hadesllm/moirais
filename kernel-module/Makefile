# SPDX-License-Identifier: GPL-2.0-or-later
# Out-of-tree build for the morie.ko kernel companion module.
#
# Usage:
#   make            # builds morie.ko against the running kernel headers
#   sudo insmod morie.ko
#   cat /sys/kernel/morie/license   # "GPL-2.0-or-later"
#   sudo rmmod morie
#   make clean
#
# Requires linux-headers-$(uname -r) (Debian/Ubuntu) or
# kernel-devel matching $(uname -r) (Fedora/RHEL).

obj-m += morie.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install

.PHONY: all clean install
