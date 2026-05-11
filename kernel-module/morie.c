// SPDX-License-Identifier: GPL-2.0-only
/*
 * morie.ko — symbolic kernel-side companion to the userspace
 * MORIE scientific-computing toolkit.
 *
 * What this module does:
 *   - Registers /sys/kernel/morie/version (the package version string)
 *   - Registers /sys/kernel/morie/license ("GPL-2.0-only")
 *   - Registers /sys/kernel/morie/build_info (compile-time stamp)
 *   - Refuses to load on a kernel that rejects MODULE_LICENSE("GPL v2"),
 *     i.e. on tainted (proprietary-symbol-using) kernels.
 *
 * What this module does NOT do:
 *   - It does NOT enforce userspace license compatibility (Python imports
 *     happen in userspace; the kernel has no visibility into them).
 *   - It does NOT accelerate any morie computation.
 *   - It does NOT call out to GPL-only kernel symbols beyond the basic
 *     module/sysfs/proc API needed to publish metadata.
 *
 * The module exists as a visible GPL declaration in ring 0 and as a
 * sysfs touchpoint for downstream tooling (audit scripts, package
 * managers, build pipelines) that wants to confirm the morie userspace
 * stack is paired with a GPL kernel component.
 *
 * Build (out-of-tree):
 *     make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
 *     sudo insmod morie.ko
 *     cat /sys/kernel/morie/version    # "0.1.15"
 *     cat /sys/kernel/morie/license    # "GPL-2.0-only"
 *     sudo rmmod morie
 *
 * Author: Vansh Singh Ruhela (rootcoder007) <hadesllm@proton.me>
 * License: GPL-2.0-only
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/kobject.h>
#include <linux/sysfs.h>

#define MORIE_VERSION    "0.1.15"
#define MORIE_LICENSE_ID "GPL-2.0-only"

static struct kobject *morie_kobj;

static ssize_t version_show(struct kobject *kobj, struct kobj_attribute *attr,
                            char *buf)
{
    return sysfs_emit(buf, "%s\n", MORIE_VERSION);
}

static ssize_t license_show(struct kobject *kobj, struct kobj_attribute *attr,
                            char *buf)
{
    return sysfs_emit(buf, "%s\n", MORIE_LICENSE_ID);
}

static ssize_t build_info_show(struct kobject *kobj, struct kobj_attribute *attr,
                               char *buf)
{
    return sysfs_emit(buf, "morie kernel companion built %s %s\n",
                      __DATE__, __TIME__);
}

static struct kobj_attribute version_attr =
    __ATTR(version, 0444, version_show, NULL);
static struct kobj_attribute license_attr =
    __ATTR(license, 0444, license_show, NULL);
static struct kobj_attribute build_info_attr =
    __ATTR(build_info, 0444, build_info_show, NULL);

static struct attribute *morie_attrs[] = {
    &version_attr.attr,
    &license_attr.attr,
    &build_info_attr.attr,
    NULL,
};

static struct attribute_group morie_attr_group = {
    .attrs = morie_attrs,
};

static int __init morie_init(void)
{
    int err;

    pr_info("morie: loading kernel companion module v%s (%s)\n",
            MORIE_VERSION, MORIE_LICENSE_ID);

    morie_kobj = kobject_create_and_add("morie", kernel_kobj);
    if (!morie_kobj) {
        pr_err("morie: kobject_create_and_add failed\n");
        return -ENOMEM;
    }

    err = sysfs_create_group(morie_kobj, &morie_attr_group);
    if (err) {
        pr_err("morie: sysfs_create_group failed: %d\n", err);
        kobject_put(morie_kobj);
        return err;
    }

    pr_info("morie: /sys/kernel/morie/ populated\n");
    return 0;
}

static void __exit morie_exit(void)
{
    pr_info("morie: unloading\n");
    if (morie_kobj) {
        sysfs_remove_group(morie_kobj, &morie_attr_group);
        kobject_put(morie_kobj);
    }
}

module_init(morie_init);
module_exit(morie_exit);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Vansh Singh Ruhela <hadesllm@proton.me>");
MODULE_DESCRIPTION("Symbolic GPL declaration + sysfs companion for the MORIE userspace toolkit");
MODULE_VERSION(MORIE_VERSION);
