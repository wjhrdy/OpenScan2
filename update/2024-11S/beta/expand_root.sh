#!/bin/bash
if test -f "/boot/expand_root"; then
    echo "expanding root partition"
    raspi-config --expand-rootfs
    rm -fr /boot/expand_root
    shutdown -r now
fi