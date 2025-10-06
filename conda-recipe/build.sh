#!/bin/bash
set -ex

make clean
make gpu-tRNA-mapper -j 2 TARGET_GPU_ARCH=all GPU_COMPILE_THREADS=8
make install INSTALL_PREFIX=$PREFIX
