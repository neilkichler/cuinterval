#!/bin/sh
# --backtrace=lbr 
# --sample=process-tree 
# --trace=cuda,nvtx,osrt
nsys profile --vulkan-gpu-workload=false --opengl-gpu-workload=false --trace='cuda,osrt' $1
