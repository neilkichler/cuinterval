#!/bin/sh
# --backtrace=lbr 
# --sample=process-tree 
# --trace=cuda,nvtx,osrt
# --cuda-graph-trace=graph
nsys profile --cuda-memory-usage=true --vulkan-gpu-workload=false --opengl-gpu-workload=false --trace='cuda,osrt' $1
