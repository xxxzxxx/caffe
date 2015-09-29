#!/usr/bin/env sh

TOOLS=./build/tools
GPU_ID=3

$TOOLS/caffe train \
	--solver=examples/highways/cifar100-19layers/cifar100_solver.prototxt \
	--gpu=$GPU_ID \
	2>&1  | tee highway.log
