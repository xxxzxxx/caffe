#!/bin/bash
# Script called by Travis to build and test Caffe.

set -e
MAKE="make --jobs=$NUM_THREADS --keep-going"

if $WITH_CMAKE; then
  mkdir build
  cd build
  CPU_ONLY=" -DCPU_ONLY=ON"
  if ! $WITH_CUDA; then
    CPU_ONLY=" -DCPU_ONLY=OFF"
  fi
  PYTHON_ARGS=""
  if [ "$PYTHON_VERSION" = "3" ]; then
    PYTHON_ARGS="$PYTHON_ARGS -Dpython_version=3 -DBOOST_LIBRARYDIR=$CONDA_DIR/lib/"
  fi
  if $WITH_IO; then
    cmake -DBUILD_python=ON -DCMAKE_BUILD_TYPE=Release $CPU_ONLY $PYTHON_ARGS -DUSE_HDF5=ON -DUSE_OPENCV=ON -DUSE_LMDB=ON -DUSE_LEVELDB=ON -DUSE_SNAPPY=ON -DCMAKE_INCLUDE_PATH="$CONDA_DIR/include/" -DCMAKE_LIBRARY_PATH="$CONDA_DIR/lib/" ..
  else
    cmake -DBUILD_python=ON -DCMAKE_BUILD_TYPE=Release $CPU_ONLY $PYTHON_ARGS -DUSE_HDF5=OFF -DUSE_OPENCV=OFF -DUSE_LMDB=OFF -DUSE_LEVELDB=OFF -DUSE_SNAPPY=OFF -DCMAKE_INCLUDE_PATH="$CONDA_DIR/include/" -DCMAKE_LIBRARY_PATH="$CONDA_DIR/lib/" ..
  fi
  $MAKE
  $MAKE pytest
  if ! $WITH_CUDA; then
    $MAKE runtest
    $MAKE lint
  fi
  $MAKE clean
  cd -
else
  if ! $WITH_CUDA; then
    export CPU_ONLY=1
  fi
  if $WITH_IO; then
    export USE_LMDB=1
    export USE_LEVELDB=1
    export USE_OPENCV=1
    export USE_SNAPPY=1
    export USE_HDF5=1
  fi
  $MAKE all test pycaffe warn lint || true
  if ! $WITH_CUDA; then
    $MAKE runtest
  fi
  $MAKE all
  $MAKE test
  $MAKE pycaffe
  $MAKE pytest
  $MAKE warn
  if ! $WITH_CUDA; then
    $MAKE lint
  fi
fi
