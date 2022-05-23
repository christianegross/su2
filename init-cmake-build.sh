#!/bin/bash
# Copyright © 2017 Martin Ueding <dev@martin-ueding.de>
# Licensed under the GNU Public License version 3


set -e
set -u


if [[ -d build ]]; then
    echo "Directory build exists, this script will do nothing."
    exit 1
fi

mkdir -p build/debug
mkdir -p build/release

d1=$(pwd -P)

if ! [ -d external/yaml-cpp/ ]; then
  # installing yaml-cpp
  git clone https://github.com/jbeder/yaml-cpp external/yaml-cpp
#  rm -rf external/yaml-cpp/build/ 
  cd external/yaml-cpp
  mkdir build 
  cd build
  cmake ..
  make -j8
  cd $d1
fi

YAML_SRC_PATH=$d1/external/yaml-cpp/

pushd build/debug
cmake \
  ../.. -DCMAKE_BUILD_TYPE=Debug \
  -D YAML_SRC=$YAML_SRC_PATH 
popd


pushd build/release
cmake \
  ../.. -DCMAKE_BUILD_TYPE=Release \
  -D YAML_SRC=$YAML_SRC_PATH 
popd
