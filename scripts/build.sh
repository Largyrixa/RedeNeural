#! /bin/bash

mkdir -v build
cd build && cmake .. && make && cd ..
