#!/bin/bash
# 用法：
#   ./script/build.sh              # 自动检测架构
#   ./script/build.sh amd64        # 强制 x86_64
#   ./script/build.sh arm64v8      # 强制 ARM64

ARCH=${1:-$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64v8/')}

docker build \
  --build-arg ARCH=$ARCH \
  -t m3e-large-api:latest \
  .