#!/bin/bash
# 用法：
#   ./script/build.sh              # 版本号 = latest
#   ./script/build.sh 1.2.3        # 版本号 = 1.2.3
#   ./script/build.sh amd64        # 强制 x86_64，版本号 = latest
#   ./script/build.sh arm64v8 1.2.3 # 强制 ARM64，版本号 = 1.2.3

ARCH=${1:-$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64v8/')}
VERSION=${2:-latest}

# 如果第一个参数是架构，第二个参数是版本号
# 如果第一个参数不是架构（没有第二个参数），则第一个参数可能是版本号
if [ "$ARCH" != "amd64" ] && [ "$ARCH" != "arm64v8" ]; then
    VERSION=$ARCH
    ARCH=$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64v8/')
fi

IMAGE_NAME="rag-api"
echo "[INFO] 构建镜像: ${IMAGE_NAME}:${VERSION} (arch: ${ARCH})"

docker build \
  --build-arg ARCH=$ARCH \
  -t ${IMAGE_NAME}:${VERSION} \
  -t ${IMAGE_NAME}:latest \
  .