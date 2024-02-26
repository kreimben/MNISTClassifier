#!/bin/bash

# 변수 설정
LAYER_NAME="pytorch-layer"
PYTHON_VERSION="3.10"
LAMBDA_LAYER_ZIP="pytorch_layer.zip"
DOCKER_IMAGE_NAME="pytorch_layer_builder"
CONTAINER_NAME="pytorch_layer_container"

# Dockerfile 생성
cat <<EOF >Dockerfile
FROM amazonlinux:2

# 환경 변수 설정
ENV PATH="/root/.local/bin:${PATH}"

# 개발 도구 및 Python 3.10 설치
RUN yum install -y python3 python3-pip zip && \
    yum install -y gcc-c++ python3-devel

# 작업 디렉토리 설정
WORKDIR /lambda_build

# 라이브러리 설치
RUN python3 -m pip install torch torchvision torchaudio --target ./python

# Lambda Layer용 ZIP 파일 생성
RUN zip -r $LAMBDA_LAYER_ZIP ./python
EOF

# Docker 이미지 빌드
docker build -t $DOCKER_IMAGE_NAME . || { echo "Docker build failed"; exit 1; }

# Docker 컨테이너 실행 및 종료
docker run --name $CONTAINER_NAME $DOCKER_IMAGE_NAME
docker cp $CONTAINER_NAME:/lambda_build/$LAMBDA_LAYER_ZIP . || { echo "Failed to copy $LAMBDA_LAYER_ZIP from container"; exit 1; }
docker rm $CONTAINER_NAME

# AWS Lambda Layer로 업로드 (AWS CLI가 설치되어 있어야 함)
aws lambda publish-layer-version --layer-name $LAYER_NAME \
--zip-file fileb://$LAMBDA_LAYER_ZIP --compatible-runtimes python3.10 \
--region ap-northeast-2 # 실제 AWS 리전으로 변경 필요

# 생성된 Dockerfile 및 ZIP 파일 정리
rm Dockerfile #$LAMBDA_LAYER_ZIP
