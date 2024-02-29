#!/bin/bash

# remove past results first!
docker buildx prune
docker rmi -f $(docker images -aq)

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables
AWS_ACCOUNT_ID="016538895551"
AWS_REGION="ap-northeast-2"
ECR_REPOSITORY_NAME="mnist_classifier"
IMAGE_NAME="mnist_classifier"
IMAGE_TAG="$(date +%Y%m%d)" # Use date-based tag for uniqueness

# Authenticate Docker to your default AWS ECR registry
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build the Docker image
docker buildx build --platform linux/amd64 -t $IMAGE_NAME .

# Tag the Docker image for your ECR repository
docker tag $IMAGE_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$IMAGE_TAG
docker tag $IMAGE_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:latest

# Push the image to Amazon ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:latest

echo "Image pushed to ECR successfully."

# Clean up the build results.
docker buildx prune
docker rmi -f $(docker images -aq)