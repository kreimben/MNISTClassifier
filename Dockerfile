# Start from the Amazon Lambda Python base image for Python 3.10
FROM public.ecr.aws/lambda/python:3.10-x86_64

# Install system dependencies (if any)
# Note: AWS Lambda base image comes with most essentials, but you might need additional system packages
RUN yum install -y \
        gcc-c++ \
        make \
        cmake \
        git \
        curl \
        libjpeg-devel \
        libpng-devel \
    && yum clean all

# Upgrade pip and install Python dependencies from requirements-prod.txt
RUN python3 -m pip install --upgrade pip && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install lightning pillow

# Copy your model and handler code to the container
COPY . /var/task

# Set the CMD to your handler (this will be the function Lambda calls)
CMD ["app.lambda_handler"]
