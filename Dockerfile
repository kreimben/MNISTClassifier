# Start from the Amazon Lambda Python base image for Python 3.10
FROM public.ecr.aws/lambda/python:3.10

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
COPY requirements-prod.txt .
RUN python -m pip install --upgrade pip && \
    pip install -r requirements-prod.txt

# Copy your model and handler code to the container
COPY . /var/task

# Set the CMD to your handler (this will be the function Lambda calls)
CMD ["app.lambda_handler"]
