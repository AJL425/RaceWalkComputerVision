# Use the official Amazon Linux 2 image for Python 3.10 Lambda
FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies
RUN yum install -y \
    gcc \
    cmake \
    make \
    unzip \
    git \
    libglvnd-devel \
    libX11-devel \
    mesa-libGL \
    ffmpeg \
    && yum clean all

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Lambda handler and code
COPY app.py .

# Set the Lambda entrypoint
CMD ["app.lambda_handler"]
