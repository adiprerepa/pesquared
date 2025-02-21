# Use an Ubuntu base image compatible with ARM64
FROM ubuntu:latest

# Set the environment to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install required packages including libclang-dev
RUN apt update && apt install -y \
    clang \
    llvm \
    libclang-dev \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Set Python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Default command to open bash
CMD ["/bin/bash"]
