# syntax=docker/dockerfile:1.4
FROM rust:1.79-slim-bullseye AS base

# 1) Install Python + venv support
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    build-essential curl pkg-config libssl-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) Create and activate venv
ENV VENV_DIR=/opt/venv
RUN python3 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"

# 3) Install the 'uv' CLI
RUN pip install uv

WORKDIR /app

# 4) Copy dependency descriptors
COPY pyproject.toml uv.lock ./

# 5) Sync deps
RUN uv venv --python python3 $VENV_DIR \
 && uv sync

# Pre install.
ENV VIRTUAL_ENV=$VENV_DIR
RUN uv pip install -e .

# 6) Copy your code & install it
COPY . .
ENV VIRTUAL_ENV=$VENV_DIR
RUN uv pip install -e .

# 7) Set system limits
RUN echo "* soft nofile 1048576" >> /etc/security/limits.conf && \
    echo "* hard nofile 1048576" >> /etc/security/limits.conf && \
    echo "* soft nproc 1048576" >> /etc/security/limits.conf && \
    echo "* hard nproc 1048576" >> /etc/security/limits.conf && \
    echo "root soft nofile 1048576" >> /etc/security/limits.conf && \
    echo "root hard nofile 1048576" >> /etc/security/limits.conf

# 8) Create script to check limits on startup
RUN echo '#!/bin/bash\necho "Current limits:"\nulimit -n\nulimit -u\nexec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "af"]
