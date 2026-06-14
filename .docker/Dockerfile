# ============================================================
# MiniDetect Dev Environment — x86_64 + CUDA 12.8 + RTX 3090
# Host: Ubuntu 24.04, NVIDIA RTX 3090 (sm_86), driver 595+
# Future: see .docker/Dockerfile.jetson for arm64/Jetson (sm_87)
# ============================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Detroit

# RTX 3090 = Ampere sm_86
# Set here so PyTorch and any custom CUDA kernels compile for the right arch
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda

# ── System packages ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.12 ships with Ubuntu 24.04
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # C++ toolchain
    build-essential \
    cmake \
    ninja-build \
    clang-18 \
    clang-format-18 \
    clangd-18 \
    clang-tidy-18 \
    gdb \
    valgrind \
    # OpenCV (system build — good enough for dev; replace with source build for Jetson)
    libopencv-dev \
    python3-opencv \
    # Useful dev tools
    git \
    curl \
    wget \
    vim \
    tmux \
    tree \
    htop \
    pkg-config \
    # Jupyter deps
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# Symlink clang-18 as default clang
RUN update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-18   100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 && \
    update-alternatives --install /usr/bin/cc      cc      /usr/bin/clang-18   100 && \
    update-alternatives --install /usr/bin/c++     c++     /usr/bin/clang++-18 100 && \
    update-alternatives --install /usr/bin/clangd  clangd  /usr/bin/clangd-18  100 && \
    update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-18 100

# ── Python packages ────────────────────────────────────────
COPY .docker/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# ── GoogleTest (for C++ DSA test harness) ──────────────────
RUN cd /tmp && \
    git clone --depth 1 --branch v1.14.0 https://github.com/google/googletest.git && \
    cmake -S googletest -B googletest/build -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build googletest/build && \
    cmake --install googletest/build && \
    rm -rf /tmp/googletest

# ── Python env vars ────────────────────────────────────────
ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Workspace ──────────────────────────────────────────────
WORKDIR /workspace

# ── Non-root user matching your host UID (avoids file permission issues) ──
# Ubuntu 24.04 base ships a default 'ubuntu' user at UID/GID 1000 — remove it first
ARG UID=1000
ARG GID=1000
RUN userdel -r ubuntu 2>/dev/null || true && \
    groupdel ubuntu 2>/dev/null || true && \
    groupadd -g ${GID} dev && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash dev && \
    chown -R dev:dev /workspace

USER dev

CMD ["bash"]
