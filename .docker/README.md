# MiniDetect — Docker Dev Environment

RTX 3090 (sm_86) + Ubuntu 24.04 host. Python 3.12, PyTorch 2.7 + CUDA 12.8, clang-18.

All `docker compose` commands must be run from the **repo root** with `-f .docker/docker-compose.yml`.

---

## Quick start

```bash
# 1. Build the image (once; ~10 min first time due to PyTorch wheels)
docker compose -f .docker/docker-compose.yml build

# 2. Drop into an interactive dev shell with GPU access
docker compose -f .docker/docker-compose.yml run --rm dev

# 3. Verify GPU is visible inside the container
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 3090

# 4. Run Jupyter Lab  →  open http://localhost:8888
docker compose -f .docker/docker-compose.yml up jupyter
```

---

## All services at a glance

| Service | What it does | Command |
|---|---|---|
| `dev` | Interactive bash shell | `docker compose -f .docker/docker-compose.yml run --rm dev` |
| `jupyter` | JupyterLab on port 8888 | `docker compose -f .docker/docker-compose.yml up jupyter` |
| `cpp-test` | cmake build + ctest (non-interactive) | `docker compose -f .docker/docker-compose.yml run --rm cpp-test` |

---

## Common workflows

### Interactive dev shell
```bash
docker compose -f .docker/docker-compose.yml run --rm dev
# you're now inside /workspace with your repo mounted live
python3 dsa/python/contains_duplicate.py
pytest dsa/python/
```

### Jupyter Lab (browser)
```bash
docker compose -f .docker/docker-compose.yml up jupyter
# open http://localhost:8888  (no token/password required)
# Ctrl-C to stop
```

### C++ DSA — one-shot build + test
```bash
docker compose -f .docker/docker-compose.yml run --rm cpp-test
# cmake + ninja + ctest all run automatically, then the container exits
```

### C++ DSA — interactive (inside dev shell)
```bash
docker compose -f .docker/docker-compose.yml run --rm dev
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
./build/dsa_tests
```

Symlink `compile_commands.json` so clangd (IDE autocomplete) picks it up:
```bash
ln -sf build/compile_commands.json compile_commands.json
```

### Rebuild after requirements.txt changes
```bash
docker compose -f .docker/docker-compose.yml build --no-cache
```

### Rebuild with your exact host UID/GID (fixes file permission issues)
```bash
UID=$(id -u) GID=$(id -g) docker compose -f .docker/docker-compose.yml build
```

---

## Directory layout inside /workspace (your repo)

```
/workspace              ← repo root, mounted live from host
├── .docker/
│   ├── Dockerfile          # x86 image definition
│   ├── Dockerfile.jetson   # arm64/Jetson stub (future)
│   ├── docker-compose.yml  # all service definitions
│   ├── requirements.txt    # Python packages
│   ├── CMakeLists.txt      # C++ build for DSA + MiniDetect
│   └── README.md           # this file
├── dsa/
│   ├── python/         # Neetcode 150 in Python
│   ├── cpp/            # Neetcode 150 in C++ (picked up by CMakeLists)
│   └── INDEX.md        # append-only problem log
├── minidetect/         # MiniDetect project (Track A deliverable)
├── notes/              # linear algebra, ML theory notes
├── blog/               # writing
└── README.md
```

---

## Jetson (future — Phase 2 C++ plan)

See `Dockerfile.jetson` for notes on the arm64/Orin transition:
- Base image: NVIDIA NGC `l4t-pytorch`
- sm_86 → sm_87
- onnxruntime TensorRT EP instead of CUDA EP
- OpenCV source build (apt package is CPU-only on Jetson)

Cross-compile from x86:
```bash
docker buildx build --platform linux/arm64 -f .docker/Dockerfile.jetson -t minidetect:jetson .
```

---

## Troubleshooting

**`could not select device driver "nvidia"`**
```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

**Permission errors on mounted files**
```bash
UID=$(id -u) GID=$(id -g) docker compose -f .docker/docker-compose.yml build
```

**PyTorch can't see GPU**
Check `nvidia-smi` works on host first, then verify `CUDA_VISIBLE_DEVICES=0` is set in `docker-compose.yml`.
