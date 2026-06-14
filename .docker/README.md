# MiniDetect — Docker Dev Environment

RTX 3090 (sm_86) + Ubuntu 24.04 host. Python 3.12, PyTorch 2.3 + CUDA 12.4, clang-18.

---

## Quick start

```bash
# 1. Build the image (do this once; takes ~10 min first time due to PyTorch)
docker compose build

# 2. Drop into a dev shell with GPU access
docker compose run --rm dev

# 3. Verify GPU is visible inside the container
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 3090

# 4. Run Jupyter Lab  →  open http://localhost:8888
docker compose up jupyter
```

---

## Common workflows

### Python / ML work (Track A)
```bash
docker compose run --rm dev
# you're now inside /workspace with your repo mounted live
python3 dsa/python/contains_duplicate.py
jupyter lab   # or use the jupyter service above
```

### C++ DSA problems (Track A)
```bash
docker compose run --rm cpp-test
# builds with cmake+ninja, runs ctest
```

Or interactively:
```bash
docker compose run --rm dev
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
./build/dsa_tests
```

The `compile_commands.json` in `build/` feeds clangd for IDE autocompletion — symlink it to root:
```bash
ln -sf build/compile_commands.json compile_commands.json
```

### Rebuild after requirements.txt changes
```bash
docker compose build --no-cache
```

---

## Directory layout expected inside /workspace (your repo)

```
/workspace
├── dsa/
│   ├── python/          # Neetcode 150 in Python
│   ├── cpp/             # Neetcode 150 in C++ (picked up by CMakeLists)
│   └── INDEX.md         # append-only problem log
├── minidetect/          # MiniDetect project (Track A deliverable)
├── linear_algebra/      # 3B1B notes
├── notebooks/           # Jupyter exploration
├── CMakeLists.txt       # this file
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Jetson (future — when C++ Staff plan activates)

See `Dockerfile.jetson` — it has all the notes for the arm64/Orin transition:
- Base image swap (NGC l4t-pytorch)
- sm_86 → sm_87
- onnxruntime TensorRT EP
- OpenCV source build

Cross-compile from x86:
```bash
docker buildx build --platform linux/arm64 -f Dockerfile.jetson -t minidetect:jetson .
```

---

## Troubleshooting

**`docker: Error response from daemon: could not select device driver "nvidia"`**
→ Install nvidia-container-toolkit on your host:
```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

**Permission errors on mounted files**
→ Rebuild with your actual UID/GID:
```bash
UID=$(id -u) GID=$(id -g) docker compose build
```

**PyTorch can't see GPU**
→ Check `nvidia-smi` works on host first. Then verify `CUDA_VISIBLE_DEVICES=0` in compose.
