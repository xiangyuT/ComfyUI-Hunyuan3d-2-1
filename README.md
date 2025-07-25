# 🌀 ComfyUI Wrapper for [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)

> **ComfyUI integration** for Tencent's powerful **Hunyuan3D-2.1** model. Supports textured 3D generation with optional high-quality UV mapping.

---

## 📦 Repository & Models

* **GitHub:** [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
* **Model Weights (HuggingFace):**
  👉 [Main page](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main)

### 🔧 Required Checkpoints

Place the following checkpoints into the appropriate folders under `ComfyUI`:

```
ComfyUI/
├── models/
│   ├── diffusion_models/
│   │   └── hunyuan3d-dit-v2-1.ckpt
│   ├── vae/
│   │   └── hunyuan3d-vae-v2-1.ckpt
```

---

## ⚙️ Installation Guide

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> 📌 **Note**: Requires **Torch >= 2.6.0**

---

### 2. Compile TextureGen Modules

You need to build two C++ extensions:

```bash
# Compile custom rasterizer
cd hy3dpaint/custom_rasterizer
python setup.py install

# Compile differentiable renderer
cd ../DifferentiableRenderer
python setup.py install
```

> ✅ You can also try precompiled `.whl` files under the `dist/` folders of each directory.

---

## 🩻 Fix UV Wrapping for High Poly Meshes (Xatlas Upgrade)

```bash
# Step 1: Uninstall existing xatlas
python_embeded\python.exe -m pip uninstall xatlas

# Step 2: Clone updated xatlas wrapper
cd ComfyUI_windows_portable
git clone --recursive https://github.com/mworchel/xatlas-python.git

# Step 3: Replace internal xatlas source
cd .\xatlas-python\extern
del /s /q xatlas
git clone --recursive https://github.com/jpcy/xatlas

# Step 4: Patch xatlas.cpp
# File: xatlas-python/extern/xatlas/source/xatlas/xatlas.cpp

# Change:
#   Line 6774:   #if 0      →   //#if 0
#   Line 6778:   #endif     →   //#endif

# Step 5: Reinstall patched version
cd ../../..
.\python_embeded\python.exe -m pip install .\xatlas-python\
```

---

## 📂 Directory Overview

```
ComfyUI/
├── hy3dpaint/
│   ├── custom_rasterizer/         # Contains custom rasterizer module
│   │   ├── setup.py
│   │   └── dist/
│   ├── DifferentiableRenderer/    # Differentiable renderer
│   │   ├── setup.py
│   │   └── dist/
├── models/
│   ├── diffusion_models/
│   │   └── hunyuan3d-dit-v2-1.ckpt
│   └── vae/
│       └── hunyuan3d-vae-v2-1.ckpt
├── xatlas-python/                 # Patched UV unwrapper
│   └── extern/
│       └── xatlas/
```

---

## 🙏 Acknowledgements

* **[kijai](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)** for the initial wrapper
* TrueMike, Agee, Palindar, and the amazing community on Discord
* The Tencent team for [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
