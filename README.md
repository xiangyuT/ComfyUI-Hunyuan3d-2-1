
# ComfyUI wrapper for [Hunyuan3D-2.1]([https://github.com/Tencent/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1))

## Models
Main model, original: [https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main)

Hunyuan3d-dit-v2-1 checkpoint to be installed in diffusion_models folder

Hunyuan3d-vae-v2-1 checkpoint to be installed in vae folder


# Installation
Dependencies, in your python env:

`pip install -r requirements.txt`

or with portable:

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Hunyuan3d-2-1\requirements.txt`

For the texturegen part compilation is needed

Minimum version for Torch is v2.6.0

Go in the folder hy3dpaint/custom_rasterizer and execute this command: `python setup.py install`

Go in the folder hy3dpaint/DifferentiableRenderer and execute this command: `python setup.py install`

---

# Xatlas upgrade procedure to fix UV wrapping high poly meshes

`python_embeded\python.exe -m pip uninstall xatlas`

in the portable root folder (`ComfyUI_windows_portable`):

`git clone --recursive https://github.com/mworchel/xatlas-python.git`

`cd .\xatlas-python\extern`

delete `xatlas` folder 

`git clone --recursive https://github.com/jpcy/xatlas`

in `xatlas-python\extern\xatlas\source\xatlas` modify `xatlas.cpp`

change line 6774: `#if 0` to `//#if 0`

change line 6778: `#endif` to `//#endif`

Finally go back to portable root (`ComfyUI_windows_portable`) folder:

`.\python_embeded\python.exe -m pip install .\xatlas-python\`

---

## Acknowledgements

I would like to thank
kijai for [https://github.com/kijai/ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)

People on Discord, TrueMike, Agee, Palindar and everyone else on this community
