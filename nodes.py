from PIL import Image, ImageSequence, ImageOps
from torch.utils.data import Dataset
import torch
import shutil
import argparse
import copy
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
from torchvision import transforms
import os
import time
import re
import numpy as np
import torch.nn.functional as F
import trimesh as Trimesh
import gc
from .hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from typing import Union, Optional, Tuple, List, Any, Callable

#painting
from .hy3dpaint.DifferentiableRenderer.MeshRender import MeshRender
from .hy3dpaint.utils.simplify_mesh_utils import remesh_mesh
from .hy3dpaint.utils.multiview_utils import multiviewDiffusionNet
from .hy3dpaint.utils.pipeline_utils import ViewProcessor
from .hy3dpaint.utils.image_super_utils import imageSuperNet
from .hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap
from .hy3dpaint.convert_utils import create_glb_with_pbr_materials
from .hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from .hy3dshape.hy3dshape.models.autoencoders import ShapeVAE

import folder_paths
import node_helpers
import hashlib

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
diffusions_dir = os.path.join(comfy_path, "models", "diffusers")

def _convert_texture_format(tex: Union[np.ndarray, torch.Tensor, Image.Image], 
                          texture_size: Tuple[int, int], device: str, force_set: bool = False) -> torch.Tensor:
    """Unified texture format conversion logic."""
    if not force_set:
        if isinstance(tex, np.ndarray):
            tex = Image.fromarray((tex * 255).astype(np.uint8))
        elif isinstance(tex, torch.Tensor):            
            tex_np = tex.cpu().numpy()

            # 2. Handle potential batch dimension (B, C, H, W) or (B, H, W, C)
            if tex_np.ndim == 4:
                if tex_np.shape[0] == 1:
                    tex_np = tex_np.squeeze(0)
                else:
                    tex_np = tex_np[0]
            
            # 3. Handle data type and channel order for PIL
            if tex_np.ndim == 3:
                if tex_np.shape[0] in [1, 3, 4] and tex_np.shape[0] < tex_np.shape[1] and tex_np.shape[0] < tex_np.shape[2]:
                    tex_np = np.transpose(tex_np, (1, 2, 0))
                elif tex_np.shape[2] in [1, 3, 4] and tex_np.shape[0] > 4 and tex_np.shape[1] > 4:
                    pass
                else:
                    raise ValueError(f"Unsupported 3D tensor shape after squeezing batch and moving to CPU. "
                                     f"Expected (C, H, W) or (H, W, C) but got {tex_np.shape}")
                
                if tex_np.shape[2] == 1:
                    tex_np = tex_np.squeeze(2) # Remove the channel dimension

            elif tex_np.ndim == 2:
                pass
            else:
                raise ValueError(f"Unsupported tensor dimension after squeezing batch and moving to CPU: {tex_np.ndim} "
                                 f"with shape {tex_np.shape}. Expected 2D or 3D image data.")

            tex_np_uint8 = (tex_np * 255).astype(np.uint8)    
            
            tex = Image.fromarray(tex_np_uint8)

        
        tex = tex.resize(texture_size).convert("RGB")
        tex = np.array(tex) / 255.0
        return torch.from_numpy(tex).to(device).float()
    else:
        if isinstance(tex, np.ndarray):
            tex = torch.from_numpy(tex)
        return tex.to(device).float()

def convert_ndarray_to_pil(texture):
    texture_size = len(texture)
    tex = _convert_texture_format(texture,(texture_size, texture_size),"cuda")
    tex = tex.cpu().numpy()
    processed_texture = (tex * 255).astype(np.uint8)
    pil_texture = Image.fromarray(processed_texture)    
    return pil_texture

def get_filename_list(folder_name: str):
    files = [f for f in os.listdir(folder_name)]
    return files
    
# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0) 

def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image.squeeze(), 0, 255).astype(np.uint8))    

def convert_pil_images_to_tensor(images):
    tensor_array = []
    
    for image in images:
        tensor_array.append(pil2tensor(image))
        
    return tensor_array
    
def convert_tensor_images_to_pil(images):
    pil_array = []
    
    for image in images:
        pil_array.append(tensor2pil(image))
        
    return pil_array    
        
class Hy3DMeshGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder"}),
                "image": ("IMAGE", {"tooltip": "Image to generate mesh from"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "tooltip": "Number of diffusion steps"}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1, "max": 30, "step": 0.1, "tooltip": "Guidance scale"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("HY3DLATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3D21Wrapper"

    def loadmodel(self, model, image, steps, guidance_scale, seed):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()
        
        seed = seed % (2**32)

        from .hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        #from .hy3dshape.hy3dshape.rembg import BackgroundRemover
        #import torchvision.transforms as T

        model_path = folder_paths.get_full_path("diffusion_models", model)
        
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            config_path=os.path.join(script_directory, 'configs', 'dit_config_2_1.yaml'),
            ckpt_path=model_path,
            offload_device=offload_device)
        
        # to_pil = T.ToPILImage()
        # image = to_pil(image[0].permute(2, 0, 1))
        
        # if image.mode == 'RGB':
            # rembg = BackgroundRemover()
            # image = rembg(image)
            
        image = tensor2pil(image)
        
        latents = pipeline(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed)
            )
            
        del pipeline
        #del vae
        
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()            
        
        return (latents,)
        
class Hy3DMultiViewsGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "camera_config": ("HY3D21CAMERA",),
                "view_size": ("INT", {"default": 512, "min": 512, "max":1024, "step":256}),
                "image": ("IMAGE", {"tooltip": "Image to generate mesh from"}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Number of steps"}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1, "max": 10, "step": 0.1, "tooltip": "Guidance scale"}),
                "texture_size": ("INT", {"default":1024,"min":512,"max":4096,"step":512}),
                "unwrap_mesh": ("BOOLEAN", {"default":True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("HY3DPIPELINE", "IMAGE","IMAGE",)
    RETURN_NAMES = ("pipeline", "albedo","mr",)
    FUNCTION = "genmultiviews"
    CATEGORY = "Hunyuan3D21Wrapper"

    def genmultiviews(self, trimesh, camera_config, view_size, image, steps, guidance_scale, texture_size, unwrap_mesh, seed):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()
        
        seed = seed % (2**32)
        
        conf = Hunyuan3DPaintConfig(view_size, camera_config["selected_camera_azims"], camera_config["selected_camera_elevs"], camera_config["selected_view_weights"], camera_config["ortho_scale"], texture_size)
        paint_pipeline = Hunyuan3DPaintPipeline(conf)
        
        image = tensor2pil(image)
        
        temp_output_path = os.path.join(comfy_path, "temp", "textured_mesh.obj")
        
        albedo, mr = paint_pipeline(mesh=trimesh, image_path=image, output_mesh_path=temp_output_path, num_steps=steps, guidance_scale=guidance_scale, unwrap=unwrap_mesh, seed=seed)
        
        albedo_tensor = []
        mr_tensor = []
        
        for pil_img in albedo:
            np_img = np.array(pil_img).astype(np.uint8)
            np_img = np_img / 255.0
            tensor_img = torch.from_numpy(np_img)
            albedo_tensor.append(tensor_img)

        for pil_img in mr:
            np_img = np.array(pil_img).astype(np.uint8)
            np_img = np_img / 255.0
            tensor_img = torch.from_numpy(np_img)
            mr_tensor.append(tensor_img)
        
        # albedo = convert_pil_images_to_tensor(albedo)
        # mr = convert_pil_images_to_tensor(mr)       
        
        return (paint_pipeline, albedo_tensor, mr_tensor,)       
        
class Hy3DBakeMultiViews:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DPIPELINE", ),
                "camera_config": ("HY3D21CAMERA", ),
                "albedo": ("IMAGE", ),
                "mr": ("IMAGE", )                
            },
        }

    RETURN_TYPES = ("HY3DPIPELINE", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("pipeline", "albedo", "albedo_mask", "mr", "mr_mask", "albedo_texture", "mr_texture",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, pipeline, camera_config, albedo, mr):
        
        albedo = convert_tensor_images_to_pil(albedo)
        mr = convert_tensor_images_to_pil(mr)
        
        texture, mask, texture_mr, mask_mr = pipeline.bake_from_multiview(albedo,mr,camera_config["selected_camera_elevs"], camera_config["selected_camera_azims"], camera_config["selected_view_weights"])
        
        texture_pil = convert_ndarray_to_pil(texture)
        #mask_pil = convert_ndarray_to_pil(mask)
        texture_mr_pil = convert_ndarray_to_pil(texture_mr)
        #mask_mr_pil = convert_ndarray_to_pil(mask_mr)
        
        texture_tensor = pil2tensor(texture_pil)
        #mask_tensor = pil2tensor(mask_pil)
        texture_mr_tensor = pil2tensor(texture_mr_pil)
        #mask_mr_tensor = pil2tensor(mask_mr_pil)
        
        return (pipeline, texture, mask, texture_mr, mask_mr, texture_tensor, texture_mr_tensor)
        
class Hy3DInPaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DPIPELINE", ),
                "albedo": ("NPARRAY", ),
                "albedo_mask": ("NPARRAY", ),
                "mr": ("NPARRAY", ),
                "mr_mask": ("NPARRAY",),
                "output_mesh_name": ("STRING",),
            },
        }

    RETURN_TYPES = ("HY3DPIPELINE", "IMAGE","IMAGE","TRIMESH", "STRING",)
    RETURN_NAMES = ("pipeline", "albedo", "mr", "trimesh", "output_glb_path")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, pipeline, albedo, albedo_mask, mr, mr_mask, output_mesh_name):
        
        #albedo = tensor2pil(albedo)
        #albedo_mask = tensor2pil(albedo_mask)
        #mr = tensor2pil(mr)
        #mr_mask = tensor2pil(mr_mask)       
        
        albedo, mr = pipeline.inpaint(albedo, albedo_mask, mr, mr_mask)
        
        pipeline.set_texture_albedo(albedo)
        pipeline.set_texture_mr(mr)
        
        output_mesh_path = os.path.join(comfy_path, "temp", f"{output_mesh_name}.obj")
        output_temp_path = pipeline.save_mesh(output_mesh_path)
        
        output_glb_path = os.path.join(comfy_path, "output", f"{output_mesh_name}.glb")
        shutil.copyfile(output_temp_path, output_glb_path)
        
        trimesh = Trimesh.load(output_glb_path)
        
        texture_pil = convert_ndarray_to_pil(albedo)
        texture_mr_pil = convert_ndarray_to_pil(mr)
        texture_tensor = pil2tensor(texture_pil)
        texture_mr_tensor = pil2tensor(texture_mr_pil)
        
        output_glb_path = f"{output_mesh_name}.glb"
        
        return (pipeline, texture_tensor, texture_mr_tensor, trimesh, output_glb_path)         
        
class Hy3D21CameraConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 0, 180", "multiline": False}),
                "camera_elevations": ("STRING", {"default": "0, 0, 0, 0, 90, -90", "multiline": False}),
                "view_weights": ("STRING", {"default": "1, 0.1, 0.5, 0.1, 0.05, 0.05", "multiline": False}),
                "ortho_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("HY3D21CAMERA",)
    RETURN_NAMES = ("camera_config",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, camera_azimuths, camera_elevations, view_weights, ortho_scale):
        angles_list = list(map(int, camera_azimuths.replace(" ", "").split(',')))
        elevations_list = list(map(int, camera_elevations.replace(" ", "").split(',')))
        weights_list = list(map(float, view_weights.replace(" ", "").split(',')))

        camera_config = {
            "selected_camera_azims": angles_list,
            "selected_camera_elevs": elevations_list,
            "selected_view_weights": weights_list,
            "ortho_scale": ortho_scale,
            }
        
        return (camera_config,)
        
class Hy3D21VAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional":{
                "vae_config": ("HY3D21VAECONFIG",),
            }
        }

    RETURN_TYPES = ("HY3DVAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3D21Wrapper"

    def loadmodel(self, model_name, vae_config=None):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        model_path = folder_paths.get_full_path("vae", model_name)

        vae_sd = load_torch_file(model_path)
        
        if(vae_config==None):
            vae_config = {
                'num_latents': 4096,
                'embed_dim': 64,
                'num_freqs': 8,
                'include_pi': False,
                'heads': 16,
                'width': 1024,
                'num_encoder_layers': 8,
                'num_decoder_layers': 16,
                'qkv_bias': False,
                'qk_norm': True,
                'scale_factor': 1.0039506158752403,
                'geo_decoder_mlp_expand_ratio': 4,
                'geo_decoder_downsample_ratio': 1,
                'geo_decoder_ln_post': True,
                'point_feats': 4,
                'pc_size': 81920,
                'pc_sharpedge_size': 0
            }

        vae = ShapeVAE(**vae_config)
        vae.load_state_dict(vae_sd)
        vae.eval().to(torch.float16)
        
        return (vae,)   
        
class Hy3D21VAEConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_latents": ("INT", {"default": 4096, "min":0, "max":256000}),
                "embed_dim": ("INT", {"default": 64, "min":0, "max":256000}),
                "num_freqs": ("INT", {"default": 8, "min":0, "max":256000}),
                "include_pi": ("BOOLEAN", {"default": False}),
                "heads": ("INT", {"default":16, "min":0, "max":256000}),
                "width": ("INT", {"default":1024, "min":0, "max":256000}),
                "num_encoder_layers": ("INT", {"default":8, "min":0, "max":256000}),
                "num_decoder_layers": ("INT", {"default":16, "min":0, "max":256000}),
                "qkv_bias": ("BOOLEAN", {"default":False}),
                "qk_norm": ("BOOLEAN", {"default":True}),
                "scale_factor": ("FLOAT", {"default":1.0039506158752403}),
                "geo_decoder_mlp_expand_ratio": ("INT", {"default":4, "min":0, "max":256000}),
                "geo_decoder_downsample_ratio": ("INT", {"default":1, "min":0, "max":256000}),
                "geo_decoder_ln_post": ("BOOLEAN", {"default":True}),
                "point_feats": ("INT", {"default":4, "min":0, "max":256000}),
                "pc_size": ("INT", {"default":81920, "min":0, "max":256000}),
                "pc_sharpedge_size": ("INT", {"default":0, "min":0, "max":256000}),
            },
        }

    RETURN_TYPES = ("HY3D21VAECONFIG",)
    RETURN_NAMES = ("vae_config",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, num_latents, embed_dim, num_freqs, include_pi, heads, width, num_encoder_layers, num_decoder_layers, qkv_bias, qk_norm, scale_factor, geo_decoder_mlp_expand_ratio, geo_decoder_downsample_ratio, geo_decoder_ln_post, point_feats, pc_size, pc_sharpedge_size):
        vae_config = {
            "num_latents": num_latents,
            "embed_dim": embed_dim,
            "num_freqs": num_freqs,
            "include_pi": include_pi,
            "heads":heads,
            "width":width,
            "num_encoder_layers":num_encoder_layers,
            "num_decoder_layers":num_decoder_layers,
            "qkv_bias":qkv_bias,
            "qk_norm":qk_norm,
            "scale_factor":scale_factor,
            "geo_decoder_mlp_expand_ratio":geo_decoder_mlp_expand_ratio,
            "geo_decoder_downsample_ratio":geo_decoder_downsample_ratio,
            "geo_decoder_ln_post":geo_decoder_ln_post,
            "point_feats":point_feats,
            "pc_size":pc_size,
            "pc_sharpedge_size":pc_sharpedge_size
            }
        
        return (vae_config,)        

class Hy3D21VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("HY3DVAE",),
                "latents": ("HY3DLATENT", ),
                "box_v": ("FLOAT", {"default": 1.01, "min": -10.0, "max": 10.0, "step": 0.001}),
                "octree_resolution": ("INT", {"default": 384, "min": 8, "max": 4096, "step": 8}),
                "num_chunks": ("INT", {"default": 8000, "min": 1, "max": 10000000, "step": 1, "tooltip": "Number of chunks to process at once, higher values use more memory, but make the process faster"}),
                "mc_level": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.0001}),
                "mc_algo": (["mc", "dmc"], {"default": "mc"}),
            },
            "optional": {
                "enable_flash_vdm": ("BOOLEAN", {"default": True}),
                "force_offload": ("BOOLEAN", {"default": False, "tooltip": "Offloads the model to the offload device once the process is done."}),
            }            
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, vae, latents, box_v, octree_resolution, mc_level, num_chunks, mc_algo, enable_flash_vdm, force_offload):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        torch.cuda.empty_cache()

        vae.to(device)
        
        vae.enable_flashvdm_decoder(enabled=enable_flash_vdm, mc_algo=mc_algo)
        
        latents = vae.decode(latents)
        outputs = vae.latents2mesh(
            latents,
            output_type='trimesh',
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=True
        )[0]
        
        if force_offload==True:
            vae.to(offload_device)
        
        outputs.mesh_f = outputs.mesh_f[:, ::-1]
        mesh_output = Trimesh.Trimesh(outputs.mesh_v, outputs.mesh_f)
        print(f"Decoded mesh with {mesh_output.vertices.shape[0]} vertices and {mesh_output.faces.shape[0]} faces")
        
        #del pipeline
        del vae
        
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        
        return (mesh_output, )        
        
class Hy3D21ResizeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "width": ("INT", {"default":1024, "min":16, "max":8192} ),
                "height": ("INT", {"default":1024, "min":16, "max":8192} ),
                "sampling": (["NEAREST","LANCZOS","BILINEAR","BICUBIC","BOX","HAMMING"], {"default":"BICUBIC"})
            },          
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, images, width, height, sampling):        
        if sampling=='NEAREST':
            resampling = Image.Resampling.NEAREST
        elif sampling=='LANCZOS':
            resampling = Image.Resampling.LANCZOS
        elif sampling=='BILINEAR':
            resampling = Image.Resampling.BILINEAR
        elif sampling=='BICUBIC':
            resampling = Image.Resampling.BICUBIC
        elif sampling=='BOX':
            resampling = Image.Resampling.BOX
        elif sampling=='HAMMING':
            resampling = Image.Resampling.HAMMING
        else:
            raise Exception('Unknown sampling')
        
        if isinstance(images, List):
            for i in range(len(images)):
                if isinstance(images[i], torch.Tensor):
                    images[i] = tensor2pil(images[i])
                images[i] = images[i].resize((width,height), resampling)
                images[i] = pil2tensor(images[i])
        elif isinstance(images, torch.Tensor):
            images = tensor2pil(images)
            images = images.resize((width,height), resampling)
            images = pil2tensor(images)
        elif isinstance(images, Image):
            images = images.resize((width,height), resampling)
            images = pil2tensor(images)
        else:
            raise Exception("Unsupported images format")                     
        
        return (images, )
        
class Hy3D21LoadImageWithTransparency:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "Hunyuan3D21Wrapper"

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", )
    RETURN_NAMES = ("image", "mask", "image_with_alpha")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        output_images_ori = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            
            output_images_ori.append(pil2tensor(i))

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
            output_image_ori = torch.cat(output_images_ori, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            output_image_ori = output_images_ori[0]

        return (output_image, output_mask, output_image_ori)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True        

NODE_CLASS_MAPPINGS = {
    "Hy3DMeshGenerator": Hy3DMeshGenerator,
    "Hy3DMultiViewsGenerator": Hy3DMultiViewsGenerator,
    "Hy3DBakeMultiViews": Hy3DBakeMultiViews,
    "Hy3DInPaint": Hy3DInPaint,
    "Hy3D21CameraConfig": Hy3D21CameraConfig,
    "Hy3D21VAELoader": Hy3D21VAELoader,
    "Hy3D21VAEDecode": Hy3D21VAEDecode,
    "Hy3D21VAEConfig": Hy3D21VAEConfig,
    "Hy3D21ResizeImages": Hy3D21ResizeImages,
    "Hy3D21LoadImageWithTransparency": Hy3D21LoadImageWithTransparency,
    }
    
NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3DMeshGenerator": "Hunyuan 3D 2.1 Mesh Generator",
    "Hy3DMultiViewsGenerator": "Hunyuan 3D 2.1 MultiViews Generator",
    "Hy3DBakeMultiViews": "Hunyuan 3D 2.1 Bake MultiViews",
    "Hy3DInPaint": "Hunyuan 3D 2.1 InPaint",
    "Hy3D21CameraConfig": "Hunyuan 3D 2.1 Camera Config",
    "Hy3D21VAELoader": "Hunyuan 3D 2.1 VAE Loader",
    "Hy3D21VAEDecode": "Hunyuan 3D 2.1 VAE Decoder",
    "Hy3D21VAEConfig": "Hunyuan 3D 2.1 VAE Config",
    "Hy3D21ResizeImages": "Hunyuan 3D 2.1 Resize Images",
    "Hy3D21LoadImageWithTransparency": "Hunyuan 3D 2.1 Load Image with Transparency"
    }