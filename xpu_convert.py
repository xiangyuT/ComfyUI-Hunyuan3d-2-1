import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from typing import Callable, List, Optional, Tuple, Union
import math

_original_layer_norm_forward = nn.LayerNorm.forward

def _new_layer_norm_forward(self, hidden_states: torch.Tensor):
    if (
        hidden_states.device.type == 'xpu' and 
        hidden_states.dtype in (torch.float, torch.half) and
        self.weight is not None
    ):
        try:
            import bigdl_core
            hidden_size = math.prod(self.normalized_shape)
            x_2d = hidden_states.reshape(-1, hidden_size).contiguous()
            output = bigdl_core.layer_norm(x_2d, self.weight, self.bias, self.eps)
            return output.reshape(hidden_states.shape)

        except ImportError:
            return _original_layer_norm_forward(self, hidden_states)
    else:
        print(hidden_states.dtype)
        return _original_layer_norm_forward(self, hidden_states)


_original_F_sdpa = F.scaled_dot_product_attention
def chunk_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    chunk_size=1024,
):
    if chunk_size is None or query.size(2) <= chunk_size:
        return _original_F_sdpa(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale
        )

    if scale is not None:
        return _original_F_sdpa(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale
        )
    
    if is_causal:
        warnings.warn("Chunked computation may not work correctly with causal attention. "
                      "Consider setting chunk_size=None for causal attention.")
    
    if dropout_p > 0:
        warnings.warn("Dropout is applied independently to each chunk, which may "
                      "result in slightly different behavior compared to non-chunked version.")
    
    Lq = query.size(2)
    query_chunks = torch.split(query, chunk_size, dim=2)
    
    mask_chunks = None
    if attn_mask is not None:
        split_dim = -2 if attn_mask.dim() >= 2 else 0
        if attn_mask.size(split_dim) == 1:
            mask_chunks = [attn_mask] * len(query_chunks)
        elif attn_mask.size(split_dim) == Lq:
            mask_chunks = torch.split(attn_mask, chunk_size, dim=split_dim)
        else:
            raise ValueError(f"Attention mask size {attn_mask.size()} is incompatible "
                             f"with query size {query.size()} for chunked computation")
    else:
        mask_chunks = [None] * len(query_chunks)
    
    output_chunks = []
    
    for q_chunk, m_chunk in zip(query_chunks, mask_chunks):
        chunk_output = F.scaled_dot_product_attention(
            q_chunk, key, value, 
            attn_mask=m_chunk,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
        output_chunks.append(chunk_output)
    
    return torch.cat(output_chunks, dim=2)

def chunked_diffusers_attention_processor_call(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = chunk_scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states

from realesrgan import RealESRGANer
def process_on_xpu(self):
    self.output = self.model(self.img.to("xpu"))

import torch
import torch.nn.functional as F
import functools  # Used to preserve function metadata like docstrings

# Global variables to store the original function and patch status
_original_interpolate_func = None
_is_interpolate_patched = False


def patch_xpu_interpolate_to_cpu():
    """
     patches torch.nn.functional.interpolate. If an input tensor is on an XPU device,
    it will be moved to CPU for interpolation, and the result will be moved back
    to the original XPU device.
    """
    global _original_interpolate_func, _is_interpolate_patched

    if _is_interpolate_patched:
        print("torch.nn.functional.interpolate is already patched for XPU. Skipping.")
        return

    # Store the original function
    _original_interpolate_func = F.interpolate

    @functools.wraps(_original_interpolate_func)
    def _custom_interpolate(input_tensor, *args, **kwargs):
        """
        Custom wrapper for interpolate. Moves XPU tensors to CPU for computation.
        """

        if input_tensor.device.type == "xpu":
            print(
                f"Intercepted interpolate call for XPU tensor at device {input_tensor.device}. Moving to CPU for computation."
            )
            original_device = input_tensor.device

            # Move input to CPU
            input_on_cpu = input_tensor.to("cpu")

            # Call the original interpolate function on CPU
            result_on_cpu = _original_interpolate_func(input_on_cpu, *args, **kwargs)

            # Move the result back to the original XPU device
            result_on_xpu = result_on_cpu.to(original_device)
            print(
                f"Interpolation completed on CPU, result moved back to {original_device}."
            )
            return result_on_xpu
        else:
            # If not an XPU tensor, just call the original function directly
            return _original_interpolate_func(input_tensor, *args, **kwargs)

    # Replace the original function with our custom one
    F.interpolate = _custom_interpolate
    _is_interpolate_patched = True
    print(
        "Successfully patched torch.nn.functional.interpolate to handle XPU tensors on CPU."
    )


def unpatch_xpu_interpolate_to_cpu():
    """
    Restores the original torch.nn.functional.interpolate function if it was patched.
    """
    global _original_interpolate_func, _is_interpolate_patched

    if not _is_interpolate_patched:
        print(
            "torch.nn.functional.interpolate is not currently patched. Skipping unpatch."
        )
        return

    if _original_interpolate_func is not None:
        F.interpolate = _original_interpolate_func
        _original_interpolate_func = None
        _is_interpolate_patched = False
        print("Successfully unpatched torch.nn.functional.interpolate.")
    else:
        print("Error: Could not unpatch. Original function reference missing.")

def convert_to_xpu():
    nn.LayerNorm.forward = _new_layer_norm_forward
    # AttnProcessor2_0.__call__ = chunked_diffusers_attention_processor_call
    F.scaled_dot_product_attention = chunk_scaled_dot_product_attention
    RealESRGANer.process = process_on_xpu
    patch_xpu_interpolate_to_cpu()

    print("Converted to XPU compatible functions.")
