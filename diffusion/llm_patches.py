"""
SAR (Semantic Anchor Regularization) patches for the diffusion training loop.

Import and call apply_llm_patches() from main.py before training starts.
"""

import torch
import torch.nn.functional as F
import os
import sys
from ldm.util import instantiate_from_config

def patched_p_losses(self, x_start, cond, t, mask=None, noise=None):
    """Replace DDPM.p_losses to add semantic anchor regularization loss."""
    original_p_losses = self.p_losses
    loss, loss_dict = original_p_losses(x_start, cond, t, mask, noise)

    prefix = 'train' if self.training else 'val'
    if hasattr(self.embedding_manager, 'llm_regularization_loss') and hasattr(self.embedding_manager, 'use_llm_enhancement'):
        if self.embedding_manager.use_llm_enhancement:
            loss_llm_reg = self.embedding_manager.llm_regularization_loss()
            loss_dict.update({f'{prefix}/loss_llm_reg': loss_llm_reg})
            loss += loss_llm_reg
            loss_dict.update({f'{prefix}/loss': loss})

    return loss, loss_dict

def patched_instantiate_embedding_manager(self, config, embedder):
    """Replace LatentDiffusion.instantiate_embedding_manager for SAR support."""
    model = instantiate_from_config(config, embedder=embedder)

    if config.params.get("embedding_manager_ckpt", None):
        model.load(config.params.embedding_manager_ckpt)

    if hasattr(model, 'use_llm_enhancement') and model.use_llm_enhancement:
        print("Using SAR-enabled embedding manager:")
        print(f"  Sample: {getattr(model, 'sample_name', 'None')}")
        print(f"  Defect: {getattr(model, 'defect_name', 'None')}")
        print(f"  Prompt dir: {getattr(model, 'prompt_dir', 'None')}")
        print(f"  LLM weight: {getattr(model, 'llm_weight', 0.5)}")

    return model

def monkey_patch_method(cls, method_name, new_method):
    """Apply a monkey patch to a class method."""
    setattr(cls, method_name, new_method.__get__(cls, cls))
    print(f"Applied patch to {cls.__name__} :: {method_name}")

def apply_llm_patches():
    """Apply all SAR-related patches to the diffusion model."""
    try:
        from ldm.models.diffusion.ddpm import DDPM, LatentDiffusion

        print("Applying SAR patches...")

        DDPM._original_p_losses = DDPM.p_losses

        def new_p_losses(self, x_start, cond, t, mask=None, noise=None):
            loss, loss_dict = self._original_p_losses(x_start, cond, t, mask, noise)

            prefix = 'train' if self.training else 'val'
            if hasattr(self.embedding_manager, 'llm_regularization_loss') and hasattr(self.embedding_manager, 'use_llm_enhancement'):
                if self.embedding_manager.use_llm_enhancement:
                    loss_llm_reg = self.embedding_manager.llm_regularization_loss()
                    loss_dict.update({f'{prefix}/loss_llm_reg': loss_llm_reg})
                    loss += loss_llm_reg
                    loss_dict.update({f'{prefix}/loss': loss})

            return loss, loss_dict

        monkey_patch_method(DDPM, 'p_losses', new_p_losses)
        monkey_patch_method(LatentDiffusion, 'instantiate_embedding_manager', patched_instantiate_embedding_manager)

        print("SAR patches applied successfully.")
        return True
    except Exception as e:
        print(f"Failed to apply SAR patches: {e}")
        import traceback
        traceback.print_exc()
        return False
