import torch
import torch.nn as nn

from typing import Dict

from .lora_layers import LoRALayer


# Helper function to mark modules as trainable/untrainable
def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    Freezes all parameters except LoRA parameters for training.
    
    Args:
        model: The model with LoRA layers
        bias: Whether to train bias parameters ('none', 'all', 'lora_only')
    """
    # Freeze all parameters except LoRA parameters
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
        else:
            if bias == 'none':
                p.requires_grad = False
            elif bias == 'all':
                p.requires_grad = 'bias' in n
            elif bias == 'lora_only':
                p.requires_grad = False
                # If this is a bias in a LoRA layer, enable it
                if 'bias' in n:
                    parent_name = '.'.join(n.split('.')[:-1])
                    for m_name, module in model.named_modules():
                        if m_name == parent_name and isinstance(module, LoRALayer) and module.r > 0:
                            p.requires_grad = True
                            break

# Helper function to merge LoRA weights for inference
def lora_state_dict(model: nn.Module, bias: str = 'none') -> dict:
    """
    Get the state dict with LoRA parameters.
    
    Args:
        model: The model with LoRA layers
        bias: Whether to include bias parameters ('none', 'all', 'lora_only')
    
    Returns:
        A state dict with only LoRA parameters
    """
    my_state_dict = {}
    
    for n, p in model.named_parameters():
        if 'lora_' in n:
            my_state_dict[n] = p
        elif bias == 'all' and 'bias' in n:
            my_state_dict[n] = p
        elif bias == 'lora_only' and 'bias' in n:
            # Check if this bias belongs to a LoRA layer
            parent_name = '.'.join(n.split('.')[:-1])
            for m_name, module in model.named_modules():
                if m_name == parent_name and isinstance(module, LoRALayer) and module.r > 0:
                    my_state_dict[n] = p
                    break
    
    return my_state_dict

# Add a function to save and load LoRA weights
def save_lora_weights(model: nn.Module, path: str, bias: str = 'none'):
    """
    Save only the LoRA weights to a file.
    
    Args:
        model: The model with LoRA layers
        path: Path to save the weights
        bias: Whether to save bias parameters ('none', 'all', 'lora_only')
    """
    torch.save(lora_state_dict(model, bias), path)

def load_lora_weights(model: nn.Module, path: str, bias: str = 'none'):
    """
    Load LoRA weights from a file.
    
    Args:
        model: The model with LoRA layers
        path: Path to load the weights from
        bias: Whether to load bias parameters ('none', 'all', 'lora_only')
    """
    state_dict = torch.load(path, map_location='cpu')
    
    # Filter the state dict to only include parameters that exist in the model
    filtered_state_dict = {}
    for name, param in model.named_parameters():
        if name in state_dict and (
            'lora_' in name or 
            (bias == 'all' and 'bias' in name) or
            (bias == 'lora_only' and 'bias' in name)
        ):
            filtered_state_dict[name] = state_dict[name]
    
    # Load the filtered state dict
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    
    # Print warnings for any missing or unexpected keys
    if len(missing) > 0 and not all(['lora_' not in m for m in missing]):
        print(f"Warning: Missing keys: {[m for m in missing if 'lora_' in m]}")
    if len(unexpected) > 0:
        print(f"Warning: Unexpected keys: {unexpected}")