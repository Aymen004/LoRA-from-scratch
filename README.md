# LoRA-from-scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) built from scratch with a focus on clarity and educational value.

## Overview

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique for large language models. It works by freezing the pre-trained model weights and injecting trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.

This implementation provides a clean, educational codebase that closely follows the original paper while maintaining readability and extensibility.

## Features

- **Modular Design**: Easy-to-understand implementation of the core LoRA concept
- **Comprehensive Layer Support**:
  - LoRA for Linear layers
  - LoRA for Embedding layers
  - LoRA for Conv2d layers
  - Merged Linear layers for attention projections (QKV)
- **Training Utilities**:
  - Parameter freezing helpers
  - Weight merging for efficient inference
  - Save and load functions for LoRA weights

## Installation

```bash
# Clone the repository
git clone https://github.com/Aymen004/LoRA-from-scratch.git
cd LoRA-from-scratch

# Install directly from the repository
pip install -e .
```

## Usage

Here's a simple example of how to use LoRA with a pre-trained model:

```python
import torch
from transformers import AutoModel
from main import Linear
from main.helper import mark_only_lora_as_trainable, save_lora_weights, load_lora_weights

# Load a pre-trained model
model = AutoModel.from_pretrained("bert-base-uncased")

# Replace Linear layers with LoRA layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # Get the parent module
        parent_name = '.'.join(name.split('.')[:-1])
        parent = model.get_submodule(parent_name)
        # Replace the module with a LoRA version
        setattr(parent, name.split('.')[-1], Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            r=8,  # LoRA rank
            lora_alpha=16,  # LoRA alpha scaling factor
            lora_dropout=0.1  # LoRA dropout
        ))

# Make only LoRA parameters trainable
mark_only_lora_as_trainable(model)

# Fine-tune the model with LoRA...
# ... your training code here ...

# Save only the LoRA weights
save_lora_weights(model, "lora_weights.pt")

# Load the LoRA weights
load_lora_weights(model, "lora_weights.pt")
```

## Advanced Usage

### Working with attention layers

```python
from main import MergedLinear

# Replace a QKV projection with LoRA
# Assuming qkv_proj is a linear layer that projects to query, key and value
qkv_proj = MergedLinear(
    in_features=hidden_size,
    out_features=3 * hidden_size,  # QKV combined
    bias=True,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    enable_lora=[True, False, True]  # Enable LoRA for Q and V only
)
```

### Merging weights for inference

```python
from main.helper import merge_lora_weights

# Merge LoRA weights with base weights for faster inference
merge_lora_weights(model)

# Run inference with the merged model
outputs = model(inputs)
```

## Project Structure

- `main/`
  - `__init__.py`: Package exports
  - `lora_layers.py`: Core LoRA layer implementations
  - `helper.py`: Utility functions for training and inference


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the implementation.