import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Union, Tuple

class LoRALayer():
    """Base class for LoRA layers that handles shared functionality."""
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Proper dropout handling
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    """LoRA implemented in an Embedding layer."""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        
        # Only create LoRA parameters if r > 0
        if r > 0:
            # A ∈ ℝʳˣᵈ and B ∈ ℝᵏˣʳ (as per the paper)
            self.lora_A = nn.Parameter(torch.zeros((r, embedding_dim)))
            self.lora_B = nn.Parameter(torch.zeros((num_embeddings, r)))
            self.scaling = self.lora_alpha / self.r
            # Freeze the original weight
            self.weight.requires_grad = False
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # Initialize A with normal distribution as per the paper
            # Variance = 1/r means std = sqrt(1/r)
            nn.init.normal_(self.lora_A, mean=0, std=math.sqrt(1.0 / self.r))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            # Switching to training mode - unmerge weights if necessary
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # W = W_0 - BA (unmerge)
                    self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            # Switching to evaluation mode - merge weights for efficiency
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # W = W_0 + BA (merge)
                    self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            # Get the original embedding output
            result = nn.Embedding.forward(self, x)
            
            # Apply dropout to indices
            x_d = self.lora_dropout(x) if x.dtype == torch.float else x
            
            # First multiply by A (r x embedding_dim) then by B (num_embeddings x r)
            # For embeddings, we need special handling due to how embedding lookup works
            embed_indices = x_d.view(-1)
            B_selected = self.lora_B[embed_indices]  # Shape: [batch_size*seq_len, r]
            
            # Apply LoRA: result += (B⋅A)x = B(Ax)
            lora_output = B_selected @ self.lora_A  # [batch_size*seq_len, embedding_dim]
            lora_output = lora_output.view(*x.shape, -1)  # Reshape back
            
            return result + lora_output * self.scaling
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):
    """LoRA implemented in a Linear layer."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,  # Set to True for Conv2d layers
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        
        # Handle the case when fan_in_fan_out is True
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        
        # Only create LoRA parameters if r > 0
        if r > 0:
            # A ∈ ℝʳˣⁿ (r x in_features)
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            # B ∈ ℝᵏˣʳ (out_features x r)
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freeze the original weight
            self.weight.requires_grad = False
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # Initialize as per the paper: variance = 1/r means std = sqrt(1/r)
            nn.init.normal_(self.lora_A, mean=0, std=math.sqrt(1.0 / self.r))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
            
        nn.Linear.train(self, mode)
        if mode:
            # Switching to training mode - unmerge weights if necessary
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # W = W_0 - BA (unmerge)
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            # Switching to evaluation mode - merge weights for efficiency
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # W = W_0 + BA (merge)
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
            
        if self.r > 0 and not self.merged:
            # Get output from original weights
            result = F.linear(x, T(self.weight), bias=self.bias)
            
            # Apply dropout
            x_d = self.lora_dropout(x)
            
            # Apply LoRA: result += (B⋅A)x = B(Ax)
            # First multiply by A (r x in_features)
            after_A = F.linear(x_d, self.lora_A)  # [batch_size, r]
            # Then multiply by B (out_features x r)
            after_B = F.linear(after_A, self.lora_B)  # [batch_size, out_features]
            
            return result + after_B * self.scaling
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    """
    LoRA implemented in a Linear layer where only subset of the weight matrix is adapted.
    Used for split qkv projections in attention layers.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],  # Which parts of the weights to adapt (e.g., qkv)
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        assert out_features % len(enable_lora) == 0, 'out_features must be divisible by the length of enable_lora'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        
        # Handle the case when fan_in_fan_out is True
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        
        # Calculate the size of each segment (e.g., q, k, v)
        self.segment_size = out_features // len(enable_lora)
        
        # Only create LoRA parameters if r > 0 and at least one part is adapted
        if r > 0 and any(enable_lora):
            # Create A and B matrices only for parts that are adapted
            active_adapters = sum(enable_lora)
            
            # A ∈ ℝʳˣⁿ (r * active_adapters x in_features)
            self.lora_A = nn.Parameter(torch.zeros((active_adapters * r, in_features)))
            
            # Create weight pointers for B matrices
            self.lora_B = nn.ParameterDict()
            for i, adapt in enumerate(enable_lora):
                if adapt:
                    # B_i ∈ ℝᵈᵏˣʳ (segment_size x r)
                    self.lora_B[f"layer_{i}"] = nn.Parameter(
                        torch.zeros((self.segment_size, r))
                    )
            
            self.scaling = self.lora_alpha / self.r
            # Freeze the original weight
            self.weight.requires_grad = False
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # Initialize as per the paper: variance = 1/r means std = sqrt(1/r)
            nn.init.normal_(self.lora_A, mean=0, std=math.sqrt(1.0 / self.r))
            for layer_id in self.lora_B:
                nn.init.zeros_(self.lora_B[layer_id])

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
            
        nn.Linear.train(self, mode)
        if mode:
            # Switching to training mode - unmerge weights if necessary
            if self.merge_weights and self.merged:
                if self.r > 0 and any(self.enable_lora):
                    # Unmerge weights
                    delta_w = self.get_delta_weight()
                    self.weight.data -= T(delta_w) * self.scaling
                self.merged = False
        else:
            # Switching to evaluation mode - merge weights for efficiency
            if self.merge_weights and not self.merged:
                if self.r > 0 and any(self.enable_lora):
                    # Merge weights
                    delta_w = self.get_delta_weight()
                    self.weight.data += T(delta_w) * self.scaling
                self.merged = True

    def get_delta_weight(self) -> torch.Tensor:
        """Calculate the delta weight matrix from LoRA A and B matrices."""
        delta_weight = torch.zeros_like(self.weight)
        
        if hasattr(self, 'lora_A') and self.r > 0:
            adapter_index = 0
            for i, adapt in enumerate(self.enable_lora):
                if adapt:
                    # Get the segment indices
                    segment_start = i * self.segment_size
                    segment_end = (i + 1) * self.segment_size
                    
                    # Get the corresponding A and B matrices
                    A_segment = self.lora_A[adapter_index * self.r:(adapter_index + 1) * self.r]
                    B_segment = self.lora_B[f"layer_{i}"]
                    
                    # Compute the delta weight for this segment: BA
                    delta_weight[segment_start:segment_end] = B_segment @ A_segment
                    
                    adapter_index += 1
                    
        return delta_weight

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
            
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            # Get output from original weights
            result = F.linear(x, T(self.weight), bias=self.bias)
            
            # Apply dropout
            x_d = self.lora_dropout(x)
            
            # Apply LoRA for each enabled adapter
            adapter_index = 0
            for i, adapt in enumerate(self.enable_lora):
                if adapt:
                    # Get the segment indices
                    segment_start = i * self.segment_size
                    segment_end = (i + 1) * self.segment_size
                    
                    # Get the corresponding A and B matrices
                    A_segment = self.lora_A[adapter_index * self.r:(adapter_index + 1) * self.r]
                    B_segment = self.lora_B[f"layer_{i}"]
                    
                    # Apply LoRA: B(Ax) for this segment
                    after_A = F.linear(x_d, A_segment)  # [batch_size, r]
                    after_B = F.linear(after_A, B_segment)  # [batch_size, segment_size]
                    
                    # Add to the corresponding segment in the result
                    result[:, segment_start:segment_end] += after_B * self.scaling
                    
                    adapter_index += 1
                    
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class Conv2d(nn.Conv2d, LoRALayer):
    """LoRA implemented in a Conv2d layer."""
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
            
        # Only create LoRA parameters if r > 0
        if r > 0:
            # Create LoRA tensors for convolutional layers
            # For convolutions, we reshape to treat them similar to linear layers
            # A ∈ ℝʳˣ⁽ᶜᵢⁿ⨯ᵏ⨯ᵏ⁾
            self.lora_A = nn.Parameter(
                torch.zeros((r, in_channels * kernel_size[0] * kernel_size[1]))
            )
            # B ∈ ℝᶜᵒᵘᵗˣʳ
            self.lora_B = nn.Parameter(torch.zeros((out_channels, r)))
            self.scaling = self.lora_alpha / self.r
            
            # Freeze the original weight
            self.weight.requires_grad = False
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # Initialize as per the paper: variance = 1/r means std = sqrt(1/r)
            nn.init.normal_(self.lora_A, mean=0, std=math.sqrt(1.0 / self.r))
            nn.init.zeros_(self.lora_B)
            
    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if mode:
            # Switching to training mode - unmerge weights if necessary
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Reshape to original conv weight format
                    delta_w = self.get_delta_weight()
                    self.weight.data -= delta_w
                self.merged = False
        else:
            # Switching to evaluation mode - merge weights for efficiency
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Reshape to original conv weight format
                    delta_w = self.get_delta_weight()
                    self.weight.data += delta_w
                self.merged = True
                
    def get_delta_weight(self) -> torch.Tensor:
        """Calculate the delta weight for convolution from LoRA A and B matrices."""
        # B @ A gives a matrix of shape (out_channels, in_channels * kernel_size[0] * kernel_size[1])
        delta_w = self.lora_B @ self.lora_A  # (out_channels, in_channels * kernel_size[0] * kernel_size[1])
        
        # Reshape to match the conv weight shape
        delta_w = delta_w.view(
            self.weight.shape[0],  # out_channels
            self.weight.shape[1],  # in_channels
            self.weight.shape[2],  # kernel_size[0]
            self.weight.shape[3],  # kernel_size[1]
        )
        
        return delta_w * self.scaling
    
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            # Original conv output
            result = F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding, 
                self.dilation, self.groups
            )
            
            # Apply dropout
            x_d = self.lora_dropout(x)
            
            # Handle groups properly for LoRA path
            if self.groups > 1:
                # When using groups, we need to handle each group separately
                # This is a simplification - for complex group convs, consider refactoring
                x_d = x_d.reshape(x_d.shape[0] * self.groups, x_d.shape[1] // self.groups, x_d.shape[2], x_d.shape[3])
            
            # LoRA path for convolution
            # Unfold the input tensor: extract sliding local blocks (im2col)
            # (batch_size, in_channels, height, width) -> (batch_size, in_channels * kernel_size[0] * kernel_size[1], output_height * output_width)
            batch_size, in_channels, height, width = x_d.shape
            
            # Calculate output dimensions
            out_h = (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
            out_w = (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
            
            # Apply A: First project input from (in_channels * kernel_size[0] * kernel_size[1]) to r
            # For efficiency, we'll use unfold to extract patches and then apply A
            # This is equivalent to a standard convolution but with our LoRA A weights
            x_unfolded = F.unfold(
                x_d, self.kernel_size, self.dilation, self.padding, self.stride
            )  # (batch_size, in_channels * kernel_size[0] * kernel_size[1], out_h * out_w)
            
            # Apply A and B sequentially (BA)x - first A then B
            # A projection: (batch_size, in_channels * kernel_size[0] * kernel_size[1], out_h * out_w) @ (r, in_channels * kernel_size[0] * kernel_size[1]).T
            after_A = torch.matmul(
                self.lora_A,  # (r, in_channels * kernel_size[0] * kernel_size[1])
                x_unfolded    # (batch_size, in_channels * kernel_size[0] * kernel_size[1], out_h * out_w)
            )  # Shape: (batch_size, r, out_h * out_w)
            
            # B projection: (out_channels, r) @ (batch_size, r, out_h * out_w)
            after_B = torch.matmul(
                self.lora_B,  # (out_channels, r)
                after_A       # (batch_size, r, out_h * out_w)
            )  # Shape: (batch_size, out_channels, out_h * out_w)
            
            # Reshape back to standard output format
            lora_output = after_B.view(batch_size, self.out_channels, out_h, out_w)
            
            # Add LoRA output to the result with scaling
            return result + lora_output * self.scaling
        else:
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding, 
                self.dilation, self.groups
            )