import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """LoRA layer for linear transformations."""
    
    def __init__(self, original_layer, rank=4, alpha=1.0):
        """
        Initialize LoRA layer.
        
        Args:
            original_layer: The original linear layer to wrap
            rank: The rank of the LoRA decomposition
            alpha: Scaling parameter for LoRA
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(original_layer.in_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through LoRA layer.
        
        The LoRA forward pass should:
        1. Compute the original layer output
        2. Compute the LoRA path: x -> A -> B -> scale
            2a: Apply Layer A (down-projection)
            2b: Apply Layer B (up-projection)  
            2c: Multiply by the scale
        3. Add the LoRA output to the original output
        """
        # Original forward pass
        original_output = self.original_layer(x)
        
        # LoRA path: x @ A @ B * scaling
        lora_output = x @ self.lora_A @ self.lora_B
        lora_output = lora_output * self.scaling
        
        # Add LoRA delta to original
        return original_output + lora_output


def apply_lora(model, rank=4, alpha=1.0):
    """
    Apply LoRA to attention projection layers in the model.
    """
    modules_replaced = 0
    
    def apply_lora_recursive(parent_module, module_name=""):
        nonlocal modules_replaced
        
        for name, child_module in parent_module.named_children():
            full_name = f"{module_name}.{name}" if module_name else name
            
            # Apply LoRA only to attention projection layers
            if isinstance(child_module, nn.Linear):
                if any(proj in full_name for proj in ['compute_query', 'compute_key', 'compute_value', 'compute_output']):
                    lora_module = LoRALayer(child_module, rank=rank, alpha=alpha)
                    setattr(parent_module, name, lora_module)
                    modules_replaced += 1
                    print(f"Applied LoRA to {full_name}")
            else:
                apply_lora_recursive(child_module, full_name)
    
    apply_lora_recursive(model)
    print(f"Total LoRA layers applied: {modules_replaced}")
    return model


def count_lora_parameters(model):
    """
    Count LoRA parameters vs total parameters.
    """
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            if 'lora_' in name:
                lora_params += param_count
    
    percentage = (lora_params / total_params * 100) if total_params > 0 else 0
    return lora_params, total_params, percentage


def get_lora_optimizer_params(model):
    """
    Get only LoRA parameters for optimizer.
    """
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora_' in name:
            lora_params.append(param)
    return lora_params


def merge_lora_weights(model):
    def merge_lora_recursive(module):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALayer):
                # lora_A: (in_features, rank) = (512, 4)
                # lora_B: (rank, out_features) = (4, 512)
                # The low-rank update is A @ B = (512, 4) @ (4, 512) = (512, 512) = (in, out)
                # Then transpose to match weight layout (out, in)
                lora_delta = torch.matmul(child.lora_A, child.lora_B) * child.scaling  # (in, out)
                lora_delta = lora_delta.t()  # (out, in)

                merged_weight = child.original_layer.weight.data + lora_delta

                new_linear = nn.Linear(
                    child.original_layer.in_features,
                    child.original_layer.out_features,
                    bias=child.original_layer.bias is not None
                )
                new_linear.weight.data = merged_weight
                if child.original_layer.bias is not None:
                    new_linear.bias.data = child.original_layer.bias.data

                setattr(module, name, new_linear)
            else:
                merge_lora_recursive(child)

    merge_lora_recursive(model)
    return model