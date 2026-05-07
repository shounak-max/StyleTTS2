"""
Stub implementation of monotonic_align for Windows environments without C++ build tools.
This provides fallback functions when the compiled C extension is not available.
"""

import numpy as np
import torch


def mask_from_lens(attn_matrix, input_lengths, output_length):
    """
    Create attention mask from sequence lengths.
    
    Args:
        attn_matrix: Attention matrix [batch, seq_len, seq_len]
        input_lengths: Input sequence lengths [batch]
        output_length: Output sequence length (scalar, Tensor scalar, or Tensor [batch])
    
    Returns:
        Mask tensor of shape [batch, max_output_length, max_input_length]
    """
    batch_size = attn_matrix.shape[0] if len(attn_matrix.shape) > 0 else 1
    max_input_len = attn_matrix.shape[-1] if len(attn_matrix.shape) > 0 else 1
    
    # Handle output_length
    if isinstance(output_length, torch.Tensor):
        if output_length.numel() == 1:
            output_length = output_length.item()
            max_output_len = int(output_length)
        else:
            # Batched output_length
            max_output_len = output_length.max().item()
            output_lengths = output_length
    else:
        max_output_len = int(output_length)
        output_lengths = torch.full((batch_size,), max_output_len, device=input_lengths.device)
    
    max_output_len = min(max_output_len, attn_matrix.shape[1])
    max_input_len = min(max_input_len, attn_matrix.shape[2])
    
    # Create input mask: True for valid positions, False for padding
    input_mask = torch.arange(max_input_len).unsqueeze(0).expand(batch_size, -1).to(input_lengths.device)
    input_mask = input_mask < input_lengths.unsqueeze(1)
    
    # Create output mask: True for valid positions
    output_mask = torch.arange(max_output_len).unsqueeze(0).expand(batch_size, -1).to(input_lengths.device)
    output_mask = output_mask < output_lengths.unsqueeze(1)
    
    # Combine: [batch, max_output_len, max_input_len]
    mask = input_mask.unsqueeze(1) & output_mask.unsqueeze(2)
    
    return mask


def maximum_path_py(neg_cent, mask):
    """
    Python implementation of maximum path algorithm.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    
    neg_cent_np = neg_cent.detach().cpu().numpy().astype(np.float32)
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    
    batch_size, t_t, t_s = neg_cent_np.shape
    paths = np.zeros((batch_size, t_t, t_s), dtype=np.int32)
    
    for b in range(batch_size):
        # Forward pass to compute costs
        costs = np.full((t_t, t_s), fill_value=np.inf, dtype=np.float32)
        
        # Find actual lengths from mask
        t_t_len = int(mask_np[b, :, 0].sum())
        t_s_len = int(mask_np[b, 0, :].sum())
        
        if t_t_len == 0 or t_s_len == 0:
            continue
        
        # Initialize first cell
        costs[0, 0] = neg_cent_np[b, 0, 0]
        
        # Forward pass
        for i in range(t_t_len):
            for j in range(t_s_len):
                if i == 0 and j == 0:
                    continue
                    
                cost = neg_cent_np[b, i, j]
                min_cost = np.inf
                
                # Check three possible paths: diagonal, horizontal, vertical
                if i > 0:
                    min_cost = min(min_cost, costs[i-1, j])
                if j > 0:
                    min_cost = min(min_cost, costs[i, j-1])
                if i > 0 and j > 0:
                    min_cost = min(min_cost, costs[i-1, j-1])
                
                if min_cost != np.inf:
                    costs[i, j] = cost + min_cost
        
        # Backtrack to find path
        i, j = t_t_len - 1, t_s_len - 1
        while i > 0 or j > 0:
            paths[b, i, j] = 1
            
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # Find which previous cell led to this one
                candidates = [
                    (i-1, j, costs[i-1, j]),
                    (i, j-1, costs[i, j-1]),
                    (i-1, j-1, costs[i-1, j-1])
                ]
                best_prev = min(candidates, key=lambda x: x[2])
                i, j = best_prev[0], best_prev[1]
        
        paths[b, 0, 0] = 1
    
    return torch.from_numpy(paths).to(device=device, dtype=dtype)


def maximum_path_c(path, neg_cent, t_t_max, t_s_max):
    """
    C implementation wrapper (falls back to Python on Windows).
    This function modifies path in-place.
    """
    batch_size = neg_cent.shape[0]
    
    for b in range(batch_size):
        t_t_len = t_t_max[b]
        t_s_len = t_s_max[b]
        
        if t_t_len == 0 or t_s_len == 0:
            continue
        
        # Initialize costs array
        costs = np.full((t_t_len, t_s_len), fill_value=1e9, dtype=np.float32)
        costs[0, 0] = neg_cent[b, 0, 0]
        
        # Forward pass
        for i in range(t_t_len):
            for j in range(t_s_len):
                if i == 0 and j == 0:
                    continue
                
                min_cost = 1e9
                if i > 0:
                    min_cost = min(min_cost, costs[i-1, j])
                if j > 0:
                    min_cost = min(min_cost, costs[i, j-1])
                if i > 0 and j > 0:
                    min_cost = min(min_cost, costs[i-1, j-1])
                
                if min_cost < 1e9:
                    costs[i, j] = neg_cent[b, i, j] + min_cost
        
        # Backtrack
        i, j = t_t_len - 1, t_s_len - 1
        while i > 0 or j > 0:
            path[b, i, j] = 1
            
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                candidates = [
                    (i-1, j, costs[i-1, j] if i > 0 else 1e9),
                    (i, j-1, costs[i, j-1] if j > 0 else 1e9),
                    (i-1, j-1, costs[i-1, j-1] if (i > 0 and j > 0) else 1e9)
                ]
                best = min(candidates, key=lambda x: x[2])
                i, j = best[0], best[1]
        
        path[b, 0, 0] = 1


class CoreModule:
    """Mock core module to satisfy imports."""
    maximum_path_c = staticmethod(maximum_path_c)


# Export the module components
core = CoreModule()
