"""
Recall adapters for SmolLM2-compatible split-half RoPE and absolute-position KV-cache.

This module provides:
1. SplitHalfRoPE: SmolLM2's split-half rotary position encoding
2. AbsolutePositionKVCache: Position-aware wrapper for cached K/V tensors
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class SplitHalfRoPE(nn.Module):
    """
    Split-half rotary position encoding for SmolLM2 compatibility.
    
    SmolLM2 uses split-half RoPE: the first half of head dimensions and the 
    second half of head dimensions are rotated independently with frequencies 
    derived from split-half indices.
    
    For head_dim=64:
    - Positions 0-31: rotated with frequency base^(2*i/64) for i in 0..31
    - Positions 32-63: rotated with frequency base^(2*i/64) for i in 0..31 (same freqs)
    
    This differs from interleaved RoPE where even indices and odd indices alternate.
    """
    
    def __init__(self, head_dim: int = 64, rope_theta: float = 100000.0):
        """
        Args:
            head_dim: Head dimension (default 64 for SmolLM2)
            rope_theta: Frequency base for position encoding (default 100000 for SmolLM2)
        """
        super().__init__()
        if head_dim < 2 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even and >= 2, got {head_dim}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.half_dim = head_dim // 2
    
    def _compute_inv_freqs(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Compute inverse frequencies for split-half RoPE.
        
        For head_dim=64, computes 32 frequencies used for both the first and second halves.
        """
        # Create indices for the first half: [0, 1, 2, ..., half_dim-1]
        indices = torch.arange(self.half_dim, dtype=dtype, device=device)
        # Frequency formula: theta^(-2*i/d) where d is head_dim
        inv_freqs = 1.0 / (self.rope_theta ** (2.0 * indices / self.head_dim))
        return inv_freqs
    
    def _apply_rope_to_half(
        self,
        x_half: torch.Tensor,  # [*, seq, half_dim]
        positions: torch.Tensor,  # [seq]
        inv_freqs: torch.Tensor,  # [half_dim]
    ) -> torch.Tensor:
        """
        Apply RoPE to a half-dimension tensor.
        
        Args:
            x_half: Half of the head dimension [*, seq, half_dim]
            positions: Absolute positions [seq]
            inv_freqs: Inverse frequencies [half_dim]
        
        Returns:
            Rotated tensor [*, seq, half_dim]
        """
        # Compute angles: outer product of positions and inverse frequencies
        # Result shape: [seq, half_dim]
        angles = torch.einsum("s,d->sd", positions.float(), inv_freqs)
        
        # Expand angles to match batch dimensions: [*, seq, half_dim]
        # Insert dimensions to broadcast correctly
        while angles.ndim < x_half.ndim:
            angles = angles.unsqueeze(0)
        
        # Create rotation matrix with cos and sin
        # cos(theta) and sin(theta) for rotation matrix:
        # [cos(theta) -sin(theta)] [x]   [x*cos - y*sin]
        # [sin(theta)  cos(theta)] [y] = [x*sin + y*cos]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # Apply rotation: interleave (x, y) values
        # For each pair (x_i, x_{i+1}), apply the 2D rotation
        x_rot = torch.empty_like(x_half)
        x_rot[..., 0::2] = x_half[..., 0::2] * cos[..., 0::2] - x_half[..., 1::2] * sin[..., 0::2]
        x_rot[..., 1::2] = x_half[..., 0::2] * sin[..., 0::2] + x_half[..., 1::2] * cos[..., 0::2]
        
        return x_rot
    
    def forward(
        self,
        q: torch.Tensor,  # [batch, seq_q, num_heads, head_dim] or similar
        k: torch.Tensor,  # [batch, seq_k, num_heads, head_dim]
        positions_q: Optional[torch.Tensor] = None,  # [seq_q]
        positions_k: Optional[torch.Tensor] = None,  # [seq_k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply split-half RoPE to query and key tensors.
        
        Args:
            q: Query tensor with last dimension = head_dim
            k: Key tensor with last dimension = head_dim
            positions_q: Absolute positions for queries. If None, use [0, 1, ..., seq_q-1]
            positions_k: Absolute positions for keys. If None, use [0, 1, ..., seq_k-1]
        
        Returns:
            (q_rotated, k_rotated): Tensors with RoPE applied
        """
        if q.shape[-1] != self.head_dim:
            raise ValueError(
                f"Query last dimension must be {self.head_dim}, got {q.shape[-1]}"
            )
        if k.shape[-1] != self.head_dim:
            raise ValueError(
                f"Key last dimension must be {self.head_dim}, got {k.shape[-1]}"
            )
        if not q.is_floating_point():
            raise TypeError(f"Query must have floating-point dtype, got {q.dtype}")
        if not k.is_floating_point():
            raise TypeError(f"Key must have floating-point dtype, got {k.dtype}")
        if q.dtype != k.dtype:
            raise ValueError(f"Query and key must have same dtype, got {q.dtype} and {k.dtype}")
        
        # Extract sequence length from second-to-last dimension (convention: [..., seq, head_dim])
        seq_q = q.shape[-2]
        seq_k = k.shape[-2]
        
        # Default positions: [0, 1, 2, ...]
        if positions_q is None:
            positions_q = torch.arange(seq_q, dtype=q.dtype, device=q.device)
        if positions_k is None:
            positions_k = torch.arange(seq_k, dtype=k.dtype, device=k.device)
        
        if positions_q.shape[0] != seq_q:
            raise ValueError(
                f"positions_q length {positions_q.shape[0]} != query sequence length {seq_q}"
            )
        if positions_k.shape[0] != seq_k:
            raise ValueError(
                f"positions_k length {positions_k.shape[0]} != key sequence length {seq_k}"
            )
        
        # Get inverse frequencies
        inv_freqs = self._compute_inv_freqs(q.dtype, q.device)
        
        # Split Q into two halves along the last dimension
        q_first_half = q[..., :self.half_dim]
        q_second_half = q[..., self.half_dim:]
        
        # Split K into two halves along the last dimension
        k_first_half = k[..., :self.half_dim]
        k_second_half = k[..., self.half_dim:]
        
        # Apply RoPE to each half independently with the same frequencies
        q_first_rot = self._apply_rope_to_half(q_first_half, positions_q, inv_freqs)
        q_second_rot = self._apply_rope_to_half(q_second_half, positions_q, inv_freqs)
        k_first_rot = self._apply_rope_to_half(k_first_half, positions_k, inv_freqs)
        k_second_rot = self._apply_rope_to_half(k_second_half, positions_k, inv_freqs)
        
        # Concatenate rotated halves
        q_rotated = torch.cat([q_first_rot, q_second_rot], dim=-1)
        k_rotated = torch.cat([k_first_rot, k_second_rot], dim=-1)
        
        return q_rotated, k_rotated


class AbsolutePositionKVCache:
    """
    Position-aware KV cache wrapper for SmolLM2 inference.
    
    Maps between cache storage indices and absolute token positions, enabling:
    - Correct position tracking across cache append/lookup operations
    - Support for discontiguous token sequences
    - Explicit handling of cache eviction boundaries
    - Validation against missing/evicted positions
    
    Does not modify the underlying Zeta StreamingKVCache; wraps its interface.
    """
    
    def __init__(
        self,
        num_kv_heads: int = 5,
        head_dim: int = 64,
        max_cache_tokens: Optional[int] = None,
    ):
        """
        Args:
            num_kv_heads: Number of key/value heads (5 for SmolLM2)
            head_dim: Head dimension (64 for SmolLM2)
            max_cache_tokens: Maximum cached tokens (None for unlimited)
        """
        if num_kv_heads < 1:
            raise ValueError(f"num_kv_heads must be >= 1, got {num_kv_heads}")
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")
        
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_cache_tokens = max_cache_tokens
        
        # Cache storage: [batch, num_kv_heads, seq_cache, head_dim]
        self.k_cache = None
        self.v_cache = None
        
        # Position tracking: maps cache index -> absolute token position
        self.cache_positions = None  # [seq_cache]
        
        # Batch size (inferred from first append)
        self.batch_size = None
    
    def append(
        self,
        k_new: torch.Tensor,  # [batch, num_kv_heads, seq_new, head_dim]
        v_new: torch.Tensor,  # [batch, num_kv_heads, seq_new, head_dim]
        positions: Optional[torch.Tensor] = None,  # [seq_new] absolute positions
    ) -> None:
        """
        Append new K/V tokens to the cache with absolute position tracking.
        
        Args:
            k_new: New key tokens
            v_new: New value tokens
            positions: Absolute positions for new tokens. If None, auto-increment from cache end.
        """
        if k_new.ndim != 4:
            raise ValueError(f"k_new must have rank 4, got rank {k_new.ndim}")
        if v_new.ndim != 4:
            raise ValueError(f"v_new must have rank 4, got rank {v_new.ndim}")
        if not k_new.is_floating_point():
            raise TypeError(f"k_new must have floating-point dtype, got {k_new.dtype}")
        if not v_new.is_floating_point():
            raise TypeError(f"v_new must have floating-point dtype, got {v_new.dtype}")
        
        batch, heads, seq_new, head_dim = k_new.shape
        
        if k_new.dtype != v_new.dtype:
            raise ValueError(f"k_new and v_new must have same dtype, got {k_new.dtype} and {v_new.dtype}")
        if k_new.device != v_new.device:
            raise ValueError(f"k_new and v_new must be on same device, got {k_new.device} and {v_new.device}")
        if heads != self.num_kv_heads:
            raise ValueError(
                f"k_new heads {heads} != {self.num_kv_heads}"
            )
        if head_dim != self.head_dim:
            raise ValueError(
                f"k_new head_dim {head_dim} != {self.head_dim}"
            )
        if v_new.shape != k_new.shape:
            raise ValueError(f"v_new shape {v_new.shape} != k_new shape {k_new.shape}")
        
        # Initialize cache on first append
        if self.k_cache is None:
            self.batch_size = batch
            self.k_cache = k_new.clone()
            self.v_cache = v_new.clone()
            if positions is None:
                self.cache_positions = torch.arange(seq_new, dtype=torch.int64, device=k_new.device)
            else:
                if positions.shape[0] != seq_new:
                    raise ValueError(
                        f"positions length {positions.shape[0]} != seq_new {seq_new}"
                    )
                self.cache_positions = positions.clone().to(dtype=torch.int64, device=k_new.device)
        else:
            if batch != self.batch_size:
                raise ValueError(f"Batch size {batch} != cached batch size {self.batch_size}")
            
            # Determine positions for new tokens
            if positions is None:
                # Auto-increment from max position in cache
                next_pos = self.cache_positions.max().item() + 1
                new_positions = torch.arange(
                    next_pos, next_pos + seq_new, dtype=torch.int64, device=k_new.device
                )
            else:
                if positions.shape[0] != seq_new:
                    raise ValueError(
                        f"positions length {positions.shape[0]} != seq_new {seq_new}"
                    )
                new_positions = positions.to(dtype=torch.int64, device=k_new.device)
            
            # Check for duplicate positions
            combined_positions = torch.cat([self.cache_positions, new_positions])
            if combined_positions.unique().shape[0] < combined_positions.shape[0]:
                duplicates = combined_positions[combined_positions.argsort(descending=True).argsort() 
                                              != torch.arange(combined_positions.shape[0], device=combined_positions.device)]
                raise ValueError(f"Duplicate positions in cache append: {duplicates.unique().tolist()}")
            
            # Append to cache
            self.k_cache = torch.cat([self.k_cache, k_new], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=2)
            self.cache_positions = torch.cat([self.cache_positions, new_positions])
    
    def lookup(
        self,
        positions: torch.Tensor,  # [seq] absolute positions to retrieve
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve K/V cache entries for specified absolute positions.
        
        Args:
            positions: [seq] absolute positions to retrieve
        
        Returns:
            (k_retrieved, v_retrieved): [batch, num_kv_heads, seq, head_dim]
        
        Raises:
            ValueError: If any position is not in the cache
        """
        if self.k_cache is None:
            raise RuntimeError("Cache is empty; append before lookup")
        if positions.ndim != 1:
            raise ValueError(f"positions must have rank 1, got rank {positions.ndim}")
        
        seq_lookup = positions.shape[0]
        
        # Map absolute positions to cache indices
        cache_indices = torch.full(
            (seq_lookup,), -1, dtype=torch.int64, device=positions.device
        )
        for i, pos in enumerate(positions):
            # Find cache index for this position
            matches = (self.cache_positions == pos).nonzero(as_tuple=True)[0]
            if matches.shape[0] == 0:
                raise ValueError(f"Position {pos.item()} not found in cache")
            if matches.shape[0] > 1:
                raise ValueError(f"Position {pos.item()} appears multiple times in cache")
            cache_indices[i] = matches[0]
        
        # Retrieve using cache indices
        k_retrieved = self.k_cache[:, :, cache_indices, :]
        v_retrieved = self.v_cache[:, :, cache_indices, :]
        
        return k_retrieved, v_retrieved
    
    def get_all(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get full cache state.
        
        Returns:
            (k_cache, v_cache, positions) or (None, None, None) if empty
        """
        return self.k_cache, self.v_cache, self.cache_positions
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.k_cache = None
        self.v_cache = None
        self.cache_positions = None
