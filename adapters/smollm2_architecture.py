"""
SmolLM2 architecture shell implementation for Recall.

This module provides the complete transformer architecture components required for
SmolLM2-360M-Instruct compatibility, excluding model weights:

1. RMSNorm: Root Mean Square normalization
2. SwiGLU: Gated feed-forward network
3. Residual connections
4. Transformer block: Attention + FFN with residuals
5. Model shell: Embedding + transformer stack + output projection
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from zeta_scaling_toolkit import GroupedQueryAttention
from adapters.rope_cache_adapter import SplitHalfRoPE, AbsolutePositionKVCache


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    SmolLM2 uses RMS normalization instead of LayerNorm. This implementation
    follows the formula: normalized_x = x / sqrt(mean(x^2) + epsilon) * weight
    
    Reference: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: Feature dimension
            eps: Small epsilon for numerical stability (SmolLM2 uses 1e-5)
        """
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape [..., dim]
        
        Returns:
            Normalized tensor of same shape as input
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Input last dimension {x.shape[-1]} does not match "
                f"expected dimension {self.dim}"
            )
        
        # Compute RMS: sqrt(mean(x^2))
        # Keep dims for proper broadcasting
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        normalized = (x / rms) * self.weight
        
        return normalized


class SwiGLU(nn.Module):
    """
    Gated Linear Unit with Swish activation (SwiGLU).
    
    SmolLM2 uses SwiGLU for its feed-forward networks.
    Structure: (Linear(x) * Swish(Linear(x))) @ W_out
    
    With SmolLM2 configuration:
    - Input: hidden_size (960)
    - Gate & Up projections: intermediate_size (2560)
    - Output: hidden_size (960)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        intermediate_features: int,
        bias: bool = False,
        activation: Optional[nn.Module] = None,
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            intermediate_features: Intermediate dimension (typically 4/3 * in_features)
            bias: Whether to use bias (SmolLM2 sets bias=False)
            activation: Activation function (default: SiLU/Swish)
        """
        super().__init__()
        if in_features <= 0 or out_features <= 0 or intermediate_features <= 0:
            raise ValueError(
                f"All dimensions must be positive: "
                f"in={in_features}, out={out_features}, inter={intermediate_features}"
            )
        
        self.in_features = in_features
        self.out_features = out_features
        self.intermediate_features = intermediate_features
        
        # Gate and up-projection to intermediate dimension
        self.gate_proj = nn.Linear(in_features, intermediate_features, bias=bias)
        self.up_proj = nn.Linear(in_features, intermediate_features, bias=bias)
        
        # Down-projection back to output dimension
        self.down_proj = nn.Linear(intermediate_features, out_features, bias=bias)
        
        # Activation: SiLU (Swish) is the standard for SmolLM2
        self.activation = activation or nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape [..., in_features]
        
        Returns:
            Output tensor of shape [..., out_features]
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input last dimension {x.shape[-1]} does not match "
                f"expected dimension {self.in_features}"
            )
        
        # Compute gate: Swish(gate_proj(x))
        gate = self.activation(self.gate_proj(x))
        
        # Compute up: up_proj(x)
        up = self.up_proj(x)
        
        # Multiply gate and up
        combined = gate * up
        
        # Project down to output dimension
        output = self.down_proj(combined)
        
        return output


class TransformerBlock(nn.Module):
    """
    A single transformer block combining:
    - Self-attention with RoPE and grouped-query mechanism
    - Feed-forward network with SwiGLU
    - Residual connections with RMS normalization
    
    Follows the SmolLM2 architecture exactly.
    """
    
    def __init__(
        self,
        hidden_size: int = 960,
        num_attention_heads: int = 15,
        num_key_value_heads: int = 5,
        head_dim: int = 64,
        intermediate_size: int = 2560,
        rope_theta: float = 100000.0,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        mlp_bias: bool = False,
    ):
        """
        Args:
            hidden_size: Dimension of hidden states (960 for SmolLM2)
            num_attention_heads: Number of query attention heads (15 for SmolLM2)
            num_key_value_heads: Number of key/value heads (5 for SmolLM2)
            head_dim: Dimension per head (64 for SmolLM2)
            intermediate_size: Intermediate FFN dimension (2560 for SmolLM2)
            rope_theta: RoPE frequency base (100000 for SmolLM2)
            rms_norm_eps: RMSNorm epsilon (1e-5 for SmolLM2)
            attention_bias: Whether attention linear layers use bias (False for SmolLM2)
            mlp_bias: Whether MLP linear layers use bias (False for SmolLM2)
        """
        super().__init__()
        
        # Validate dimensions
        if hidden_size != num_attention_heads * head_dim:
            raise ValueError(
                f"hidden_size ({hidden_size}) must equal "
                f"num_attention_heads ({num_attention_heads}) * "
                f"head_dim ({head_dim})"
            )
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        
        # Pre-attention normalization
        self.attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # Attention: Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        
        # Grouped Query Attention from Zeta
        self.gqa = GroupedQueryAttention(
            num_query_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        
        # RoPE for position encoding
        self.rope = SplitHalfRoPE(head_dim=head_dim, rope_theta=rope_theta)
        
        # KV Cache wrapper
        self.kv_cache = AbsolutePositionKVCache()
        
        # Output projection after attention
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        
        # Pre-FFN normalization
        self.ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # Feed-forward network with SwiGLU
        self.mlp = SwiGLU(
            in_features=hidden_size,
            out_features=hidden_size,
            intermediate_features=intermediate_size,
            bias=mlp_bias,
            activation=nn.SiLU(),  # SmolLM2 hidden_act is 'silu'
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: Input tensor [batch, seq, hidden_size]
            attention_mask: Optional attention mask
            past_key_values: Optional cached K/V from previous positions
            use_cache: Whether to cache K/V for future positions
        
        Returns:
            Tuple of (output, cache) where cache is (key, value) if use_cache else None
        """
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input hidden dimension {hidden_states.shape[-1]} "
                f"does not match expected {self.hidden_size}"
            )
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # ============================================================
        # Self-Attention with Residual
        # ============================================================
        
        # Normalize before attention
        attn_input = self.attn_norm(hidden_states)
        
        # Project to Q, K, V and reshape to [batch, heads, seq, head_dim]
        q = self.q_proj(attn_input)
        k = self.k_proj(attn_input)
        v = self.v_proj(attn_input)
        
        # Reshape for attention computation
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        k = k.transpose(1, 2)  # [batch, kv_heads, seq, head_dim]
        
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.transpose(1, 2)  # [batch, kv_heads, seq, head_dim]
        
        # Apply RoPE position encoding
        positions = torch.arange(seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
        q_rot, k_rot = self.rope(q, k, positions)
        
        # Handle KV cache if provided
        if past_key_values is not None:
            # Concatenate cached K/V with new K/V
            past_k, past_v = past_key_values
            k_rot = torch.cat([past_k, k_rot], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        # Apply grouped query attention
        attn_output = self.gqa(q_rot, k_rot, v, is_causal=True)
        
        # attn_output is [batch, seq, num_heads * head_dim] from GQA
        # Project attention output
        attn_output = self.o_proj(attn_output)
        
        # Residual connection
        hidden_states = hidden_states + attn_output
        
        # Prepare cache if requested
        new_cache = None
        if use_cache:
            new_cache = (k_rot, v)
        
        # ============================================================
        # Feed-Forward with Residual
        # ============================================================
        
        # Normalize before FFN
        ffn_input = self.ffn_norm(hidden_states)
        
        # Apply SwiGLU
        ffn_output = self.mlp(ffn_input)
        
        # Residual connection
        output = hidden_states + ffn_output
        
        return output, new_cache


class SmolLM2ModelShell(nn.Module):
    """
    SmolLM2-360M architecture shell without model weights.
    
    This is a synthetic implementation for validation purposes, using:
    - Embedding interface (not instantiated with weights)
    - Transformer block stack
    - Final RMS normalization
    - Language model output projection interface
    """
    
    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 960,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 15,
        num_key_value_heads: int = 5,
        head_dim: int = 64,
        intermediate_size: int = 2560,
        max_position_embeddings: int = 8192,
        rope_theta: float = 100000.0,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
    ):
        """
        Args:
            vocab_size: Vocabulary size (49152 for SmolLM2)
            hidden_size: Hidden state dimension (960 for SmolLM2)
            num_hidden_layers: Number of transformer blocks (32 for SmolLM2)
            num_attention_heads: Query heads (15 for SmolLM2)
            num_key_value_heads: KV heads (5 for SmolLM2)
            head_dim: Dimension per head (64 for SmolLM2)
            intermediate_size: FFN intermediate dimension (2560 for SmolLM2)
            max_position_embeddings: Max sequence length (8192 for SmolLM2)
            rope_theta: RoPE frequency base (100000 for SmolLM2)
            rms_norm_eps: RMSNorm epsilon (1e-5 for SmolLM2)
            attention_bias: Whether attention uses bias (False for SmolLM2)
            mlp_bias: Whether MLP uses bias (False for SmolLM2)
            tie_word_embeddings: Whether to tie embedding and output projection
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        
        # Embedding interface (synthetic - weights not loaded)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                mlp_bias=mlp_bias,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # Language model output projection (not tied to embeddings - separate interface)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq]
            attention_mask: Optional attention mask
            use_cache: Whether to cache K/V states
        
        Returns:
            Tuple of (logits, caches) where:
            - logits: [batch, seq, vocab_size]
            - caches: List of (k, v) for each layer if use_cache else None
        """
        if input_ids.shape[-1] > self.max_position_embeddings:
            raise ValueError(
                f"Input sequence length {input_ids.shape[-1]} exceeds "
                f"max_position_embeddings {self.max_position_embeddings}"
            )
        
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize cache list
        caches = [] if use_cache else None
        
        # Apply transformer blocks
        for layer in self.layers:
            hidden_states, layer_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,  # No past context in this shell
                use_cache=use_cache,
            )
            if use_cache:
                caches.append(layer_cache)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits, caches
