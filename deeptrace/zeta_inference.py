"""Native Zeta Scaling Toolkit inference engine for retrieval tests."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

from zeta_scaling_toolkit import (
    NonCollapsingAttention,
    StreamingKVCache,
    rotary_embedding,
)


class ZetaTransformerBlock(nn.Module):
    """Single transformer block using Zeta's attention mechanism."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        tau0: float = 10.0,
        tau_max: float = 50.0,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        # Zeta's attention
        self.attention = NonCollapsingAttention(
            d_k=self.d_k,
            tau0=tau0,
            tau_max=tau_max,
            check_finite=True,
        )

        # Projection layers for multi-head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Zeta attention and RoPE.

        Args:
            x: (batch, seq_len, d_model)
            positions: (seq_len,) int64 absolute positions
            attn_mask: (seq_len, seq_len) optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)

        # Project to Q, K, V
        q = self.W_q(x_norm)  # (batch, seq, d_model)
        k = self.W_k(x_norm)
        v = self.W_v(x_norm)

        # Reshape to multi-head
        batch, seq, _ = q.shape
        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        # q, k, v: (batch, num_heads, seq, d_k)

        # Apply RoPE if enabled
        if self.use_rope:
            # Broadcast positions to heads: (seq,) → (1, 1, seq)
            pos_for_rope = positions.unsqueeze(0).unsqueeze(0)
            q = rotary_embedding(q, pos_for_rope)
            k = rotary_embedding(k, pos_for_rope)

        # Zeta attention (operates on (batch, num_heads, seq, d_k))
        attn_output = self.attention(q, k, v, attn_mask=attn_mask)
        # attn_output: (batch, num_heads, seq, d_k)

        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq, self.d_model)

        # Output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        x = x + output

        # Feed-forward with residual
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        ff_output = self.dropout(ff_output)
        output = x + ff_output

        return output


class ZetaLanguageModel(nn.Module):
    """Minimal LLM using Zeta's native attention for retrieval testing."""

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 8192,
        tau0: float = 10.0,
        tau_max: float = 50.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embedding (without positional encoding, using RoPE instead)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                ZetaTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    tau0=tau0,
                    tau_max=tau_max,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Zeta model.

        Args:
            input_ids: (batch, seq_len)
            positions: (seq_len,) int64 absolute positions; auto-generated if None
            attn_mask: (seq_len, seq_len) optional attention mask

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        seq_len = input_ids.shape[1]

        # Generate positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, dtype=torch.int64, device=input_ids.device)

        # Embed
        x = self.embedding(input_ids)  # (batch, seq, d_model)

        # Verify embeddings are finite
        if not torch.isfinite(x).all():
            raise ValueError("Embedding contains NaN or inf")

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, positions, attn_mask=attn_mask)

            # Check for numerical issues
            if not torch.isfinite(x).all():
                raise ValueError("Layer output contains NaN or inf")

        # Final layer norm
        x = self.final_norm(x)

        # LM head
        logits = self.lm_head(x)  # (batch, seq, vocab_size)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Autoregressive generation with optional KV caching.

        Args:
            input_ids: (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Softmax temperature
            top_k: Top-k sampling (None for argmax)
            use_cache: Whether to use StreamingKVCache

        Returns:
            generated_ids: (batch, seq_len + max_new_tokens)
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Initialize cache if using
        if use_cache:
            cache = StreamingKVCache(window=1024, sink_tokens=4)
        else:
            cache = None

        generated = input_ids.clone()
        current_pos = input_ids.shape[1]

        for _ in range(max_new_tokens):
            # Forward pass on full sequence (or just new token if caching)
            if cache is None:
                # Full context each time (slower but simpler)
                logits = self.forward(generated)  # (batch, seq, vocab)
                next_logits = logits[:, -1, :] / temperature  # (batch, vocab)
            else:
                # Use only last token (would need full cache refactor for real streaming)
                # For now, still do full pass for correctness
                positions = torch.arange(
                    generated.shape[1], dtype=torch.int64, device=device
                )
                logits = self.forward(generated, positions=positions)
                next_logits = logits[:, -1, :] / temperature

            # Sampling
            if top_k is not None:
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
                probs = torch.softmax(top_k_logits, dim=-1)
                sampled = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, sampled)
            else:
                # Greedy (argmax)
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            current_pos += 1

            # Check for EOF or stop token (token 2 is typical </s>)
            if (next_token == 2).all():
                break

        return generated
