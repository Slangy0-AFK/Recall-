"""
Comprehensive tests for SmolLM2 architecture components.

Phase 2: Validate RMSNorm, SwiGLU, residual connections, transformer blocks, and model shell.
Tests include:
- Mathematical correctness against reference implementations
- Tensor shape validation
- Gradient flow
- Numerical stability
- Multiple data types (FP32, FP16, FP64)
- Batch sizes 1-8
- Sequence lengths 1-64
- Deterministic behavior
- Edge cases and error handling
"""

import pytest
import torch
import torch.nn as nn
import math
from adapters.smollm2_architecture import (
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    SmolLM2ModelShell,
)


# ============================================================================
# RMSNORM TESTS
# ============================================================================

class TestRMSNorm:
    """Test RMSNorm implementation."""
    
    def test_instantiation(self):
        """Test RMSNorm instantiation."""
        norm = RMSNorm(dim=960)
        assert norm.dim == 960
        assert norm.eps == 1e-5
        assert norm.weight.shape == (960,)
        assert torch.allclose(norm.weight, torch.ones(960))
    
    def test_instantiation_custom_eps(self):
        """Test RMSNorm with custom epsilon."""
        norm = RMSNorm(dim=960, eps=1e-6)
        assert norm.eps == 1e-6
    
    def test_invalid_dim_zero(self):
        """Reject zero or negative dimensions."""
        with pytest.raises(ValueError, match="dim must be positive"):
            RMSNorm(dim=0)
        with pytest.raises(ValueError, match="dim must be positive"):
            RMSNorm(dim=-1)
    
    def test_invalid_eps_zero(self):
        """Reject zero or negative epsilon."""
        with pytest.raises(ValueError, match="eps must be positive"):
            RMSNorm(dim=960, eps=0)
        with pytest.raises(ValueError, match="eps must be positive"):
            RMSNorm(dim=960, eps=-1e-5)
    
    def test_shape_preservation_fp32(self):
        """Verify output shape matches input shape (FP32)."""
        norm = RMSNorm(dim=960)
        
        for batch_size in [1, 2, 4, 8]:
            for seq_len in [1, 4, 16, 64]:
                x = torch.randn(batch_size, seq_len, 960, dtype=torch.float32)
                y = norm(x)
                assert y.shape == x.shape
                assert y.dtype == torch.float32
    
    def test_shape_preservation_fp16(self):
        """Verify output shape matches input shape (FP16)."""
        norm = RMSNorm(dim=960).to(dtype=torch.float16)
        
        for batch_size in [1, 2, 4]:
            for seq_len in [1, 4, 16]:
                x = torch.randn(batch_size, seq_len, 960, dtype=torch.float16)
                y = norm(x)
                assert y.shape == x.shape
                assert y.dtype == torch.float16
    
    def test_shape_preservation_fp64(self):
        """Verify output shape matches input shape (FP64)."""
        norm = RMSNorm(dim=960)
        
        for batch_size in [1, 2]:
            for seq_len in [1, 4]:
                x = torch.randn(batch_size, seq_len, 960, dtype=torch.float64)
                y = norm(x)
                assert y.shape == x.shape
                assert y.dtype == torch.float64
    
    def test_dimension_mismatch(self):
        """Reject inputs with wrong last dimension."""
        norm = RMSNorm(dim=960)
        x = torch.randn(2, 4, 512)  # Wrong last dimension
        
        with pytest.raises(ValueError, match="does not match expected dimension"):
            norm(x)
    
    def test_mathematical_correctness_fp32(self):
        """Verify RMSNorm formula: x / sqrt(mean(x^2) + eps) * weight."""
        dim = 64
        norm = RMSNorm(dim=dim, eps=1e-5)
        
        # Simple input for verification
        x = torch.randn(2, 8, dim, dtype=torch.float32)
        y = norm(x)
        
        # Manual computation
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-5)
        expected = (x / rms) * norm.weight
        
        assert torch.allclose(y, expected, atol=1e-6)
    
    def test_identity_with_unit_weight(self):
        """With weight=1 and zero input, output should be near zero."""
        norm = RMSNorm(dim=64, eps=1e-5)
        norm.weight.data.fill_(1.0)
        
        x = torch.zeros(2, 4, 64, dtype=torch.float32)
        y = norm(x)
        
        # With zero input and eps, output should still be small
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-5)
    
    def test_finite_outputs(self):
        """Verify all outputs are finite."""
        norm = RMSNorm(dim=960)
        
        for _ in range(10):
            x = torch.randn(4, 16, 960)
            y = norm(x)
            assert torch.isfinite(y).all()
    
    def test_finite_gradients(self):
        """Verify gradients flow and are finite."""
        norm = RMSNorm(dim=960)
        
        x = torch.randn(4, 16, 960, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert norm.weight.grad is not None
        assert torch.isfinite(norm.weight.grad).all()
    
    def test_deterministic_behavior(self):
        """Verify deterministic output."""
        norm = RMSNorm(dim=960)
        x = torch.randn(4, 16, 960)
        
        y1 = norm(x)
        y2 = norm(x)
        
        assert torch.equal(y1, y2)
    
    def test_no_input_mutation(self):
        """Verify input tensor is not modified."""
        norm = RMSNorm(dim=960)
        x = torch.randn(4, 16, 960)
        x_original = x.clone()
        
        _ = norm(x)
        
        assert torch.equal(x, x_original)


# ============================================================================
# SWIGLU TESTS
# ============================================================================

class TestSwiGLU:
    """Test SwiGLU implementation."""
    
    def test_instantiation_default(self):
        """Test SwiGLU instantiation with default activation."""
        swiglu = SwiGLU(in_features=960, out_features=960, intermediate_features=2560)
        assert swiglu.in_features == 960
        assert swiglu.out_features == 960
        assert swiglu.intermediate_features == 2560
        assert isinstance(swiglu.activation, nn.SiLU)
    
    def test_instantiation_custom_activation(self):
        """Test SwiGLU with custom activation."""
        activation = nn.GELU()
        swiglu = SwiGLU(
            in_features=960,
            out_features=960,
            intermediate_features=2560,
            activation=activation,
        )
        assert swiglu.activation is activation
    
    def test_instantiation_no_bias(self):
        """Verify bias=False as per SmolLM2."""
        swiglu = SwiGLU(
            in_features=960,
            out_features=960,
            intermediate_features=2560,
            bias=False,
        )
        assert swiglu.gate_proj.bias is None
        assert swiglu.up_proj.bias is None
        assert swiglu.down_proj.bias is None
    
    def test_instantiation_with_bias(self):
        """Verify bias=True when requested."""
        swiglu = SwiGLU(
            in_features=960,
            out_features=960,
            intermediate_features=2560,
            bias=True,
        )
        assert swiglu.gate_proj.bias is not None
        assert swiglu.up_proj.bias is not None
        assert swiglu.down_proj.bias is not None
    
    def test_invalid_dimensions_zero(self):
        """Reject zero or negative dimensions."""
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            SwiGLU(in_features=0, out_features=960, intermediate_features=2560)
        
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            SwiGLU(in_features=960, out_features=0, intermediate_features=2560)
        
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            SwiGLU(in_features=960, out_features=960, intermediate_features=0)
    
    def test_shape_preservation(self):
        """Verify output shape is [batch, seq, out_features]."""
        swiglu = SwiGLU(in_features=960, out_features=960, intermediate_features=2560)
        
        for batch_size in [1, 2, 4, 8]:
            for seq_len in [1, 4, 16, 64]:
                x = torch.randn(batch_size, seq_len, 960, dtype=torch.float32)
                y = swiglu(x)
                assert y.shape == (batch_size, seq_len, 960)
                assert y.dtype == torch.float32
    
    def test_dimension_mismatch(self):
        """Reject inputs with wrong last dimension."""
        swiglu = SwiGLU(in_features=960, out_features=960, intermediate_features=2560)
        x = torch.randn(2, 4, 512)  # Wrong in_features
        
        with pytest.raises(ValueError, match="does not match expected dimension"):
            swiglu(x)
    
    def test_mathematical_structure(self):
        """Verify SwiGLU structure: (gate * up) @ W_down."""
        swiglu = SwiGLU(in_features=64, out_features=64, intermediate_features=128, bias=False)
        
        x = torch.randn(2, 4, 64)
        y = swiglu(x)
        
        # Manual computation
        gate = torch.nn.functional.silu(swiglu.gate_proj(x))
        up = swiglu.up_proj(x)
        combined = gate * up
        expected = swiglu.down_proj(combined)
        
        assert torch.allclose(y, expected, atol=1e-6)
    
    def test_finite_outputs(self):
        """Verify all outputs are finite."""
        swiglu = SwiGLU(in_features=960, out_features=960, intermediate_features=2560)
        
        for _ in range(10):
            x = torch.randn(4, 16, 960)
            y = swiglu(x)
            assert torch.isfinite(y).all()
    
    def test_finite_gradients(self):
        """Verify gradients flow and are finite."""
        swiglu = SwiGLU(in_features=960, out_features=960, intermediate_features=2560)
        
        x = torch.randn(4, 16, 960, requires_grad=True)
        y = swiglu(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_deterministic_behavior(self):
        """Verify deterministic output."""
        swiglu = SwiGLU(in_features=960, out_features=960, intermediate_features=2560)
        x = torch.randn(4, 16, 960)
        
        y1 = swiglu(x)
        y2 = swiglu(x)
        
        assert torch.equal(y1, y2)
    
    def test_no_input_mutation(self):
        """Verify input tensor is not modified."""
        swiglu = SwiGLU(in_features=960, out_features=960, intermediate_features=2560)
        x = torch.randn(4, 16, 960)
        x_original = x.clone()
        
        _ = swiglu(x)
        
        assert torch.equal(x, x_original)


# ============================================================================
# TRANSFORMER BLOCK TESTS
# ============================================================================

class TestTransformerBlock:
    """Test TransformerBlock implementation."""
    
    def test_instantiation_default_smollm2(self):
        """Test instantiation with SmolLM2 defaults."""
        block = TransformerBlock()
        assert block.hidden_size == 960
        assert block.num_attention_heads == 15
        assert block.num_key_value_heads == 5
        assert block.head_dim == 64
    
    def test_dimension_validation(self):
        """Verify hidden_size == num_heads * head_dim."""
        # Valid configuration
        block = TransformerBlock(hidden_size=960, num_attention_heads=15, head_dim=64)
        assert block.hidden_size == 960
        
        # Invalid configuration
        with pytest.raises(ValueError, match="hidden_size"):
            TransformerBlock(hidden_size=960, num_attention_heads=16, head_dim=64)
    
    def test_forward_shape_fp32(self):
        """Verify forward pass output shape (FP32)."""
        block = TransformerBlock()
        
        for batch_size in [1, 2, 4]:
            for seq_len in [1, 4, 16, 64]:
                hidden_states = torch.randn(batch_size, seq_len, 960, dtype=torch.float32)
                output, cache = block(hidden_states, use_cache=False)
                
                assert output.shape == hidden_states.shape
                assert output.dtype == torch.float32
                assert cache is None
    
    def test_forward_shape_fp16(self):
        """Verify forward pass output shape (FP16)."""
        block = TransformerBlock().to(dtype=torch.float16)
        
        for batch_size in [1, 2]:
            for seq_len in [1, 4, 16]:
                hidden_states = torch.randn(batch_size, seq_len, 960, dtype=torch.float16)
                output, cache = block(hidden_states, use_cache=False)
                
                assert output.shape == hidden_states.shape
                assert output.dtype == torch.float16
    
    def test_cache_shape(self):
        """Verify cache shape when use_cache=True."""
        block = TransformerBlock()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, 960)
        output, cache = block(hidden_states, use_cache=True)
        
        assert cache is not None
        k, v = cache
        assert k.shape == (batch_size, 5, seq_len, 64)  # (batch, kv_heads, seq, head_dim)
        assert v.shape == (batch_size, 5, seq_len, 64)
    
    def test_dimension_mismatch(self):
        """Reject inputs with wrong hidden dimension."""
        block = TransformerBlock()
        hidden_states = torch.randn(2, 4, 512)  # Wrong hidden_size
        
        with pytest.raises(ValueError, match="does not match expected"):
            block(hidden_states)
    
    def test_finite_outputs(self):
        """Verify all outputs are finite."""
        block = TransformerBlock()
        
        for _ in range(5):
            hidden_states = torch.randn(4, 16, 960)
            output, _ = block(hidden_states)
            assert torch.isfinite(output).all()
    
    def test_finite_gradients(self):
        """Verify gradients flow and are finite."""
        block = TransformerBlock()
        
        hidden_states = torch.randn(4, 16, 960, requires_grad=True)
        output, _ = block(hidden_states)
        loss = output.sum()
        loss.backward()
        
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
    
    def test_deterministic_behavior(self):
        """Verify deterministic output."""
        block = TransformerBlock()
        hidden_states = torch.randn(2, 8, 960)
        
        output1, _ = block(hidden_states)
        output2, _ = block(hidden_states)
        
        assert torch.equal(output1, output2)
    
    def test_no_input_mutation(self):
        """Verify input tensor is not modified."""
        block = TransformerBlock()
        hidden_states = torch.randn(2, 8, 960)
        hidden_states_original = hidden_states.clone()
        
        _ = block(hidden_states)
        
        assert torch.equal(hidden_states, hidden_states_original)


# ============================================================================
# MODEL SHELL TESTS
# ============================================================================

class TestSmolLM2ModelShell:
    """Test SmolLM2ModelShell implementation."""
    
    def test_instantiation_default_smollm2(self):
        """Test instantiation with SmolLM2 defaults."""
        model = SmolLM2ModelShell()
        assert model.vocab_size == 49152
        assert model.hidden_size == 960
        assert model.num_hidden_layers == 32
        assert model.num_attention_heads == 15
        assert model.num_key_value_heads == 5
        assert len(model.layers) == 32
    
    def test_instantiation_custom_num_layers(self):
        """Test instantiation with custom number of layers."""
        model = SmolLM2ModelShell(num_hidden_layers=8)
        assert len(model.layers) == 8
    
    def test_embedding_shape(self):
        """Verify embedding layer shape."""
        model = SmolLM2ModelShell()
        assert model.embed_tokens.num_embeddings == 49152
        assert model.embed_tokens.embedding_dim == 960
    
    def test_forward_shape_fp32(self):
        """Verify forward pass output shape (FP32)."""
        model = SmolLM2ModelShell(num_hidden_layers=2)  # Smaller for speed
        
        for batch_size in [1, 2]:
            for seq_len in [1, 4, 16]:
                input_ids = torch.randint(0, 49152, (batch_size, seq_len))
                logits, caches = model(input_ids, use_cache=False)
                
                assert logits.shape == (batch_size, seq_len, 49152)
                assert logits.dtype == torch.float32
                assert caches is None
    
    def test_forward_shape_fp16(self):
        """Verify forward pass output shape (FP16)."""
        model = SmolLM2ModelShell(num_hidden_layers=2)
        model = model.to(dtype=torch.float16)
        
        input_ids = torch.randint(0, 49152, (1, 8))
        logits, caches = model(input_ids, use_cache=False)
        
        assert logits.shape == (1, 8, 49152)
        assert logits.dtype == torch.float16
    
    def test_cache_structure(self):
        """Verify cache structure when use_cache=True."""
        model = SmolLM2ModelShell(num_hidden_layers=2)
        
        input_ids = torch.randint(0, 49152, (1, 8))
        logits, caches = model(input_ids, use_cache=True)
        
        assert caches is not None
        assert len(caches) == 2  # One per layer
        
        for cache in caches:
            k, v = cache
            assert k.shape == (1, 5, 8, 64)  # (batch, kv_heads, seq, head_dim)
            assert v.shape == (1, 5, 8, 64)
    
    def test_max_position_validation(self):
        """Reject inputs exceeding max_position_embeddings."""
        model = SmolLM2ModelShell(max_position_embeddings=512, num_hidden_layers=1)
        
        input_ids = torch.randint(0, 49152, (1, 1024))  # Exceeds max
        
        with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
            model(input_ids)
    
    def test_finite_outputs(self):
        """Verify all outputs are finite."""
        model = SmolLM2ModelShell(num_hidden_layers=2)
        
        for _ in range(3):
            input_ids = torch.randint(0, 49152, (2, 8))
            logits, _ = model(input_ids)
            assert torch.isfinite(logits).all()
    
    def test_finite_gradients(self):
        """Verify gradients flow and are finite."""
        model = SmolLM2ModelShell(num_hidden_layers=2)
        
        input_ids = torch.randint(0, 49152, (2, 8), dtype=torch.long)
        logits, _ = model(input_ids)
        loss = logits.sum()
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
                has_grads = True
        
        assert has_grads, "No gradients were computed"
    
    def test_deterministic_behavior(self):
        """Verify deterministic output."""
        model = SmolLM2ModelShell(num_hidden_layers=2)
        input_ids = torch.randint(0, 49152, (1, 8))
        
        logits1, _ = model(input_ids)
        logits2, _ = model(input_ids)
        
        assert torch.equal(logits1, logits2)
    
    def test_no_input_mutation(self):
        """Verify input tensor is not modified."""
        model = SmolLM2ModelShell(num_hidden_layers=2)
        input_ids = torch.randint(0, 49152, (1, 8))
        input_ids_original = input_ids.clone()
        
        _ = model(input_ids)
        
        assert torch.equal(input_ids, input_ids_original)
    
    def test_vocab_size_validation(self):
        """Verify LM head has correct output dimension."""
        model = SmolLM2ModelShell(vocab_size=10000)
        assert model.lm_head.out_features == 10000


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestArchitectureIntegration:
    """Test integration of architecture components."""
    
    def test_phase2_rope_integration(self):
        """Verify SplitHalfRoPE integration in transformer block."""
        block = TransformerBlock(head_dim=64)
        
        # Verify RoPE is instantiated
        assert hasattr(block, 'rope')
        assert block.rope.head_dim == 64
    
    def test_phase2_kv_cache_integration(self):
        """Verify AbsolutePositionKVCache integration."""
        block = TransformerBlock()
        
        # Verify KV cache wrapper is present
        assert hasattr(block, 'kv_cache')
    
    def test_zeta_gqa_integration(self):
        """Verify Zeta GroupedQueryAttention integration."""
        block = TransformerBlock()
        
        # Verify GQA is instantiated with correct parameters
        assert hasattr(block, 'gqa')
        assert block.gqa.num_query_heads == 15
        assert block.gqa.num_key_value_heads == 5
        assert block.gqa.head_dim == 64
    
    def test_full_stack_forward_pass(self):
        """Test complete forward pass through all components."""
        model = SmolLM2ModelShell(num_hidden_layers=3)
        
        input_ids = torch.randint(0, 49152, (1, 16))
        logits, caches = model(input_ids, use_cache=True)
        
        # Verify shapes
        assert logits.shape == (1, 16, 49152)
        assert len(caches) == 3
        
        # Verify values are reasonable
        assert torch.isfinite(logits).all()
        assert (logits >= -100).all() and (logits <= 100).all()  # Reasonable logit range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
