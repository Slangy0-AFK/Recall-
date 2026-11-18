"""
Tests for split-half RoPE and absolute-position KV-cache adapters.

Phase 2: Verify split-half RoPE implementation
Phase 3: Verify absolute-position KV-cache handling
"""

import pytest
import torch
import math
from adapters.rope_cache_adapter import SplitHalfRoPE, AbsolutePositionKVCache


# ============================================================================
# PHASE 2: SPLIT-HALF ROPE TESTS
# ============================================================================

class TestSplitHalfRoPE:
    """Tests for split-half RoPE adapter."""
    
    def test_instantiation(self):
        """Test RoPE instantiation with valid parameters."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        assert rope.head_dim == 64
        assert rope.half_dim == 32
        assert rope.rope_theta == 100000.0
    
    def test_invalid_head_dim_odd(self):
        """Reject odd head dimensions."""
        with pytest.raises(ValueError, match="head_dim must be even"):
            SplitHalfRoPE(head_dim=63)
    
    def test_invalid_head_dim_zero(self):
        """Reject non-positive head dimensions."""
        with pytest.raises(ValueError, match="head_dim must be even"):
            SplitHalfRoPE(head_dim=0)
    
    def test_invalid_rope_theta(self):
        """Reject non-positive rope_theta."""
        with pytest.raises(ValueError, match="rope_theta must be positive"):
            SplitHalfRoPE(rope_theta=-1.0)
    
    def test_forward_shape_fp32(self):
        """Verify output shapes with FP32."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        # GQA-compatible layout: [batch, num_heads, seq, head_dim]
        batch, num_heads, seq, head_dim = 2, 15, 4, 64
        q = torch.randn(batch, num_heads, seq, head_dim, dtype=torch.float32)
        k = torch.randn(batch, num_heads, seq, head_dim, dtype=torch.float32)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert q_rot.dtype == q.dtype
        assert k_rot.dtype == k.dtype
    
    def test_forward_shape_different_seq_lengths(self):
        """Verify handling of different query and key sequence lengths."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        # GQA layout: [batch, num_heads, seq, head_dim]
        q = torch.randn(2, 15, 4, 64, dtype=torch.float32)
        k = torch.randn(2, 5, 8, 64, dtype=torch.float32)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_forward_positions_none_defaults(self):
        """Verify default position generation."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        q = torch.randn(2, 15, 4, 64, dtype=torch.float32)
        k = torch.randn(2, 5, 8, 64, dtype=torch.float32)
        
        # Should not raise, positions default to [0, 1, 2, ...]
        q_rot, k_rot = rope(q, k, positions_q=None, positions_k=None)
        
        assert q_rot.shape == (2, 15, 4, 64)
        assert k_rot.shape == (2, 5, 8, 64)
    
    def test_forward_explicit_positions(self):
        """Verify explicit position specification."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        q = torch.randn(2, 15, 4, 64, dtype=torch.float32)
        k = torch.randn(2, 5, 8, 64, dtype=torch.float32)
        
        positions_q = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
        positions_k = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.float32)
        
        q_rot, k_rot = rope(q, k, positions_q, positions_k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_forward_nonzero_positions(self):
        """Test with nonzero initial positions."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        q = torch.randn(1, 15, 4, 64, dtype=torch.float32)
        k = torch.randn(1, 5, 8, 64, dtype=torch.float32)
        
        # Offset positions: cache started at token 100
        positions_q = torch.arange(100, 104, dtype=torch.float32)
        positions_k = torch.arange(96, 104, dtype=torch.float32)
        
        q_rot, k_rot = rope(q, k, positions_q, positions_k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_finite_outputs(self):
        """Verify all outputs are finite."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        q = torch.randn(2, 15, 16, 64, dtype=torch.float32)
        k = torch.randn(2, 5, 32, 64, dtype=torch.float32)
        
        q_rot, k_rot = rope(q, k)
        
        assert torch.isfinite(q_rot).all(), "Query output has non-finite values"
        assert torch.isfinite(k_rot).all(), "Key output has non-finite values"
    
    def test_finite_gradients(self):
        """Verify finite gradients for backprop."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        q = torch.randn(2, 15, 4, 64, dtype=torch.float32, requires_grad=True)
        k = torch.randn(2, 5, 8, 64, dtype=torch.float32, requires_grad=True)
        
        q_rot, k_rot = rope(q, k)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()
        
        assert torch.isfinite(q.grad).all(), "Query gradient has non-finite values"
        assert torch.isfinite(k.grad).all(), "Key gradient has non-finite values"
    
    def test_deterministic_behavior(self):
        """Verify deterministic/reproducible outputs."""
        torch.manual_seed(42)
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        q = torch.randn(2, 15, 4, 64, dtype=torch.float32)
        k = torch.randn(2, 5, 8, 64, dtype=torch.float32)
        
        q_rot1, k_rot1 = rope(q.clone(), k.clone())
        q_rot2, k_rot2 = rope(q.clone(), k.clone())
        
        assert torch.allclose(q_rot1, q_rot2, atol=1e-6)
        assert torch.allclose(k_rot1, k_rot2, atol=1e-6)
    
    def test_position_zero(self):
        """Test encoding at position zero."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        q = torch.randn(1, 15, 1, 64, dtype=torch.float32)
        k = torch.randn(1, 5, 1, 64, dtype=torch.float32)
        
        positions_q = torch.tensor([0.0], dtype=torch.float32)
        positions_k = torch.tensor([0.0], dtype=torch.float32)
        
        q_rot, k_rot = rope(q, k, positions_q, positions_k)
        
        # At position 0, angles should be 0, so cos(0)=1, sin(0)=0
        # Rotation by 0 should approximately preserve the original tensor
        assert torch.isfinite(q_rot).all()
        assert torch.isfinite(k_rot).all()
    
    def test_discontiguous_positions(self):
        """Test with non-sequential positions."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        q = torch.randn(1, 15, 3, 64, dtype=torch.float32)
        k = torch.randn(1, 5, 5, 64, dtype=torch.float32)
        
        # Discontiguous positions: skip some tokens
        positions_q = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float32)
        positions_k = torch.tensor([0.0, 1.0, 2.0, 5.0, 7.0], dtype=torch.float32)
        
        q_rot, k_rot = rope(q, k, positions_q, positions_k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_mismatched_head_dim_query(self):
        """Reject query with wrong head dimension."""
        rope = SplitHalfRoPE(head_dim=64)
        
        q = torch.randn(2, 15, 4, 32, dtype=torch.float32)  # Wrong head_dim
        k = torch.randn(2, 5, 8, 64, dtype=torch.float32)
        
        with pytest.raises(ValueError, match="Query last dimension must be"):
            rope(q, k)
    
    def test_mismatched_head_dim_key(self):
        """Reject key with wrong head dimension."""
        rope = SplitHalfRoPE(head_dim=64)
        
        q = torch.randn(2, 15, 4, 64, dtype=torch.float32)
        k = torch.randn(2, 5, 8, 32, dtype=torch.float32)  # Wrong head_dim
        
        with pytest.raises(ValueError, match="Key last dimension must be"):
            rope(q, k)
    
    def test_mismatched_dtype(self):
        """Reject mismatched dtypes between Q and K."""
        rope = SplitHalfRoPE(head_dim=64)
        
        q = torch.randn(2, 15, 4, 64, dtype=torch.float32)
        k = torch.randn(2, 5, 8, 64, dtype=torch.float64)
        
        with pytest.raises(ValueError, match="Query and key must have same dtype"):
            rope(q, k)
    
    def test_non_floating_query(self):
        """Reject integer query tensor."""
        rope = SplitHalfRoPE(head_dim=64)
        
        q = torch.randint(0, 10, (2, 15, 4, 64))
        k = torch.randn(2, 5, 8, 64, dtype=torch.float32)
        
        with pytest.raises(TypeError, match="Query must have floating-point"):
            rope(q, k)
    
    def test_mismatched_positions_length(self):
        """Reject positions with wrong length."""
        rope = SplitHalfRoPE(head_dim=64)
        
        q = torch.randn(2, 15, 4, 64, dtype=torch.float32)
        k = torch.randn(2, 5, 8, 64, dtype=torch.float32)
        
        positions_q = torch.arange(5, dtype=torch.float32)  # Wrong length
        
        with pytest.raises(ValueError, match="positions_q length"):
            rope(q, k, positions_q=positions_q)
    
    def test_multiple_batch_sizes(self):
        """Test various batch sizes."""
        rope = SplitHalfRoPE(head_dim=64)
        
        for batch in [1, 2, 4, 8]:
            q = torch.randn(batch, 15, 4, 64, dtype=torch.float32)
            k = torch.randn(batch, 5, 8, 64, dtype=torch.float32)
            
            q_rot, k_rot = rope(q, k)
            
            assert q_rot.shape[0] == batch
            assert k_rot.shape[0] == batch
    
    def test_multiple_sequence_lengths(self):
        """Test various sequence lengths."""
        rope = SplitHalfRoPE(head_dim=64)
        
        for seq_len in [1, 4, 16, 64]:
            q = torch.randn(2, 15, seq_len, 64, dtype=torch.float32)
            k = torch.randn(2, 5, seq_len, 64, dtype=torch.float32)
            
            q_rot, k_rot = rope(q, k)
            
            assert q_rot.shape[-2] == seq_len
            assert k_rot.shape[-2] == seq_len
    
    def test_dtype_fp16(self):
        """Test FP16 precision."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        q = torch.randn(2, 15, 4, 64, dtype=torch.float16)
        k = torch.randn(2, 5, 8, 64, dtype=torch.float16)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.dtype == torch.float16
        assert k_rot.dtype == torch.float16
    
    def test_dtype_fp64(self):
        """Test FP64 precision."""
        rope = SplitHalfRoPE(head_dim=64, rope_theta=100000.0)
        
        q = torch.randn(2, 15, 4, 64, dtype=torch.float64)
        k = torch.randn(2, 5, 8, 64, dtype=torch.float64)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.dtype == torch.float64
        assert k_rot.dtype == torch.float64


# ============================================================================
# PHASE 3: ABSOLUTE-POSITION KV-CACHE TESTS
# ============================================================================

class TestAbsolutePositionKVCache:
    """Tests for absolute-position KV-cache wrapper."""
    
    def test_instantiation(self):
        """Test cache instantiation."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        assert cache.num_kv_heads == 5
        assert cache.head_dim == 64
        assert cache.k_cache is None
        assert cache.v_cache is None
    
    def test_invalid_num_kv_heads(self):
        """Reject invalid num_kv_heads."""
        with pytest.raises(ValueError, match="num_kv_heads must be"):
            AbsolutePositionKVCache(num_kv_heads=0)
    
    def test_invalid_head_dim(self):
        """Reject invalid head_dim."""
        with pytest.raises(ValueError, match="head_dim must be"):
            AbsolutePositionKVCache(num_kv_heads=5, head_dim=0)
    
    def test_sequential_append(self):
        """Test sequential append with auto-incrementing positions."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        # First append: tokens at positions 0-3
        k1 = torch.randn(2, 5, 4, 64, dtype=torch.float32)
        v1 = torch.randn(2, 5, 4, 64, dtype=torch.float32)
        cache.append(k1, v1)
        
        k_cached, v_cached, positions = cache.get_all()
        assert k_cached.shape == (2, 5, 4, 64)
        assert v_cached.shape == (2, 5, 4, 64)
        assert positions.tolist() == [0, 1, 2, 3]
        
        # Second append: tokens at positions 4-7
        k2 = torch.randn(2, 5, 4, 64, dtype=torch.float32)
        v2 = torch.randn(2, 5, 4, 64, dtype=torch.float32)
        cache.append(k2, v2)
        
        k_cached, v_cached, positions = cache.get_all()
        assert k_cached.shape == (2, 5, 8, 64)
        assert v_cached.shape == (2, 5, 8, 64)
        assert positions.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    
    def test_nonzero_initial_position(self):
        """Test starting with non-zero initial position."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k = torch.randn(1, 5, 4, 64, dtype=torch.float32)
        v = torch.randn(1, 5, 4, 64, dtype=torch.float32)
        
        # Explicitly specify positions starting at 100
        positions = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        cache.append(k, v, positions)
        
        _, _, cached_positions = cache.get_all()
        assert cached_positions.tolist() == [100, 101, 102, 103]
    
    def test_discontiguous_positions(self):
        """Test appending discontiguous positions."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k1 = torch.randn(1, 5, 2, 64, dtype=torch.float32)
        v1 = torch.randn(1, 5, 2, 64, dtype=torch.float32)
        positions1 = torch.tensor([0, 2], dtype=torch.int64)
        cache.append(k1, v1, positions1)
        
        k2 = torch.randn(1, 5, 2, 64, dtype=torch.float32)
        v2 = torch.randn(1, 5, 2, 64, dtype=torch.float32)
        positions2 = torch.tensor([1, 3], dtype=torch.int64)
        cache.append(k2, v2, positions2)
        
        _, _, cached_positions = cache.get_all()
        assert set(cached_positions.tolist()) == {0, 1, 2, 3}
    
    def test_lookup_contiguous(self):
        """Test looking up contiguous positions."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k_orig = torch.randn(2, 5, 4, 64, dtype=torch.float32)
        v_orig = torch.randn(2, 5, 4, 64, dtype=torch.float32)
        cache.append(k_orig, v_orig)
        
        # Lookup positions 0, 1, 2
        lookup_positions = torch.tensor([0, 1, 2], dtype=torch.int64)
        k_retrieved, v_retrieved = cache.lookup(lookup_positions)
        
        assert k_retrieved.shape == (2, 5, 3, 64)
        assert v_retrieved.shape == (2, 5, 3, 64)
        # Check values match originals at those positions
        assert torch.allclose(k_retrieved, k_orig[:, :, :3, :])
        assert torch.allclose(v_retrieved, v_orig[:, :, :3, :])
    
    def test_lookup_discontiguous(self):
        """Test looking up discontiguous positions."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k_orig = torch.randn(1, 5, 5, 64, dtype=torch.float32)
        v_orig = torch.randn(1, 5, 5, 64, dtype=torch.float32)
        cache.append(k_orig, v_orig)
        
        # Lookup in different order: [4, 0, 2]
        lookup_positions = torch.tensor([4, 0, 2], dtype=torch.int64)
        k_retrieved, v_retrieved = cache.lookup(lookup_positions)
        
        assert k_retrieved.shape == (1, 5, 3, 64)
        # Verify correct retrieval order
        assert torch.allclose(k_retrieved[0, :, 0, :], k_orig[0, :, 4, :])
        assert torch.allclose(k_retrieved[0, :, 1, :], k_orig[0, :, 0, :])
        assert torch.allclose(k_retrieved[0, :, 2, :], k_orig[0, :, 2, :])
    
    def test_lookup_missing_position(self):
        """Test that lookup raises for missing positions."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k = torch.randn(1, 5, 2, 64, dtype=torch.float32)
        v = torch.randn(1, 5, 2, 64, dtype=torch.float32)
        positions = torch.tensor([0, 1], dtype=torch.int64)
        cache.append(k, v, positions)
        
        # Try to lookup non-existent position
        with pytest.raises(ValueError, match="Position .* not found in cache"):
            cache.lookup(torch.tensor([5], dtype=torch.int64))
    
    def test_lookup_empty_cache(self):
        """Test that lookup on empty cache raises."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        with pytest.raises(RuntimeError, match="Cache is empty"):
            cache.lookup(torch.tensor([0], dtype=torch.int64))
    
    def test_duplicate_position_error(self):
        """Test that duplicate positions are rejected."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k1 = torch.randn(1, 5, 2, 64, dtype=torch.float32)
        v1 = torch.randn(1, 5, 2, 64, dtype=torch.float32)
        cache.append(k1, v1)  # Positions [0, 1]
        
        # Try to append with duplicate position
        k2 = torch.randn(1, 5, 1, 64, dtype=torch.float32)
        v2 = torch.randn(1, 5, 1, 64, dtype=torch.float32)
        positions2 = torch.tensor([1], dtype=torch.int64)  # Position 1 already exists
        
        with pytest.raises(ValueError, match="Duplicate positions"):
            cache.append(k2, v2, positions2)
    
    def test_mismatched_batch_size(self):
        """Test that mismatched batch size is rejected."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k1 = torch.randn(2, 5, 2, 64, dtype=torch.float32)
        v1 = torch.randn(2, 5, 2, 64, dtype=torch.float32)
        cache.append(k1, v1)
        
        # Try to append with different batch size
        k2 = torch.randn(3, 5, 2, 64, dtype=torch.float32)
        v2 = torch.randn(3, 5, 2, 64, dtype=torch.float32)
        
        with pytest.raises(ValueError, match="Batch size .* != cached batch"):
            cache.append(k2, v2)
    
    def test_mismatched_head_count(self):
        """Test that mismatched head count is rejected."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k = torch.randn(2, 3, 2, 64, dtype=torch.float32)  # 3 heads instead of 5
        v = torch.randn(2, 3, 2, 64, dtype=torch.float32)
        
        with pytest.raises(ValueError, match="heads .* != "):
            cache.append(k, v)
    
    def test_mismatched_head_dim(self):
        """Test that mismatched head_dim is rejected."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k = torch.randn(2, 5, 2, 32, dtype=torch.float32)  # 32 instead of 64
        v = torch.randn(2, 5, 2, 32, dtype=torch.float32)
        
        with pytest.raises(ValueError, match="head_dim .* != "):
            cache.append(k, v)
    
    def test_mismatched_k_v_shapes(self):
        """Test that mismatched K/V shapes are rejected."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k = torch.randn(2, 5, 2, 64, dtype=torch.float32)
        v = torch.randn(2, 5, 3, 64, dtype=torch.float32)  # Different seq length
        
        with pytest.raises(ValueError, match="v_new shape .* != k_new shape"):
            cache.append(k, v)
    
    def test_mismatched_dtype(self):
        """Test that mismatched K/V dtypes are rejected."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k = torch.randn(2, 5, 2, 64, dtype=torch.float32)
        v = torch.randn(2, 5, 2, 64, dtype=torch.float64)
        
        with pytest.raises(ValueError, match="k_new and v_new must have same dtype"):
            cache.append(k, v)
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k = torch.randn(2, 5, 4, 64, dtype=torch.float32)
        v = torch.randn(2, 5, 4, 64, dtype=torch.float32)
        cache.append(k, v)
        
        assert cache.k_cache is not None
        
        cache.clear()
        
        assert cache.k_cache is None
        assert cache.v_cache is None
        assert cache.cache_positions is None
    
    def test_shape_validation_query_rank(self):
        """Test that invalid query rank is rejected."""
        cache = AbsolutePositionKVCache(num_kv_heads=5, head_dim=64)
        
        k = torch.randn(2, 5, 4, 64, dtype=torch.float32)
        v = torch.randn(2, 5, 4, 64, dtype=torch.float32)
        cache.append(k, v)
        
        # Try lookup with wrong positions rank
        with pytest.raises(ValueError, match="positions must have rank 1"):
            cache.lookup(torch.tensor([[0, 1]], dtype=torch.int64))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
