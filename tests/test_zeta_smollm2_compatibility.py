from __future__ import annotations

import unittest

import torch
import zeta_scaling_toolkit as zeta


class PinnedZetaCompatibilityTests(unittest.TestCase):
    def test_attention_semantics_differ_from_llama_scaled_dot_product(self) -> None:
        query = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
        key = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]], dtype=torch.float32)
        value = torch.tensor([[[[0.0], [1.0]]]], dtype=torch.float32)
        attention = zeta.NonCollapsingAttention(d_k=2, check_finite=True).eval()

        with torch.inference_mode():
            zeta_output = attention(query, key, value)
            llama_output = (
                torch.softmax(query @ key.transpose(-2, -1) / (2**0.5), dim=-1)
                @ value
            )

        self.assertTrue(torch.isfinite(zeta_output).all())
        self.assertFalse(torch.allclose(zeta_output, llama_output, atol=1e-4))
        self.assertGreater(float((zeta_output - llama_output).abs().max()), 0.3)

    def test_gqa_requires_expanding_five_kv_heads_to_fifteen_query_heads(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(20261117)
        query = torch.randn((1, 15, 1, 64), generator=generator)
        key = torch.randn((1, 5, 1, 64), generator=generator)
        value = torch.randn((1, 5, 1, 64), generator=generator)
        attention = zeta.NonCollapsingAttention(d_k=64, check_finite=True).eval()

        with self.assertRaisesRegex(ValueError, "batch dimensions must be broadcastable"):
            attention(query, key, value)

        with torch.inference_mode():
            output = attention(
                query,
                key.repeat_interleave(3, dim=1),
                value.repeat_interleave(3, dim=1),
            )

        self.assertEqual(tuple(output.shape), (1, 15, 1, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_zeta_interleaved_rope_differs_from_llama_split_half_rope(self) -> None:
        dimension = 64
        positions = torch.tensor([[[1]]], dtype=torch.int64)
        values = torch.arange(dimension, dtype=torch.float64).reshape(1, 1, 1, -1) / 13
        base = 100000.0
        frequencies = torch.exp(
            -torch.log(torch.tensor(base, dtype=torch.float64))
            * torch.arange(0, dimension, 2, dtype=torch.float64)
            / dimension
        )
        angles = positions.to(torch.float64)[..., None] * frequencies
        cosine, sine = angles.cos(), angles.sin()
        first, second = values[..., : dimension // 2], values[..., dimension // 2 :]
        llama_output = torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine),
            dim=-1,
        )

        zeta_output = zeta.rotary_embedding(values, positions, base=base)

        self.assertEqual(tuple(zeta_output.shape), tuple(values.shape))
        self.assertTrue(torch.isfinite(zeta_output).all())
        self.assertFalse(torch.allclose(zeta_output, llama_output, atol=1e-10))

    def test_bounded_cache_preserves_sinks_and_recent_absolute_positions(self) -> None:
        cache = zeta.StreamingKVCache(window=64, sink_tokens=64)
        for position in range(130):
            token = torch.full((1, 5, 1, 64), float(position), dtype=torch.float32)
            keys, values, positions = cache.append(token, token)

        expected_positions = torch.tensor(
            [*range(64), *range(66, 130)], dtype=torch.int64
        )
        self.assertTrue(torch.equal(positions, expected_positions))
        self.assertEqual(tuple(keys.shape), (1, 5, 128, 64))
        self.assertEqual(tuple(values.shape), (1, 5, 128, 64))
        self.assertEqual(cache.next_position, 130)
        self.assertTrue(torch.isfinite(keys).all())
        self.assertTrue(torch.isfinite(values).all())

        next_token = torch.full((1, 5, 1, 64), 130.0, dtype=torch.float32)
        _, _, next_positions = cache.append(next_token, next_token)
        self.assertEqual(cache.next_position, 131)
        self.assertEqual(int(next_positions[-1]), 130)


if __name__ == "__main__":
    unittest.main()
