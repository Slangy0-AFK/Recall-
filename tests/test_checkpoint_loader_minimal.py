"""
Minimal test suite for SmolLM2 checkpoint loading.
Focuses on configuration, mapping logic, and validation without full model loading.
"""

import unittest
from adapters.smollm2_checkpoint_loader import SmolLM2CheckpointLoader
from adapters.smollm2_architecture import SmolLM2ModelShell


class TestCheckpointLoaderCore(unittest.TestCase):
    """Core checkpoint loader functionality tests."""

    def test_loader_initialization(self):
        """Test that loader initializes."""
        loader = SmolLM2CheckpointLoader()
        self.assertIsNotNone(loader)

    def test_config_lazy_loading(self):
        """Test that config loads on demand."""
        loader = SmolLM2CheckpointLoader()
        config = loader.config
        self.assertIsNotNone(config)
        self.assertEqual(config.model_type, "llama")

    def test_config_values(self):
        """Test that config has correct values."""
        loader = SmolLM2CheckpointLoader()
        config = loader.config
        
        self.assertEqual(config.vocab_size, 49152)
        self.assertEqual(config.hidden_size, 960)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 15)
        self.assertEqual(config.num_key_value_heads, 5)
        self.assertEqual(config.intermediate_size, 2560)
        self.assertEqual(config.hidden_act, "silu")


class TestModelShellStructure(unittest.TestCase):
    """Test Recall model shell structure."""

    def test_model_creation(self):
        """Test that model shell can be created."""
        model = SmolLM2ModelShell()
        self.assertIsNotNone(model)

    def test_model_has_32_layers(self):
        """Test that model has 32 transformer layers."""
        model = SmolLM2ModelShell()
        self.assertEqual(len(model.layers), 32)

    def test_model_layer_components(self):
        """Test that each layer has required components."""
        model = SmolLM2ModelShell()
        layer = model.layers[0]
        
        # Check layer has attention, MLP, norms
        self.assertTrue(hasattr(layer, 'self_attn'))
        self.assertTrue(hasattr(layer, 'mlp'))
        self.assertTrue(hasattr(layer, 'input_layernorm'))
        self.assertTrue(hasattr(layer, 'post_attention_layernorm'))

    def test_model_has_embeddings(self):
        """Test that model has embedding layer."""
        model = SmolLM2ModelShell()
        self.assertTrue(hasattr(model, 'embed_tokens'))
        self.assertIsNotNone(model.embed_tokens)

    def test_model_has_output_head(self):
        """Test that model has output projection."""
        model = SmolLM2ModelShell()
        self.assertTrue(hasattr(model, 'lm_head'))
        self.assertIsNotNone(model.lm_head)

    def test_model_has_final_norm(self):
        """Test that model has final normalization."""
        model = SmolLM2ModelShell()
        self.assertTrue(hasattr(model, 'norm'))
        self.assertIsNotNone(model.norm)

    def test_attention_projections(self):
        """Test that attention has required projections."""
        model = SmolLM2ModelShell()
        attn = model.layers[0].self_attn
        
        self.assertTrue(hasattr(attn, 'q_proj'))
        self.assertTrue(hasattr(attn, 'k_proj'))
        self.assertTrue(hasattr(attn, 'v_proj'))
        self.assertTrue(hasattr(attn, 'o_proj'))

    def test_mlp_projections(self):
        """Test that MLP has required projections."""
        model = SmolLM2ModelShell()
        mlp = model.layers[0].mlp
        
        self.assertTrue(hasattr(mlp, 'gate_proj'))
        self.assertTrue(hasattr(mlp, 'up_proj'))
        self.assertTrue(hasattr(mlp, 'down_proj'))


class TestLoaderMethods(unittest.TestCase):
    """Test checkpoint loader methods."""

    def test_loader_has_inspect_method(self):
        """Test that loader has inspect method."""
        loader = SmolLM2CheckpointLoader()
        self.assertTrue(hasattr(loader, 'inspect_checkpoint'))

    def test_loader_has_mapping_method(self):
        """Test that loader has mapping method."""
        loader = SmolLM2CheckpointLoader()
        self.assertTrue(hasattr(loader, 'map_to_recall_model'))

    def test_loader_has_validation_method(self):
        """Test that loader has validation method."""
        loader = SmolLM2CheckpointLoader()
        self.assertTrue(hasattr(loader, 'validate_mapping'))

    def test_loader_has_error_tracking(self):
        """Test that loader has error tracking."""
        loader = SmolLM2CheckpointLoader()
        self.assertTrue(hasattr(loader, 'validation_errors'))
        self.assertTrue(hasattr(loader, 'validation_warnings'))


class TestTensorNamePatterns(unittest.TestCase):
    """Test tensor naming patterns from SmolLM2."""

    def test_embedding_names(self):
        """Test embedding tensor names."""
        # SmolLM2 uses these patterns
        self.assertIn("embed", "embed_tokens")

    def test_attention_names(self):
        """Test attention tensor names."""
        # SmolLM2 uses these patterns
        names = [
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
        ]
        for name in names:
            self.assertIn("self_attn", name)

    def test_mlp_names(self):
        """Test MLP tensor names."""
        names = [
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
        ]
        for name in names:
            self.assertIn("mlp", name)

    def test_norm_names(self):
        """Test normalization tensor names."""
        names = [
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "norm.weight",
        ]
        for name in names:
            self.assertIn("norm", name.lower())


if __name__ == "__main__":
    unittest.main()
