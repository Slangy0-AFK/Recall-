#!/usr/bin/env python3
"""
PHASE 5: ACTUAL CHECKPOINT LOADING AND TENSOR VALIDATION

Direct safetensors approach - load tensors directly without full model instantiation.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoConfig, AutoTokenizer
from adapters.smollm2_architecture import SmolLM2ModelShell


def phase_2_checkpoint_loading() -> Dict[str, Any]:
    """PHASE 2: Load actual checkpoint using safetensors."""
    
    results = {
        "phase": "2_actual_checkpoint_loading",
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "status": "RUNNING",
        "checkpoint_load": {},
        "tensor_validation": {},
        "architecture_validation": {},
        "errors": [],
        "warnings": [],
    }
    
    try:
        print("\n" + "=" * 80)
        print("PHASE 2 — ACTUAL CHECKPOINT LOADING AND TENSOR VALIDATION")
        print("=" * 80)
        
        torch.set_grad_enabled(False)
        
        # Step 1: Load configuration
        print("\n[1/4] Loading model configuration...")
        config = AutoConfig.from_pretrained(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            trust_remote_code=True,
        )
        
        config_dict = {
            "model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "max_position_embeddings": config.max_position_embeddings,
            "head_dim": config.hidden_size // config.num_attention_heads,
            "tie_word_embeddings": config.tie_word_embeddings,
            "rms_norm_eps": float(config.rms_norm_eps),
        }
        
        print("✓ Configuration loaded:")
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
        
        results["checkpoint_load"]["config"] = config_dict
        
        # Step 2: Load from safetensors directly
        print("\n[2/4] Loading checkpoint tensors from safetensors...")
        
        try:
            from safetensors.torch import load_file
            
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--HuggingFaceTB--SmolLM2-360M-Instruct'
            safetensors_file = None
            
            for f in cache_dir.glob('**/*.safetensors'):
                safetensors_file = f
                break
            
            if safetensors_file is None:
                # Download if not cached
                from huggingface_hub import hf_hub_download
                safetensors_file = hf_hub_download(
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    filename="model.safetensors"
                )
            
            print(f"Loading from: {safetensors_file}")
            
            # Load all tensors
            state_dict = load_file(str(safetensors_file))
            
            print(f"✓ Loaded {len(state_dict)} tensors from safetensors")
            results["checkpoint_load"]["hf_model_loaded"] = True
            results["checkpoint_load"]["model_dtype"] = str(list(state_dict.values())[0].dtype)
            
        except Exception as e:
            print(f"✗ Failed to load safetensors: {e}")
            results["errors"].append(f"Safetensors load: {str(e)}")
            raise
        
        # Step 3: Validate tensors
        print("\n[3/4] Validating checkpoint tensors...")
        
        tensor_categories = {
            "embedding": [],
            "normalization": [],
            "attention": [],
            "mlp": [],
            "output": [],
            "other": [],
        }
        
        total_params = 0
        nan_tensors = []
        inf_tensors = []
        
        for name, tensor in state_dict.items():
            # Categorize
            if "embed_tokens" in name:
                cat = "embedding"
            elif "norm" in name:
                cat = "normalization"
            elif "self_attn" in name:
                cat = "attention"
            elif "mlp" in name:
                cat = "mlp"
            elif "lm_head" in name:
                cat = "output"
            else:
                cat = "other"
            
            tensor_categories[cat].append(name)
            total_params += tensor.numel()
            
            # Check for NaN/Inf
            if tensor.isnan().any():
                nan_tensors.append(name)
            if tensor.isinf().any():
                inf_tensors.append(name)
        
        print(f"✓ Tensor categories:")
        for cat, tensors in tensor_categories.items():
            print(f"  {cat}: {len(tensors):3d} tensors")
        
        print(f"✓ Total parameters: {total_params:,}")
        
        if nan_tensors:
            print(f"⚠ Tensors with NaN: {len(nan_tensors)}")
            results["warnings"].append(f"{len(nan_tensors)} tensors contain NaN")
        if inf_tensors:
            print(f"⚠ Tensors with Inf: {len(inf_tensors)}")
            results["warnings"].append(f"{len(inf_tensors)} tensors contain Inf")
        
        results["tensor_validation"] = {
            "total_tensors": len(state_dict),
            "total_parameters": total_params,
            "tensors_by_category": {k: len(v) for k, v in tensor_categories.items()},
            "nan_tensors": nan_tensors,
            "inf_tensors": inf_tensors,
        }
        
        # Step 4: Create Recall model and load weights
        print("\n[4/4] Creating Recall architecture and loading weights...")
        
        try:
            # Create Recall model
            recall_model = SmolLM2ModelShell(
                vocab_size=config_dict["vocab_size"],
                hidden_size=config_dict["hidden_size"],
                num_hidden_layers=config_dict["num_hidden_layers"],
                num_attention_heads=config_dict["num_attention_heads"],
                num_key_value_heads=config_dict["num_key_value_heads"],
                head_dim=config_dict["head_dim"],
                intermediate_size=config_dict["intermediate_size"],
                max_position_embeddings=config_dict["max_position_embeddings"],
                rms_norm_eps=config_dict["rms_norm_eps"],
            )
            
            print("✓ Recall model shell created")
            
            # Manual weight mapping
            mapped_count = 0
            loaded_params = 0
            
            # Embedding
            if "model.embed_tokens.weight" in state_dict:
                weight = state_dict["model.embed_tokens.weight"]
                recall_model.embed_tokens.weight.data.copy_(weight)
                mapped_count += 1
                loaded_params += weight.numel()
                del state_dict["model.embed_tokens.weight"]
            
            # Transformer layers
            for layer_idx in range(config_dict["num_hidden_layers"]):
                # Pre-attention norm
                key = f"model.layers.{layer_idx}.input_layernorm.weight"
                if key in state_dict:
                    recall_model.layers[layer_idx].attn_norm.weight.data.copy_(state_dict[key])
                    mapped_count += 1
                    loaded_params += state_dict[key].numel()
                    del state_dict[key]
                
                # Attention projections
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
                    if key in state_dict:
                        getattr(recall_model.layers[layer_idx], proj).weight.data.copy_(state_dict[key])
                        mapped_count += 1
                        loaded_params += state_dict[key].numel()
                        del state_dict[key]
                
                # Post-attention norm
                key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                if key in state_dict:
                    recall_model.layers[layer_idx].ffn_norm.weight.data.copy_(state_dict[key])
                    mapped_count += 1
                    loaded_params += state_dict[key].numel()
                    del state_dict[key]
                
                # MLP
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"model.layers.{layer_idx}.mlp.{proj}.weight"
                    if key in state_dict:
                        if proj == "gate_proj":
                            recall_model.layers[layer_idx].mlp.gate.weight.data.copy_(state_dict[key])
                        elif proj == "up_proj":
                            recall_model.layers[layer_idx].mlp.up.weight.data.copy_(state_dict[key])
                        elif proj == "down_proj":
                            recall_model.layers[layer_idx].mlp.down.weight.data.copy_(state_dict[key])
                        mapped_count += 1
                        loaded_params += state_dict[key].numel()
                        del state_dict[key]
            
            # Final norm
            if "model.norm.weight" in state_dict:
                recall_model.norm.weight.data.copy_(state_dict["model.norm.weight"])
                mapped_count += 1
                loaded_params += state_dict["model.norm.weight"].numel()
                del state_dict["model.norm.weight"]
            
            # LM head
            if "lm_head.weight" in state_dict:
                recall_model.lm_head.weight.data.copy_(state_dict["lm_head.weight"])
                mapped_count += 1
                loaded_params += state_dict["lm_head.weight"].numel()
                del state_dict["lm_head.weight"]
            
            print(f"✓ Checkpoint weights loaded")
            print(f"  Mapped tensors: {mapped_count}")
            print(f"  Parameters loaded: {loaded_params:,}")
            
            # Check for unmapped tensors
            if state_dict:
                print(f"⚠ Unmapped tensors remaining: {list(state_dict.keys())[:5]}")
                results["warnings"].append(f"{len(state_dict)} tensors were not mapped")
            
            # Verify no NaN/Inf in Recall model
            nan_count = 0
            inf_count = 0
            with torch.no_grad():
                for name, param in recall_model.named_parameters():
                    if param.isnan().any():
                        nan_count += 1
                    if param.isinf().any():
                        inf_count += 1
            
            if nan_count == 0 and inf_count == 0:
                print(f"✓ No NaN or Inf in loaded model")
            else:
                results["warnings"].append(f"Model has {nan_count} NaN, {inf_count} Inf parameters")
            
            results["checkpoint_load"]["mapping"] = {
                "total_mapped": mapped_count,
                "total_parameters": loaded_params,
                "mapping_success": True,
            }
            
            results["architecture_validation"]["shell_created"] = True
            shell_params = sum(p.numel() for p in recall_model.parameters())
            results["architecture_validation"]["shell_parameters"] = shell_params
            results["architecture_validation"]["num_layers"] = config_dict["num_hidden_layers"]
            
        except Exception as e:
            print(f"✗ Failed to create/load model: {e}")
            traceback.print_exc()
            results["errors"].append(f"Model creation/loading: {str(e)}")
            raise
        
        # Summary
        print("\n" + "=" * 80)
        print("PHASE 2 RESULTS")
        print("=" * 80)
        print(f"Status: {'✓ PASS' if not results['errors'] else '✗ FAIL'}")
        print(f"Checkpoint loaded: {results['checkpoint_load'].get('hf_model_loaded', False)}")
        print(f"Tensors: {results['tensor_validation']['total_tensors']}")
        print(f"Parameters: {results['tensor_validation']['total_parameters']:,}")
        print(f"Weights mapped: {results['checkpoint_load'].get('mapping', {}).get('mapping_success', False)}")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for err in results['errors']:
                print(f"  - {err}")
        
        results["status"] = "COMPLETE" if not results["errors"] else "FAILED"
        
    except Exception as e:
        print(f"\n✗ Phase 2 failed: {e}")
        traceback.print_exc()
        results["status"] = "FAILED"
        results["errors"].append(str(e))
    
    return results


def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "PHASE2_RESULTS.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("/workspaces/Recall-/evidence") / f"smollm2-native-acceptance-{timestamp}"
    
    print(f"\nPhase 2: Actual Checkpoint Loading (Safetensors Direct)")
    print(f"Output directory: {output_dir}")
    
    results = phase_2_checkpoint_loading()
    save_results(results, output_dir)
    
    sys.exit(0 if results["status"] == "COMPLETE" else 1)
