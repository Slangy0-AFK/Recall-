# DEEPTRACE_LIMITATIONS.md

## Known Limitations and Considerations

### Framework Limitations

#### 1. **Demonstration Mode (Current State)**
- Smoke tests run without actual model inference
- Actual answer generation is a placeholder
- Results show 0% accuracy because no model is integrated
- Tests validate framework structure, not retrieval capability

**Impact**: To measure real retrieval, integrate an actual language model.

#### 2. **Position Tracking Accuracy**
- Relies on model's tokenizer for accurate token counting
- Different tokenizers produce different token counts
- Fallback word-level tokenization is imprecise
- Mismatch between expected and actual positions invalidates distance measurements

**Mitigation**: Verify tokenizer output on sample texts before running full tests.

#### 3. **Cache Diagnostics**
- Cache status (retained/evicted) is "unknown" for most frameworks
- Some models don't expose internal cache state
- Can't directly observe whether key tokens are retained
- Correlation with retrieval failure is therefore incomplete

**Mitigation**: Use models that expose cache operations, or add instrumentation.

#### 4. **Model Inference Overhead**
- Large distance bands (32K, 65K, 100K) require substantial GPU memory
- Inference time scales with sequence length
- May take hours or days for 100K token tests
- Automatic recovery from out-of-memory is not implemented

**Mitigation**: Start with smoke tests, gradually increase distances.

### Experimental Limitations

#### 5. **Synthetic Facts**
- Facts use artificial formats (QX-XXXX, LM-XXXX, etc.)
- These codes unlikely to appear in model training data (hopefully)
- Real-world retrieval may work differently on natural facts
- Results may not generalize to natural language retrieval

**Mitigation**: Later validate with natural-language facts.

#### 6. **Controlled Distractor Distribution**
- Distractor text is artificially generated, not naturally distributed
- May not represent realistic information density
- Model's attention patterns may differ with real intervening text

**Mitigation**: Later test with real corpus text as distractors.

#### 7. **Deterministic Questions**
- All questions use fixed phrasing
- No paraphrase variation
- Model sees consistent question structure
- May not test true understanding vs. pattern matching

**Mitigation**: Later add question paraphrase variations.

#### 8. **Training Data Contamination**
- If model training data contains similar codes or patterns, results are invalid
- No control for memorization vs. retrieval
- Code patterns like "equipment code is [XXXX]" might appear in training

**Mitigation**: Validate that expected answers don't appear in model training data.

### Interpretive Limitations

#### 9. **Single Test Run Variance**
- One run may show high or low accuracy by chance
- Random sampling of facts, positions, and model behavior
- No confidence intervals or statistical significance testing

**Mitigation**: Run multiple times with different seeds.

#### 10. **Failure Mode Ambiguity**
- Low accuracy at 4K could mean:
  - Model attention can't reach back
  - Cache was evicted
  - Model didn't learn to retrieve
  - Tokenizer position tracking is wrong
  - Facts were forgotten during preprocessing
  
**Impact**: Root cause analysis requires additional investigation.

#### 11. **Cache Eviction May Not Be Sole Cause**
- Even if token is in cache, model may fail to attend to it
- Attention weights might go to distractors
- KV vectors might be corrupted or degraded
- Cache retention ≠ Cache utilization

**Implication**: Seeing cache retention doesn't guarantee success.

#### 12. **Generalization Uncertainty**
- Success on synthetic facts doesn't guarantee real retrieval
- Success on equipment codes doesn't imply location codes work same way
- Success at 4K doesn't predict 32K performance
- Different facts may have different difficulty

**Mitigation**: Test multiple fact types and distances.

### Implementation Limitations

#### 13. **No Automatic Recovery**
- Framework doesn't auto-recover from model crashes
- Out-of-memory errors terminate test
- Network errors (if using API) are not retried
- Manual intervention required for failures

#### 14. **Limited Control Sophistication**
- Positive controls are basic (short-distance, known-answer)
- Negative controls don't exhaustively test hallucination
- No controls for data leakage through model weights

#### 15. **No Adaptive Testing**
- Distance bands are fixed
- Can't automatically search for retrieval boundary
- Must specify all distances in advance

#### 16. **Result Reproducibility Dependencies**
- Reproducibility requires:
  - Same PyTorch/transformer versions
  - Same model weights
  - Same tokenizer
  - Same random seed
  - Same input format
- Any change breaks reproducibility

### Operational Limitations

#### 17. **Smoke Tests Only (Initial)**
- Larger distances (8K+) require explicit approval before running
- Cannot discover retrieval cliff without approval
- May require multiple approval rounds

#### 18. **Ground Truth Protection vs. Inspection**
- Ground truth is protected from model
- But protected files are human-readable
- Accidental exposure is possible in log files
- Requires discipline to keep separated

#### 19. **Output Storage**
- All results saved to separate directory
- Original 100K evidence untouched
- But large datasets (100K tokens) produce large output files
- May consume significant disk space

#### 20. **Model Availability**
- Tests require a specific model to be loaded
- Large models may need GPU with high VRAM
- Not all models available for all users
- Licensing/access restrictions apply

### Statistical Limitations

#### 21. **Small Sample Size per Distance**
- Smoke tests: 10 queries total (5 facts × 2 distances)
- Statistical power is low
- Confidence intervals would be wide
- May see spurious results

#### 22. **No Cross-Distance Analysis**
- Accuracy at 1K ≠ Accuracy at 4K
- No regression or trend analysis
- Cannot fit models to accuracy curve
- Limited inference about underlying mechanism

#### 23. **Control Assumptions**
- Assumes positive controls are sufficient to validate setup
- Assumes negative controls catch all hallucinations
- May have false negatives

### Documentation & Methodology Limitations

#### 24. **Incomplete Architecture Details**
- Some Zeta component usages are placeholder
- Actual model integration not yet documented
- Expected API changes as tests mature

#### 25. **No Published Baseline**
- No comparison to other retrieval tests
- Don't know what "good" accuracy looks like
- No benchmark for the benchmark

#### 26. **Methodology Not Peer-Reviewed**
- Design choices not validated by community
- May discover issues during implementation
- Ready for feedback and iteration

### Mitigation Strategy

**For Early Tests** (1K, 4K tokens):
- Accept limitations as framework validation
- Use results to test infrastructure
- Iterate on methodology based on issues

**For Extended Tests** (8K+ tokens):
- Plan multiple runs for variance estimation
- Prepare root-cause analysis framework
- Document failures thoroughly

**For Final Report**:
- Clearly state all limitations
- Distinguish observations from interpretations
- Recommend future improvements
- Provide reproducibility commands

---

**Version**: 0.1.0  
**Status**: Initial Implementation  
**Last Updated**: 2026-11-18
