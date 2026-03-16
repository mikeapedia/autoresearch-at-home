# Session Summary: 150+ Experiments on NVIDIA B200

## Starting Point
- **Swarm best:** 0.959721 (by phoenix)
- **Final best:** 0.926381 (by forge) — **0.033 improvement (3.6% relative)**

---

## What Worked (kept changes)

### Tier 1 — Breakthroughs (>0.002 improvement each)

| Change | Improvement | How |
|--------|------------|-----|
| **FA4 CUTLASS via torch.library.custom_op** | +0.008 | Registered FA4 fwd+bwd as custom ops so torch.compile treats them as opaque leaves. Replaced FlexAttention Triton backward (20% of CUDA time) with native Blackwell CUTLASS kernels (10%). MFU 15.3%→17.4% |
| **Inductor optimizations** | +0.004 | epilogue_fusion, aggressive_fusion, coordinate_descent_tuning, max-autotune with CUDA graphs, shape_padding, CUDA event timing, cudagraph_mark_step_begin |
| **Depth 14→12 at dim=768** | +0.003 | Shallower model processes more tokens (336M vs 289M). At 5-min budget, token count > depth |
| **Dim 640→768** | +0.002 | Wider model with more capacity. Sweet spot between throughput and quality |

### Tier 2 — Meaningful gains (0.0003-0.002)

| Change | Improvement |
|--------|------------|
| WSD 30/70 sqrt decay schedule | +0.0004 |
| FINAL_LR_FRAC 0.005→0.02 | +0.0006 |
| QK scale 1.10→1.15→1.20 (FA4 optimal: 1.15) | +0.0002 |
| VE gate scale 2→4 | +0.0002 |
| VE gate channels 128→64 | +0.0002 |
| MATRIX_LR tuning (0.032→0.028→0.025) | +0.001 |
| EMBEDDING_LR 1.0→1.2, UNEMBEDDING_LR 0.006→0.005 | +0.0001 |
| CUDA_DEVICE_MAX_CONNECTIONS=1 | +0.0001 |

---

## What Didn't Work

### Architecture changes (all worse)

| Experiment | val_bpb | Why it failed |
|-----------|---------|---------------|
| Parallel attn+MLP | 0.944 | Sequential application needed (MLP sees attn output) |
| Sandwich/Peri-LN norm | 0.934-0.943 | Extra norms constrain residual stream |
| Gated MLP (SwiGLU) | 0.956 | relu².square() is better for this setup |
| GQA (3 KV heads) | 0.937 | Attention quality loss > throughput gain |
| HEAD_DIM 64 (12 heads) | 0.943 | Worse per-token quality despite more tokens |
| MoE (4 experts, top-1) | crash | Data-dependent routing breaks torch.compile fullgraph |
| Differential Attention | crash | New 1D params incompatible with Muon grouping |
| DenseFormer DWA | 0.942 | O(n²) tensor ops kill throughput (15.8% MFU) |
| AttnRes (full 13-source) | 1.271 | Growing list breaks CUDA graphs (7.7% MFU) |
| AttnRes-lite (3-source) | 1.029 | Doesn't converge at 100M scale |
| Alternating MLP widths (6x/2x) | 0.938 | Different Muon groupings from varied shapes |
| MLP expansion 3x or 5x | 0.946/0.937 | 4x is the sweet spot |
| Depth 10, 11, 13, 14, 15, 16 | 0.941-0.966 | 12 is optimal for 5-min budget on B200 |
| Dim 896, 1024 | 0.949-0.975 | Too wide, not enough tokens |

### Optimizer changes (all worse)

| Experiment | val_bpb | Why |
|-----------|---------|-----|
| MANO optimizer (30+ configs) | 0.990 best | Simple norm+project can't match Newton-Schulz at 100M |
| HTMuon (4 variants) | 0.927-0.928 | Heavy-tailed correction hurts at well-conditioned 100M scale |
| Muon-VS (pre-ortho variance) | 0.928 | NorMuon already handles variance |
| Learnable per-head QK scale | 1.303 | Attention temperature too sensitive for gradient descent |
| LRM (learnable multipliers) | 0.942 | Extra AdamW params drop MFU 17.4%→16.2% |
| Gradient clipping | 0.936 | Muon+AdamW already stable |

### Loss/regularization (all worse)

| Experiment | val_bpb | Why |
|-----------|---------|-----|
| Label smoothing 0.1 | 1.280 | BPB metric directly punishes spreading probability |
| Z-loss 1e-4 | 0.928 | Softcap already constrains logits, adds compute |
| Focal loss (γ=0.3-1.0) | 0.936-0.941 | Deprioritizes easy tokens that matter for BPB |

### Training tricks (all worse)

| Experiment | val_bpb | Why |
|-----------|---------|-----|
| Stochastic depth 10% | 1.014 | torch.rand() breaks torch.compile graph |
| Embedding dropout 5% | 0.969 | Hurts representations |
| EMA (various configs) | 0.927-0.994 | WSD sqrt decay already provides smooth convergence |
| Batch warmup (grad_accum 1→4) | 0.936 | Smaller matmuls underutilize B200 tensor cores |
| Batch 2^15 (4x more steps) | 0.949 | Gradient noise too high |
| Batch 2^18 (bigger matmuls) | 0.936 | Fewer optimizer steps hurts more |
| Warmup 1-10% | 0.941-0.945 | Hurts with full decay schedule |
| Cosine LR decay | 1.045 | Keeps LR too high too long with WARMDOWN=1.0 |
| Cyclic LR perturbation | 0.937 | Noise doesn't help |
| Constant weight decay | 0.938 | Decaying WD better near convergence |

### FP8/quantization (all worse)

| Experiment | val_bpb | Why |
|-----------|---------|-----|
| Per-tensor FP8 (torchao) | 0.942 | Quality loss too high at 100M params |
| MXFP8 (torchao+cu130) | 0.960 | Quantize/dequantize overhead > benefit at dim=768 |

### Init/normalization (all worse)

| Experiment | val_bpb | Why |
|-----------|---------|-----|
| wte init std 0.8→1.0 | 0.945 | 0.8 is better |
| c_proj init small non-zero | 0.936 | Zero init (identity) is deliberate |
| Embedding × √d_model | 0.936 | RMSNorm right after cancels it |
| LayerNorm 1/√(layer) scaling | 0.929 | Conflicts with learnable resid_lambdas |
| resid_lambdas depth gradient | 0.927 | Learnable params find optimal from any init |
| skip2_lambdas init 0→0 or 0.08 | 0.944/0.965 | 0.05 is precisely optimal (very sensitive!) |
| RMSNorm on VE embeddings | 0.938 | Scale of VE carries useful information |
| Softcap 11, 14, 15, removed | 0.936-0.952 | 13 is optimal |

---

## Fine-Grained Sweep Results (all confirmed at optimum)

Every parameter tested with ±1 step. All at their local minimum:

| Parameter | Optimal Value | Sensitivity |
|-----------|--------------|-------------|
| MATRIX_LR | 0.025 | ±0.001 = +0.0003-0.0006 |
| EMBEDDING_LR | 1.2 | ±0.1 = +0.0003 |
| UNEMBEDDING_LR | 0.005 | ±0.001 = +0.0005-0.0007 |
| SCALAR_LR | 1.0 | ±0.2 = +0.0007-0.0008 |
| WEIGHT_DECAY | 0.1525 | ±0.0075 = +0.0002-0.0006 |
| ADAM beta1 | 0.8 | ±0.05 = +0.0005-0.0008 |
| ADAM beta2 | 0.99 | fixed |
| Muon beta2 | 0.90 | ±0.02 = +0.0001-0.0005 |
| WARMDOWN_RATIO | 0.7 | ±0.05 = +0.0004-0.0005 |
| FINAL_LR_FRAC | 0.02 | ±0.005 = +0.0001-0.0002 |
| VE LR | 0.30 | ±0.05 = +0.0002 |
| VE WD | 0.01 | ±0.005/0.01 = +0.0007-0.0011 |

---

## What's Left Unexplored

### Blocked by constraints
- **MXFP8 at dim≥1024** — needs larger model where overhead < benefit. Blocked by disk space
- **Sliding window warmup** — changing window_size (int) invalidates CUDA graphs
- **torch._dynamo.compiled_autograd** — conflicts with FlexAttention
- **Fused QKV projection** — breaks Muon optimizer shape grouping

### Partially explored (need more work)
- **Mousse** (curvature-aware Muon) — needs Hessian estimation, complex
- **PRISM** (anisotropic spectral shaping) — needs eigenvalue computation
- **Data curriculum/ordering** — prepare.py is read-only

### Novel ideas not attempted
- **Mixture of Depths** — route easy tokens through fewer layers
- **Multi-scale architecture** — different layers at different resolutions
- **KV sharing across layers** — needs significant forward pass refactoring
- **Token Order Prediction (TOP)** — auxiliary ranking loss
- **Register tokens (MuToR)** — interleaved learnable tokens
- **Progressive model growing** — start small, expand mid-training
- **Self-distillation** — use early checkpoint logits as soft targets
- **Intra-document attention masking** — prevent cross-document attention (needs dataloader changes)

---

## Key Takeaways

1. **Throughput is king at 5-min budget** — any change that reduces tokens/step must provide outsized quality improvement to compensate
2. **torch.compile compatibility is a hard constraint** — MoE, stochastic depth, dynamic shapes all break fullgraph/CUDA graphs
3. **The Muon optimizer is optimal at 100M scale** — MANO, HTMuon, Muon-VS all tested and worse
4. **FA4 custom_op was the biggest win** — proper integration of native CUTLASS kernels with torch.compile via torch.library
5. **Every hyperparameter is at a sharp local optimum** — the config leaves no room for single-parameter improvements
