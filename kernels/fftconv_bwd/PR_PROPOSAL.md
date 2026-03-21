# PR Proposal: FFTConv Backward Pass (`kernels/fftconv_bwd/`)

## Summary

This PR adds a CUDA backward pass for FFT convolution using ThunderKittens primitives, mirroring the existing forward kernel in `kernels/fftconv/`. The backward kernel computes both the input gradient (`du`) and filter gradient (`dk_f`) for the Monarch-decomposed FFT convolution, enabling end-to-end training of models that use this operation.

The backward pass is implemented as two LCSF producer-consumer kernels in `fftconv_bwd_pc.cu`:
- `fftconv_bwd`: computes the input gradient `du`
- `fftconv_bwd_dkf`: computes the filter gradient `dk_f`

## Mathematical Basis

### Forward Operation (Monarch FFT Convolution)
The forward pass decomposes the FFT/IFFT into a sequence of small matrix multiplies on 64×64 tiles:

```
y = real(Finv @ ((((F @ x) ⊙ tw) @ F) ⊙ kf) @ Finv) ⊙ twinv_t))
```

Where `F` is the DFT matrix, `Finv` is the IDFT matrix, `tw`/`twinv_t` are twiddle factors, and `kf` is the frequency-domain filter.

### Backward (du): Adjoint of the Forward Chain
The key insight is that the adjoint of the forward chain maps to the **same sequence of matrix multiplies**, just with conjugated pointwise operands:

| Step | Forward | Backward (du) |
|------|---------|---------------|
| Left multiply | `F @ x` | `F @ dy` |
| Twiddle | `⊙ tw` | `⊙ conj(twinv_t)` = `⊙ tw_unnorm` |
| Right multiply | `@ F` | `@ F` |
| Filter | `⊙ kf` | `⊙ conj(kf)` |
| Right multiply | `@ Finv` | `@ Finv` |
| Twiddle | `⊙ twinv_t` | `⊙ conj(tw)` |
| Left multiply | `Finv @ d` | `Finv @ d` |

This works because `F^H = Finv` and `Finv^H = F` for the DFT matrices. The Python-side conjugation means the kernel code is structurally identical to the forward.

### Backward (dk_f): Filter Gradient
```
dk_f_partial[b,h] = FFT(dy[b,h]) ⊙ conj(FFT(u[b,h]))
dk_f[h] = Σ_b dk_f_partial[b,h]
```

Both dy and u go through the same Monarch FFT (`F @ x → ⊙ tw → @ F`), then the results are element-wise multiplied with conjugation.

## ThunderKittens Primitives Used

| Primitive | Usage | Why |
|-----------|-------|-----|
| `st_bf<64,64>` / `cst_bf<64,64>` | Shared memory tiles (real/complex) | Core data unit for 64×64 Monarch tiles |
| `rt_bf<16,64>` / `crt_bf<16,64>` | Register tiles (real/complex) | Per-warp working data for compute |
| `crt_fl<16,64>` | Complex float register tile | MMA accumulator (float32 for precision) |
| `warpgroup::mm_AB` | Matrix multiply (WGMMA) | Tensor core matrix multiply for F@X, X@F, etc. |
| `warpgroup::mma_async_wait` | Wait for async MMA | Synchronize after WGMMA calls |
| `warp::mul` / `warp::add` / `warp::sub` | Pointwise register ops | Twiddle factors, filter multiply, dk_f accumulation |
| `warpgroup::load` / `warpgroup::store` | Register ↔ shared memory | Move data between compute and storage |
| `warp::tma::load_async` / `store_async` | TMA global ↔ shared | Async global memory transfers (4096 variant) |
| `warp::load_async` | CP.ASYNC global → shared | Async loads for 1024 variant (subtiled) |
| `group<N>::load` / `group<N>::sync` | Multi-warp cooperative ops | Loading constant matrices, synchronization |
| LCSF framework (`prototype::lcsf`) | Load-Store-Compute-Finish | Producer-consumer pipeline for overlapping I/O and compute |

## Key Design Decisions

1. **Reusing forward structure for du**: Rather than implementing the adjoint from scratch, the du kernel reuses the forward kernel's compute pipeline verbatim. Conjugation is done Python-side, eliminating code duplication and ensuring the backward tracks any future forward changes.

2. **dk_f batch reduction**: `fftconv_bwd_dkf` outputs per-batch partials and reduces them to `dk_f[h]` in float32 on the Python side. Float32 reduction avoids bf16 precision loss for large batch sizes.

3. **Persistent grid (132 blocks)**: Both kernels use a persistent grid of 132 blocks (matching H100 SM count), iterating over heads and batches. This amortizes kernel launch overhead across many heads.

4. **1024 vs 4096 variants**: The 1024 variant packs 4 batch elements as 32×32 subtiles within each 64×64 shared tile; the 4096 variant uses TMA for full 64×64 tile transfers. Both N=1024 and N=4096 are supported for du and dk_f.

## Correctness Verification

Three levels of validation:

1. **Closed-form reference** (`ref_fftconv_bwd`): `du = IFFT(FFT(dy) * conj(FFT(k)))`, `dk = IFFT(FFT(dy) * conj(FFT(u)))` — no dependence on forward pass
2. **Monarch decomposition reference** (`pytorch_ref_bwd.py`): Tile-by-tile backward matching the CUDA kernel's exact computation steps, verified against the closed-form
3. **Autograd cross-validation**: Full forward → loss → backward chain via PyTorch autograd, compared against kernel outputs

All tests pass with <5% relative error (expected for bf16 computation with N=4096).

### Bugs Found and Fixed
- `pytorch_ref.py`: Missing `.conj()` in backward formulas (standalone reference only; test files were correct)
- `fftconv_bwd_pc.cu`: 1024 variant head-change detection used wrong `iters_per_head` divisor

See `CHANGES.md` for full details.

## Performance Characteristics

The TK backward kernels:
- Use identical compute structure to the forward (7 steps: 4 matrix multiplies + 3 pointwise ops)
- du and dk_f together do 14 matrix multiplies per (batch, head) pair with full LCSF pipeline overlap
- Shared memory: ~224KB peak (2 pipeline stages for dk_f 1024 variant), within H100's 227KB limit

Benchmark infrastructure is in `benchmarks/run_benchmarks.py` — requires H100 GPU for actual
timing. Results saved to `benchmarks/benchmark_results.json` and PNG plots per sequence length.

### Input Validation

Added `TORCH_CHECK` assertions to all PyTorch binding functions, matching the forward kernel's
validation style. Validates N (1024 or 4096), N1*N1==N, tensor shapes, and matrix dimensions.

## Reviewer Notes

1. **Forward kernel 1024 bug**: The forward kernel (`fftconv/fftconv_pc.cu`, line 139) has the same `iters_per_head` bug as the backward's 1024 variant. The backward's copy was fixed but the forward was not modified per project rules.

2. **No causal mode**: The current implementation handles non-causal FFT convolution only. The forward kernel also appears non-causal for the tile-level operation; causal masking would be applied at a higher level.
