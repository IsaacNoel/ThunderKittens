# fftconv_bwd — Change Log

## Phase 6: N=1024 support for dk_f kernel

### New: fft_dkf_1024_template (filter gradient for N=1024)

**File:** `fftconv_bwd_pc.cu`

**Problem:** The dk_f (filter gradient) kernel only supported N=4096. N=1024 was missing,
meaning the full backward pass could not run at sequence length 1024.

**What was added:**
- `fftconv_dkf_1024_layout`: Layout struct for N=1024 dk_f. Uses `gl<bf16, -1, -1, 32, 32>`
  seq/out layouts (32x32 subtiles) and `cgl<gl<bf16, 1, 1, 64, 64>>` for FFT matrices.
- `fft_dkf_1024_template`: Template struct combining the subtile<32,32> packing pattern
  from the 1024 du template (4 batch elements per 64x64 shared tile) with the dk_f compute
  logic from the 4096 dk_f template (monarch_fft + conjugate multiply).
- `fft_dkf_template_internal` / `fft_dkf_template<N>`: Template dispatch that selects
  1024 or 4096 dk_f template based on N.
- `setup_dkf_globals<SEQ>` / `launch_dkf<SEQ>`: Templated setup and launch functions
  (matching the pattern of `setup_du_globals` / `launch_du`).
- Updated `fftconv_bwd_dkf` PyTorch binding to accept N=1024 or N=4096 and dispatch
  accordingly.

**Test changes:** `test_correctness.py` now runs all tests (du, dk_f, autograd) for both
N=4096 and N=1024, including stress tests across batch/head configurations.

**Impact:** N=4096 path is completely unchanged. Only additive changes.

## Phase 2: Correctness Audit

### Bug Fix 1: Missing `.conj()` in `pytorch_ref.py`

**File:** `pytorch_ref.py`, lines 59 and 62

**Problem:** The standalone backward reference was missing conjugation of frequency-domain
operands. The formulas `IFFT(FFT(dy) * FFT(k))` and `IFFT(FFT(dy) * FFT(u))` are wrong —
the correct backward of circular convolution requires conjugation:
`du = IFFT(FFT(dy) * conj(FFT(k)))` and `dk = IFFT(FFT(dy) * conj(FFT(u)))`.
For real signals, `conj(FFT(k)) != FFT(k)` since the DFT of a real signal is Hermitian
symmetric, not real. This caused max errors of ~1e3 vs autograd.

**Fix:** Added `.conj()` to both formulas in `ref_fftconv_bwd`.

**Note:** The monarch backward reference (`pytorch_ref_bwd.py`) and all test files already
had the correct conjugated formulas. Only this standalone validation script was affected.

### Bug Fix 2: iters_per_head mismatch in 1024 consumer head-change logic

**File:** `fftconv_bwd_pc.cu`, line ~184 (in `fft_bwd_1024_template::consumer::compute`)

**Problem:** The consumer's head-change detection used `NUM_CONSUMER_WARPGROUPS` as the
divisor for `iters_per_head`, but `common_setup` and `producer::load/store` both use
`NUM_CONSUMER_WARPGROUPS * 4`. The 1024 variant packs 4 subtiles (32x32) into each 64x64
shared memory tile, so each iteration processes 4x more batch elements than the consumer
assumed. This caused the consumer to detect head changes too late, loading the wrong head's
filter data.

**Fix:** Changed divisor from `NUM_CONSUMER_WARPGROUPS` to `NUM_CONSUMER_WARPGROUPS*4` to
match `common_setup` and producer.

**Impact:** Only affects N=1024 variant. N=4096 (the primary tested variant) was unaffected.

**Note:** The forward kernel (`kernels/fftconv/fftconv_pc.cu`, line 139) has the identical
bug in its 1024 variant. That file was not modified per project rules.

### Audit Results (no changes needed)

**4096 du kernel (fftconv_bwd_pc.cu):** The adjoint chain is mathematically correct:
- Forward: `F@X → *(tw/N) → @F → *kf → @Finv → *twinv_t → Finv@X`
- Backward: `F@dy → *conj(twinv_t) → @F → *conj(kf) → @Finv → *conj(tw/N) → Finv@d`
- All conjugations are performed Python-side before kernel launch. Verified against
  `ref_fftconv_bwd` (closed-form) and PyTorch autograd.

**4096 dk_f kernel (fftconv_bwd_pc.cu):** Correctly computes
`FFT(dy) * conj(FFT(u))` per batch element using unnormalized Monarch FFT.
Batch reduction (sum over B) is performed Python-side in float32 for precision.

**Fused kernel (fftconv_bwd_fused.cu):** Correctly combines du and dk_f in a single
kernel launch. Uses unnormalized `tw` for FFT steps (valid since `conj(twinv_t) = tw`
for unnormalized twiddles). dk_f accumulator is bf16 in shared memory — acceptable for
moderate batch sizes, but may lose precision for very large B.

**Shared memory aliasing:** Verified that `input_tile`, `fft_dy_save`, `dk_f_acc`, and
`tmp` in the fused kernel are never written/read concurrently across phases. All phase
transitions are guarded by `__syncthreads()`.

## Phase 3: Code Quality

### Input validation (matching fftconv forward style)

**Files:** `fftconv_bwd_pc.cu`, `fftconv_bwd_fused.cu`

Added `TORCH_CHECK` assertions to all PyTorch binding functions:
- `fftconv_bwd`: validates N (1024 or 4096), N1*N1==N, dy shape, matrix dimensions (64x64)
- `fftconv_bwd_dkf`: validates N==4096, N1==64, dy/u shapes, matrix dimensions
- `fftconv_bwd_fused`: validates N==4096, N1==64, dy/u shapes, matrix dimensions

Matches the validation style of `kernels/fftconv/fftconv_pc.cu` (forward kernel).

## Phase 4: Benchmarks

Benchmark infrastructure already existed in `benchmarks/run_benchmarks.py`. No changes needed.

The script benchmarks TK fused backward kernel vs PyTorch autograd backward for N=4096
across 6 configurations (B=2..32, H=4..16). Results are saved as JSON and PNG.

Note: The requested sequence lengths (768, 1536, 3072, 6144, 12288) are not supported by
the Monarch decomposition (requires N = N1^2, a perfect square). Supported sizes: 1024, 4096.

## Phase 5: PR Proposal

Updated `PR_PROPOSAL.md` with input validation section and benchmark references.
