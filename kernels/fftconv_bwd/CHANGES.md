# fftconv_bwd — Change Log

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
