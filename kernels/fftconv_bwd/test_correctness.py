import torch
import numpy as np
from _C import fftconv_bwd, fftconv_bwd_dkf


# ============================================================================
# Reference functions (closed-form, source of truth)
# ============================================================================

def ref_fftconv(u, k, N):
    """Forward: Y = IFFT(FFT(u) * FFT(k))"""
    L = u.shape[-1]
    u_f = torch.fft.fft(u.float(), n=N)
    k_f = torch.fft.fft(k.float(), n=N)
    y_f = u_f * k_f
    y = torch.fft.ifft(y_f, n=N).real[..., :L].to(u.dtype).contiguous()
    return y

def ref_fftconv_bwd(u, k, dy, N):
    """Closed-form backward. No forward pass values needed."""
    L = u.shape[-1]
    dy_f = torch.fft.fft(dy.float(), n=N)
    k_f  = torch.fft.fft(k.float(), n=N)
    u_f  = torch.fft.fft(u.float(), n=N)
    du = torch.fft.ifft(dy_f * k_f.conj(), n=N).real[..., :L]
    dk = torch.fft.ifft((dy_f * u_f.conj()).sum(dim=0), n=N).real[..., :L]
    return du, dk


# ============================================================================
# Matrix / twiddle helpers
# ============================================================================

def fft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    return torch.exp(-2j * torch.pi * n * k / N)

def ifft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    return torch.exp(2j * torch.pi * n * k / N)

def compute_twiddle_factors_fft(n, m):
    n_a = torch.arange(n).view(-1, 1)
    m_a = torch.arange(m)
    N = n * m
    return torch.exp(-2j * torch.pi * n_a * m_a / N)

def compute_twiddle_factors_ifft(n, m):
    n_a = torch.arange(n).view(-1, 1)
    m_a = torch.arange(m)
    N = n * m
    return torch.exp(2j * torch.pi * n_a * m_a / N)


# ============================================================================
# Tile helpers for N=1024 (N1=32)
#
# The kernel always uses 64x64 shared memory tiles. For N=1024, 4 batch
# elements (each a 32x32 subtile) are packed into one 64x64 tile.
# Matrix arguments must therefore always be 64x64:
#   - F/Finv (matrix multiplies): block_diag(M_32, M_32) applies M_32
#     independently to each row-block / column-block of the 64x64 tile.
#   - Twiddles and per-head filters (pointwise): tile M_32 in all 4 quadrants
#     so the same values are applied to every batch subtile.
# ============================================================================

KERNEL_TILE = 64

def to_tile_block_diag(mat):
    """Lift an N1xN1 matrix to 64x64 for left/right matrix multiplies."""
    if mat.shape[-1] == KERNEL_TILE:
        return mat
    return torch.block_diag(mat, mat)

def to_tile_pointwise(mat):
    """Lift an N1xN1 (or HxN1xN1) matrix to 64x64 (or Hx64x64) for pointwise ops."""
    if mat.shape[-1] == KERNEL_TILE:
        return mat
    if mat.dim() == 2:
        return mat.repeat(2, 2)          # (N1,N1) -> (64,64)
    else:
        return mat.repeat(1, 2, 2)       # (H,N1,N1) -> (H,64,64)


# ============================================================================
# Prepare kernel inputs for du backward
# ============================================================================

def prepare_du_inputs(u, k, dy, B, H, N, N1):
    """Prepare all inputs for the du backward kernel.

    The du kernel is the forward kernel with conjugated pointwise operands:
        tw slot      <- conj(twinv_t)
        kf slot      <- conj(kf)
        twinv_t slot <- conj(tw)
    """
    # FFT / IFFT matrices — must be 64x64 for the kernel tile size.
    # For N1=32 (N=1024), use block_diag so F_32 applies independently to
    # each of the 4 batch subtiles packed into the 64x64 shared tile.
    f_mat = to_tile_block_diag(fft_matrix(N1))
    f_real = f_mat.real.to(torch.bfloat16).contiguous()
    f_imag = f_mat.imag.to(torch.bfloat16).contiguous()

    finv_mat = to_tile_block_diag(ifft_matrix(N1))
    finv_real = finv_mat.real.to(torch.bfloat16).contiguous()
    finv_imag = finv_mat.imag.to(torch.bfloat16).contiguous()

    # Twiddle factors — pointwise, so tile the same N1xN1 values across all
    # 4 quadrants of the 64x64 tile (all batch subtiles share the same twiddle).
    tw = to_tile_pointwise(compute_twiddle_factors_fft(N1, N1) / N)
    twinv = to_tile_pointwise(compute_twiddle_factors_ifft(N1, N1))

    # For the du backward kernel:
    #   tw_bwd      = conj(twinv_t) — twinv_t is pre-transposed in forward,
    #                 but twinv is symmetric (twinv[i,j] = twinv[j,i]),
    #                 so conj(twinv_t) = conj(twinv)
    #   kf_conj     = conj(kf)
    #   twinv_t_bwd = conj(tw)
    tw_bwd = twinv.conj()
    tw_bwd_real = tw_bwd.real.to(torch.bfloat16).contiguous()
    tw_bwd_imag = tw_bwd.imag.to(torch.bfloat16).contiguous()

    twinv_t_bwd = tw.conj()
    twinv_t_bwd_real = twinv_t_bwd.real.to(torch.bfloat16).contiguous()
    twinv_t_bwd_imag = twinv_t_bwd.imag.to(torch.bfloat16).contiguous()

    # Filter: conj(kf) — pointwise per head, tile to (H, 64, 64).
    k_f = torch.fft.fft(k.float(), n=N)
    k_fT = k_f.reshape(H, N1, N1).transpose(-1, -2)
    kf_conj = to_tile_pointwise(k_fT.conj())
    kf_conj_real = kf_conj.real.to(torch.bfloat16).contiguous()
    kf_conj_imag = kf_conj.imag.to(torch.bfloat16).contiguous()

    # Reshape dy for the kernel (B, H, N) -> (B, H, N1, N1)
    dy_real = dy.reshape(B, H, N1, N1).to(torch.bfloat16).contiguous()

    return (dy_real, kf_conj_real, kf_conj_imag,
            f_real, f_imag, finv_real, finv_imag,
            tw_bwd_real, tw_bwd_imag, twinv_t_bwd_real, twinv_t_bwd_imag)


# ============================================================================
# Prepare kernel inputs for dk_f
# ============================================================================

def prepare_dkf_inputs(u, dy, B, H, N, N1):
    """Prepare inputs for the dk_f kernel.

    Uses the actual (non-conjugated) FFT matrix and twiddle,
    since both dy and u go through the same forward FFT.

    IMPORTANT: twiddle is NOT normalized by 1/N here. In the forward pass,
    tw includes 1/N to normalize the FFT-IFFT pair. But for dk_f, both
    inputs go through FFT independently — using 1/N on both would give 1/N^2.
    We use unnormalized twiddles and let the test comparison account for it.
    """
    # F matrix: block_diag for N1=32 so it applies independently to each subtile.
    f_mat = to_tile_block_diag(fft_matrix(N1))
    f_real = f_mat.real.to(torch.bfloat16).contiguous()
    f_imag = f_mat.imag.to(torch.bfloat16).contiguous()

    # Twiddle: tile across all 4 quadrants (same twiddle for all batch subtiles).
    tw = to_tile_pointwise(compute_twiddle_factors_fft(N1, N1))  # no /N for dk_f
    tw_real = tw.real.to(torch.bfloat16).contiguous()
    tw_imag = tw.imag.to(torch.bfloat16).contiguous()

    dy_real = dy.reshape(B, H, N1, N1).to(torch.bfloat16).contiguous()
    u_real  = u.reshape(B, H, N1, N1).to(torch.bfloat16).contiguous()

    return dy_real, u_real, f_real, f_imag, tw_real, tw_imag


# ============================================================================
# Test du kernel
# ============================================================================

def test_du(B, H, N, N1):
    print(f"=== Testing du kernel (B={B}, H={H}, N={N}) ===\n")

    # Generate inputs
    torch.manual_seed(42)
    u  = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).float() / H
    k  = (torch.randn((H, N),    dtype=torch.bfloat16, device='cuda')).float() / H
    dy = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).float() / H

    # Reference
    du_ref, _ = ref_fftconv_bwd(u, k, dy, N)
    du_ref = du_ref.to(torch.bfloat16).to('cuda')

    # Prepare kernel inputs
    (dy_real, kf_conj_real, kf_conj_imag,
     f_real, f_imag, finv_real, finv_imag,
     tw_bwd_real, tw_bwd_imag,
     twinv_t_bwd_real, twinv_t_bwd_imag) = prepare_du_inputs(u, k, dy, B, H, N, N1)

    # Move to GPU
    dy_real = dy_real.cuda().contiguous()
    kf_conj_real = kf_conj_real.cuda().contiguous()
    kf_conj_imag = kf_conj_imag.cuda().contiguous()
    f_real = f_real.cuda().contiguous()
    f_imag = f_imag.cuda().contiguous()
    finv_real = finv_real.cuda().contiguous()
    finv_imag = finv_imag.cuda().contiguous()
    tw_bwd_real = tw_bwd_real.cuda().contiguous()
    tw_bwd_imag = tw_bwd_imag.cuda().contiguous()
    twinv_t_bwd_real = twinv_t_bwd_real.cuda().contiguous()
    twinv_t_bwd_imag = twinv_t_bwd_imag.cuda().contiguous()

    # Run kernel
    du_out = fftconv_bwd(
        dy_real,
        kf_conj_real, kf_conj_imag,
        f_real, f_imag,
        finv_real, finv_imag,
        tw_bwd_real, tw_bwd_imag,
        twinv_t_bwd_real, twinv_t_bwd_imag,
        B, H, N, N1
    )
    du_out = du_out.reshape(B, H, N)

    # Compare
    diff = (du_out.float() - du_ref.float()).abs()
    max_abs = diff.max().item()
    max_rel = max_abs / du_ref.float().abs().max().item()

    print(f"du max abs error: {max_abs:.6e}")
    print(f"du max rel error: {max_rel:.6e}")

    # Sample values
    b_idx = torch.randint(0, B, (1,)).item()
    h_idx = torch.randint(0, H, (1,)).item()
    print(f"\nSample du_out[{b_idx},{h_idx},100:110]: {du_out[b_idx, h_idx, 100:110]}")
    print(f"Sample du_ref[{b_idx},{h_idx},100:110]: {du_ref[b_idx, h_idx, 100:110]}")

    is_zero = torch.allclose(du_out, torch.zeros_like(du_out), atol=1e-3)
    print(f"\ndu is all zeros: {is_zero}")
    print(f"du PASSED: {max_rel < 0.05}\n")
    return max_rel < 0.05


# ============================================================================
# Test dk_f kernel
# ============================================================================

def test_dkf(B, H, N, N1):
    print(f"=== Testing dk_f kernel (B={B}, H={H}, N={N}) ===\n")

    # Generate inputs
    torch.manual_seed(42)
    u  = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).float() / H
    k  = (torch.randn((H, N),    dtype=torch.bfloat16, device='cuda')).float() / H
    dy = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).float() / H

    # Reference: dk in time domain
    _, dk_ref = ref_fftconv_bwd(u, k, dy, N)
    # Convert to permuted frequency domain.
    # With unnormalized twiddles, the kernel computes FFT(dy) * conj(FFT(u))
    # = FFT(dk) (no 1/N factor). So the reference is FFT(dk) permuted.
    dk_f_ref = torch.fft.fft(dk_ref.float(), n=N)
    dk_f_permuted_ref = (dk_f_ref.reshape(H, N1, N1)
                           .transpose(-1, -2)
                           .reshape(H, N))

    # Prepare kernel inputs
    dy_real, u_real, f_real, f_imag, tw_real, tw_imag = prepare_dkf_inputs(
        u, dy, B, H, N, N1
    )

    # Move to GPU
    dy_real = dy_real.cuda().contiguous()
    u_real  = u_real.cuda().contiguous()
    f_real  = f_real.cuda().contiguous()
    f_imag  = f_imag.cuda().contiguous()
    tw_real = tw_real.cuda().contiguous()
    tw_imag = tw_imag.cuda().contiguous()

    # Run kernel (returns per-batch partials)
    dk_f_real_out, dk_f_imag_out = fftconv_bwd_dkf(
        dy_real, u_real,
        f_real, f_imag,
        tw_real, tw_imag,
        B, H, N, N1
    )

    # Sum over batch dimension (Python-side reduction)
    dk_f_out = torch.complex(
        dk_f_real_out.float().sum(dim=0),
        dk_f_imag_out.float().sum(dim=0)
    ).reshape(H, N)

    # Compare (both are complex — use complex abs for magnitude of difference)
    dk_f_ref_gpu = dk_f_permuted_ref.cuda()
    diff = (dk_f_out - dk_f_ref_gpu).abs()  # complex abs = magnitude
    max_abs = diff.max().item()
    max_rel = max_abs / dk_f_ref_gpu.abs().max().item()

    print(f"dk_f max abs error: {max_abs:.6e}")
    print(f"dk_f max rel error: {max_rel:.6e}")
    print(f"dk_f PASSED: {max_rel < 0.05}\n")
    return max_rel < 0.05


# ============================================================================
# Autograd cross-validation
# Tests the full chain: forward kernel → loss → our backward kernels
# vs. PyTorch autograd through ref_fftconv.
# This is the gold standard — no hand-derived reference involved.
# ============================================================================

def test_autograd(B, H, N, N1):
    print(f"=== Autograd cross-validation (B={B}, H={H}, N={N}) ===\n")

    torch.manual_seed(42)
    u  = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).float() / H
    k  = (torch.randn((H, N),    dtype=torch.bfloat16, device='cuda')).float() / H

    # --- Autograd reference: differentiate through ref_fftconv ---
    u_ag = u.clone().detach().requires_grad_(True)
    k_ag = k.clone().detach().requires_grad_(True)

    y = ref_fftconv(u_ag, k_ag, N)
    # Use a random loss to exercise all elements
    torch.manual_seed(99)
    loss_weights = torch.randn_like(y)
    loss = (y * loss_weights).sum()
    loss.backward()

    du_autograd = u_ag.grad.detach()            # (B, H, N)
    dk_autograd = k_ag.grad.detach()            # (H, N)

    # --- Our du kernel ---
    # dy for our kernel = d(loss)/d(y) = loss_weights (real)
    dy = loss_weights.detach()

    (dy_real, kf_conj_real, kf_conj_imag,
     f_real, f_imag, finv_real, finv_imag,
     tw_bwd_real, tw_bwd_imag,
     twinv_t_bwd_real, twinv_t_bwd_imag) = prepare_du_inputs(u, k, dy, B, H, N, N1)

    dy_real = dy_real.cuda().contiguous()
    kf_conj_real = kf_conj_real.cuda().contiguous()
    kf_conj_imag = kf_conj_imag.cuda().contiguous()
    f_real = f_real.cuda().contiguous()
    f_imag = f_imag.cuda().contiguous()
    finv_real = finv_real.cuda().contiguous()
    finv_imag = finv_imag.cuda().contiguous()
    tw_bwd_real = tw_bwd_real.cuda().contiguous()
    tw_bwd_imag = tw_bwd_imag.cuda().contiguous()
    twinv_t_bwd_real = twinv_t_bwd_real.cuda().contiguous()
    twinv_t_bwd_imag = twinv_t_bwd_imag.cuda().contiguous()

    du_out = fftconv_bwd(
        dy_real,
        kf_conj_real, kf_conj_imag,
        f_real, f_imag,
        finv_real, finv_imag,
        tw_bwd_real, tw_bwd_imag,
        twinv_t_bwd_real, twinv_t_bwd_imag,
        B, H, N, N1
    ).reshape(B, H, N)

    du_err = (du_out.float() - du_autograd.float().cuda()).abs()
    du_max_rel = du_err.max().item() / du_autograd.float().abs().max().item()

    print(f"du vs autograd  max rel error: {du_max_rel:.6e}")
    du_ok = du_max_rel < 0.05
    print(f"du PASSED: {du_ok}")

    # --- Our dk_f kernel ---
    dy_real_dkf, u_real_dkf, f_real_dkf, f_imag_dkf, tw_real_dkf, tw_imag_dkf = prepare_dkf_inputs(
        u, dy, B, H, N, N1
    )
    dy_real_dkf = dy_real_dkf.cuda().contiguous()
    u_real_dkf  = u_real_dkf.cuda().contiguous()
    f_real_dkf  = f_real_dkf.cuda().contiguous()
    f_imag_dkf  = f_imag_dkf.cuda().contiguous()
    tw_real_dkf = tw_real_dkf.cuda().contiguous()
    tw_imag_dkf = tw_imag_dkf.cuda().contiguous()

    dk_f_real_out, dk_f_imag_out = fftconv_bwd_dkf(
        dy_real_dkf, u_real_dkf,
        f_real_dkf, f_imag_dkf,
        tw_real_dkf, tw_imag_dkf,
        B, H, N, N1
    )

    # Sum over batch, convert to time domain to compare with autograd's dk
    dk_f_out = torch.complex(
        dk_f_real_out.float().sum(dim=0),
        dk_f_imag_out.float().sum(dim=0)
    ).reshape(H, N1, N1)

    # dk_f_out is in permuted frequency domain. Convert back to time domain:
    #   unpermute → IFFT → dk_time
    dk_f_unperm = dk_f_out.transpose(-1, -2).reshape(H, N)
    dk_time = torch.fft.ifft(dk_f_unperm, n=N).real

    dk_err = (dk_time.cuda() - dk_autograd.float().cuda()).abs()
    dk_max_rel = dk_err.max().item() / dk_autograd.float().abs().max().item()

    print(f"\ndk vs autograd  max rel error: {dk_max_rel:.6e}")
    dk_ok = dk_max_rel < 0.05
    print(f"dk PASSED: {dk_ok}\n")

    return du_ok and dk_ok


# ============================================================================
# Run tests
# ============================================================================

all_passed = True

for N in [4096, 1024]:
    B = 2
    H = 4
    N1 = int(np.sqrt(N))

    print(f"FFTConv backward correctness tests")
    print(f"N={N}, B={B}, H={H}, N1={N1}\n")

    du_ok  = test_du(B, H, N, N1)
    dkf_ok = test_dkf(B, H, N, N1)
    ag_ok  = test_autograd(B, H, N, N1)

    print("=" * 50)
    ok = du_ok and dkf_ok and ag_ok
    if ok:
        print(f"ALL TESTS PASSED (N={N})")
    else:
        if not du_ok:  print(f"FAILED: du kernel (N={N})")
        if not dkf_ok: print(f"FAILED: dk_f kernel (N={N})")
        if not ag_ok:  print(f"FAILED: autograd cross-validation (N={N})")
    all_passed = all_passed and ok

    # Stress tests
    print(f"\n=== Stress tests (N={N}) ===\n")
    for b, h in [(1,1), (4,8), (16,16), (32,4)]:
        d_ok = test_du(b, h, N, N1)
        k_ok = test_dkf(b, h, N, N1)
        if not (d_ok and k_ok):
            print(f"FAILED at B={b}, H={h}, N={N}")
            all_passed = False
            break
    else:
        print(f"All stress tests passed (N={N})")

print("\n" + "=" * 50)
if all_passed:
    print("ALL TESTS PASSED (all N values)")
else:
    print("SOME TESTS FAILED")
