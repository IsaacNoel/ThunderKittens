import torch
import numpy as np
from _C import fftconv_bwd_fused
# For the cross-kernel comparison test, we also need the separate kernels.
# Build the separate version first (make), save the .so, then build fused.
# If the separate kernel .so isn't available, cross-kernel tests are skipped.
try:
    from _C_separate import fftconv_bwd, fftconv_bwd_dkf
    HAS_SEPARATE = True
except ImportError:
    HAS_SEPARATE = False


def ref_fftconv_bwd(u, k, dy, N):
    """Closed-form backward. Source of truth."""
    L = u.shape[-1]
    dy_f = torch.fft.fft(dy.float(), n=N)
    k_f  = torch.fft.fft(k.float(), n=N)
    u_f  = torch.fft.fft(u.float(), n=N)
    du = torch.fft.ifft(dy_f * k_f.conj(), n=N).real[..., :L]
    dk = torch.fft.ifft((dy_f * u_f.conj()).sum(dim=0), n=N).real[..., :L]
    return du, dk

def ref_fftconv(u, k, N):
    L = u.shape[-1]
    u_f = torch.fft.fft(u.float(), n=N)
    k_f = torch.fft.fft(k.float(), n=N)
    y = torch.fft.ifft(u_f * k_f, n=N).real[..., :L]
    return y

def fft_matrix(N):
    n = torch.arange(N); k = n.view(-1, 1)
    return torch.exp(-2j * torch.pi * n * k / N)

def ifft_matrix(N):
    n = torch.arange(N); k = n.view(-1, 1)
    return torch.exp(2j * torch.pi * n * k / N)

def compute_twiddle_factors_fft(n, m):
    n_a = torch.arange(n).view(-1, 1); m_a = torch.arange(m)
    return torch.exp(-2j * torch.pi * n_a * m_a / (n * m))

def compute_twiddle_factors_ifft(n, m):
    n_a = torch.arange(n).view(-1, 1); m_a = torch.arange(m)
    return torch.exp(2j * torch.pi * n_a * m_a / (n * m))


def prepare_fused_inputs(u, k, dy, B, H, N, N1):
    """Prepare all inputs for the fused backward kernel."""
    f_mat = fft_matrix(N1)
    finv_mat = ifft_matrix(N1)

    # tw: unnormalized twiddle (for FFT of dy and u in dk_f computation,
    #     and for du backward steps 1-3 since conj(twinv_t) = tw_unnorm)
    tw = compute_twiddle_factors_fft(N1, N1)

    # twinv_bwd: conj(tw_fwd) where tw_fwd = tw / N
    # = conj(tw / N) = twinv / N
    tw_fwd = tw / N
    twinv_bwd = tw_fwd.conj()

    # conj(kf)
    k_f = torch.fft.fft(k.float(), n=N)
    k_fT = k_f.reshape(H, N1, N1).transpose(-1, -2)
    kf_conj = k_fT.conj()

    to_bf16 = lambda t: t.to(torch.bfloat16).contiguous()

    return {
        'dy':             to_bf16(dy.reshape(B, H, N1, N1)),
        'u':              to_bf16(u.reshape(B, H, N1, N1)),
        'kf_conj_real':   to_bf16(kf_conj.real),
        'kf_conj_imag':   to_bf16(kf_conj.imag),
        'f_real':         to_bf16(f_mat.real),
        'f_imag':         to_bf16(f_mat.imag),
        'finv_real':      to_bf16(finv_mat.real),
        'finv_imag':      to_bf16(finv_mat.imag),
        'tw_real':        to_bf16(tw.real),
        'tw_imag':        to_bf16(tw.imag),
        'twinv_bwd_real': to_bf16(twinv_bwd.real),
        'twinv_bwd_imag': to_bf16(twinv_bwd.imag),
    }


def test_fused(B, H, N, N1):
    print(f"=== Fused kernel test (B={B}, H={H}, N={N}) ===\n")

    torch.manual_seed(42)
    u  = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).float() / H
    k  = (torch.randn((H, N),    dtype=torch.bfloat16, device='cuda')).float() / H
    dy = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).float() / H

    # Reference
    du_ref, dk_ref = ref_fftconv_bwd(u, k, dy, N)
    du_ref = du_ref.to(torch.bfloat16).cuda()

    # dk_f reference in permuted frequency domain (unnormalized)
    dk_f_ref = torch.fft.fft(dk_ref.float(), n=N)
    dk_f_permuted_ref = dk_f_ref.reshape(H, N1, N1).transpose(-1, -2).reshape(H, N)

    # Prepare inputs
    inputs = prepare_fused_inputs(u, k, dy, B, H, N, N1)
    for key in inputs:
        inputs[key] = inputs[key].cuda().contiguous()

    # Run fused kernel
    du_out, dk_f_r_out, dk_f_i_out = fftconv_bwd_fused(
        inputs['dy'], inputs['u'],
        inputs['kf_conj_real'], inputs['kf_conj_imag'],
        inputs['f_real'], inputs['f_imag'],
        inputs['finv_real'], inputs['finv_imag'],
        inputs['tw_real'], inputs['tw_imag'],
        inputs['twinv_bwd_real'], inputs['twinv_bwd_imag'],
        B, H, N, N1
    )

    # Check du
    du_out_flat = du_out.reshape(B, H, N)
    du_diff = (du_out_flat.float() - du_ref.float()).abs()
    du_max_rel = du_diff.max().item() / du_ref.float().abs().max().item()
    print(f"du max rel error: {du_max_rel:.6e}")
    du_ok = du_max_rel < 0.05
    print(f"du PASSED: {du_ok}")

    # Check dk_f
    dk_f_out = torch.complex(dk_f_r_out.float(), dk_f_i_out.float()).reshape(H, N)
    dk_f_diff = (dk_f_out - dk_f_permuted_ref.cuda()).abs()
    dk_f_max_rel = dk_f_diff.max().item() / dk_f_permuted_ref.abs().max().item()
    print(f"dk_f max rel error: {dk_f_max_rel:.6e}")
    dk_f_ok = dk_f_max_rel < 0.05
    print(f"dk_f PASSED: {dk_f_ok}")

    return du_ok and dk_f_ok


def test_autograd_fused(B, H, N, N1):
    print(f"\n=== Autograd cross-validation (B={B}, H={H}, N={N}) ===\n")

    torch.manual_seed(42)
    u = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).float() / H
    k = (torch.randn((H, N),    dtype=torch.bfloat16, device='cuda')).float() / H

    u_ag = u.clone().detach().requires_grad_(True)
    k_ag = k.clone().detach().requires_grad_(True)
    y = ref_fftconv(u_ag, k_ag, N)
    torch.manual_seed(99)
    loss_weights = torch.randn_like(y)
    loss = (y * loss_weights).sum()
    loss.backward()
    du_autograd = u_ag.grad.detach()
    dk_autograd = k_ag.grad.detach()

    dy = loss_weights.detach()
    inputs = prepare_fused_inputs(u, k, dy, B, H, N, N1)
    for key in inputs:
        inputs[key] = inputs[key].cuda().contiguous()

    du_out, dk_f_r_out, dk_f_i_out = fftconv_bwd_fused(
        inputs['dy'], inputs['u'],
        inputs['kf_conj_real'], inputs['kf_conj_imag'],
        inputs['f_real'], inputs['f_imag'],
        inputs['finv_real'], inputs['finv_imag'],
        inputs['tw_real'], inputs['tw_imag'],
        inputs['twinv_bwd_real'], inputs['twinv_bwd_imag'],
        B, H, N, N1
    )

    # du check
    du_err = (du_out.reshape(B,H,N).float() - du_autograd.float().cuda()).abs()
    du_max_rel = du_err.max().item() / du_autograd.float().abs().max().item()
    print(f"du vs autograd  max rel error: {du_max_rel:.6e}")
    du_ok = du_max_rel < 0.05
    print(f"du PASSED: {du_ok}")

    # dk check — convert from permuted freq domain to time domain
    dk_f_out = torch.complex(dk_f_r_out.float(), dk_f_i_out.float()).reshape(H, N1, N1)
    dk_f_unperm = dk_f_out.transpose(-1, -2).reshape(H, N)
    dk_time = torch.fft.ifft(dk_f_unperm, n=N).real
    dk_err = (dk_time.cuda() - dk_autograd.float().cuda()).abs()
    dk_max_rel = dk_err.max().item() / dk_autograd.float().abs().max().item()
    print(f"dk vs autograd  max rel error: {dk_max_rel:.6e}")
    dk_ok = dk_max_rel < 0.05
    print(f"dk PASSED: {dk_ok}")

    return du_ok and dk_ok


# ============================================================================
# Fused vs separate kernel comparison
# Both kernels should produce identical (or near-identical) results since
# they perform the same operations — any difference is from smem aliasing
# or accumulation order, not math.
# ============================================================================

def prepare_separate_du_inputs(u, k, dy, B, H, N, N1):
    """Prepare inputs for the separate du kernel (conjugated twiddles)."""
    f_mat = fft_matrix(N1)
    finv_mat = ifft_matrix(N1)
    tw_fwd = compute_twiddle_factors_fft(N1, N1) / N
    twinv = compute_twiddle_factors_ifft(N1, N1)

    tw_bwd = twinv.conj()           # conj(twinv_t) for du step 2
    twinv_t_bwd = tw_fwd.conj()     # conj(tw) for du step 6

    k_f = torch.fft.fft(k.float(), n=N)
    k_fT = k_f.reshape(H, N1, N1).transpose(-1, -2)
    kf_conj = k_fT.conj()

    to_bf16 = lambda t: t.to(torch.bfloat16).contiguous().cuda()
    return {
        'dy': to_bf16(dy.reshape(B, H, N1, N1)),
        'kfc_r': to_bf16(kf_conj.real), 'kfc_i': to_bf16(kf_conj.imag),
        'f_r': to_bf16(f_mat.real), 'f_i': to_bf16(f_mat.imag),
        'fi_r': to_bf16(finv_mat.real), 'fi_i': to_bf16(finv_mat.imag),
        'twb_r': to_bf16(tw_bwd.real), 'twb_i': to_bf16(tw_bwd.imag),
        'tib_r': to_bf16(twinv_t_bwd.real), 'tib_i': to_bf16(twinv_t_bwd.imag),
    }

def prepare_separate_dkf_inputs(u, dy, B, H, N, N1):
    """Prepare inputs for the separate dk_f kernel (unnormalized twiddles)."""
    f_mat = fft_matrix(N1)
    tw = compute_twiddle_factors_fft(N1, N1)  # no /N
    to_bf16 = lambda t: t.to(torch.bfloat16).contiguous().cuda()
    return {
        'dy': to_bf16(dy.reshape(B, H, N1, N1)),
        'u': to_bf16(u.reshape(B, H, N1, N1)),
        'f_r': to_bf16(f_mat.real), 'f_i': to_bf16(f_mat.imag),
        'tw_r': to_bf16(tw.real), 'tw_i': to_bf16(tw.imag),
    }


def test_fused_vs_separate(B, H, N, N1):
    print(f"\n=== Fused vs Separate kernels (B={B}, H={H}, N={N}) ===\n")

    torch.manual_seed(42)
    u  = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).float() / H
    k  = (torch.randn((H, N),    dtype=torch.bfloat16, device='cuda')).float() / H
    dy = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).float() / H

    # ---- Run separate kernels ----
    sep_du = prepare_separate_du_inputs(u, k, dy, B, H, N, N1)
    du_separate = fftconv_bwd(
        sep_du['dy'], sep_du['kfc_r'], sep_du['kfc_i'],
        sep_du['f_r'], sep_du['f_i'], sep_du['fi_r'], sep_du['fi_i'],
        sep_du['twb_r'], sep_du['twb_i'], sep_du['tib_r'], sep_du['tib_i'],
        B, H, N, N1
    ).reshape(B, H, N)

    sep_dkf = prepare_separate_dkf_inputs(u, dy, B, H, N, N1)
    dkf_r_sep, dkf_i_sep = fftconv_bwd_dkf(
        sep_dkf['dy'], sep_dkf['u'],
        sep_dkf['f_r'], sep_dkf['f_i'],
        sep_dkf['tw_r'], sep_dkf['tw_i'],
        B, H, N, N1
    )
    # Sum over batch for separate kernel
    dkf_separate = torch.complex(
        dkf_r_sep.float().sum(dim=0),
        dkf_i_sep.float().sum(dim=0)
    ).reshape(H, N)

    # ---- Run fused kernel ----
    fused_inputs = prepare_fused_inputs(u, k, dy, B, H, N, N1)
    for key in fused_inputs:
        fused_inputs[key] = fused_inputs[key].cuda().contiguous()

    du_fused, dkf_r_fused, dkf_i_fused = fftconv_bwd_fused(
        fused_inputs['dy'], fused_inputs['u'],
        fused_inputs['kf_conj_real'], fused_inputs['kf_conj_imag'],
        fused_inputs['f_real'], fused_inputs['f_imag'],
        fused_inputs['finv_real'], fused_inputs['finv_imag'],
        fused_inputs['tw_real'], fused_inputs['tw_imag'],
        fused_inputs['twinv_bwd_real'], fused_inputs['twinv_bwd_imag'],
        B, H, N, N1
    )
    du_fused = du_fused.reshape(B, H, N)
    dkf_fused = torch.complex(dkf_r_fused.float(), dkf_i_fused.float()).reshape(H, N)

    # ---- Compare ----
    du_diff = (du_fused.float() - du_separate.float()).abs()
    du_max_abs = du_diff.max().item()
    du_max_rel = du_max_abs / du_separate.float().abs().max().item()
    print(f"du  fused vs separate  max abs: {du_max_abs:.6e}  max rel: {du_max_rel:.6e}")
    du_ok = du_max_rel < 0.01  # tighter tolerance — same math, should nearly match
    print(f"du PASSED: {du_ok}")

    dkf_diff = (dkf_fused - dkf_separate.cuda()).abs()
    dkf_max_abs = dkf_diff.max().item()
    dkf_max_rel = dkf_max_abs / dkf_separate.abs().max().item()
    print(f"dk_f fused vs separate  max abs: {dkf_max_abs:.6e}  max rel: {dkf_max_rel:.6e}")
    dkf_ok = dkf_max_rel < 0.01
    print(f"dk_f PASSED: {dkf_ok}")

    return du_ok and dkf_ok


# ============================================================================
# Run
# ============================================================================

N = 4096
B = 2
H = 4
N1 = int(np.sqrt(N))

print(f"Fused FFTConv backward correctness tests")
print(f"N={N}, B={B}, H={H}, N1={N1}\n")

fused_ok = test_fused(B, H, N, N1)
ag_ok = test_autograd_fused(B, H, N, N1)
cross_ok = test_fused_vs_separate(B, H, N, N1) if HAS_SEPARATE else True
if not HAS_SEPARATE:
    print("\n(Skipped fused vs separate test — _C_separate not found)")
    print("To enable: build separate kernel, then: cp _C*.so _C_separate*.so")

print("\n" + "=" * 50)
all_ok = fused_ok and ag_ok and cross_ok
if all_ok:
    print("ALL TESTS PASSED")
else:
    if not fused_ok: print("FAILED: fused kernel")
    if not ag_ok:    print("FAILED: autograd cross-validation")
    if not cross_ok: print("FAILED: fused vs separate comparison")

# Stress tests
print("\n=== Stress tests ===\n")
for b, h in [(1,1), (4,8), (16,16), (32,4)]:
    ok = test_fused(b, h, N, N1)
    if not ok:
        print(f"FAILED at B={b}, H={h}")
        break
else:
    print("All stress tests passed")
