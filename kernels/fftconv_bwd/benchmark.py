"""
Benchmark: FFTConv backward pass

Compares:
  1. PyTorch autograd baseline (torch.fft forward + .backward())
  2. TK separate kernels (du + dk_f as two launches)
  3. TK fused kernel (du + dk_f in one launch)

Usage:
  # With separate kernels built as _C:
  python benchmark.py --mode separate

  # With fused kernel built as _C:
  python benchmark.py --mode fused

  # Both (requires _C = fused, _C_separate = separate):
  python benchmark.py --mode all
"""

import argparse
import torch
import numpy as np
import math
import time

# ---- Import kernels ----
HAS_FUSED = False
HAS_SEPARATE = False
HAS_FLASHFFT = False

try:
    from _C import fftconv_bwd_fused
    HAS_FUSED = True
    print("Loaded: TK fused kernel")
except ImportError:
    pass

try:
    from _C_separate import fftconv_bwd, fftconv_bwd_dkf
    HAS_SEPARATE = True
    print("Loaded: TK separate kernels (from _C_separate)")
except ImportError:
    try:
        from _C import fftconv_bwd, fftconv_bwd_dkf
        HAS_SEPARATE = True
        print("Loaded: TK separate kernels (from _C)")
    except ImportError:
        pass

try:
    from flashfftconv import FlashFFTConv
    HAS_FLASHFFT = True
    print("Loaded: FlashFFTConv")
except ImportError:
    print("FlashFFTConv not available (pip install flashfftconv to enable)")


# ============================================================================
# Helpers
# ============================================================================

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

to_bf16_cuda = lambda t: t.to(torch.bfloat16).contiguous().cuda()


def prepare_du_inputs(u, k, dy, B, H, N, N1):
    f_mat = fft_matrix(N1); finv_mat = ifft_matrix(N1)
    tw_fwd = compute_twiddle_factors_fft(N1, N1) / N
    twinv = compute_twiddle_factors_ifft(N1, N1)
    k_f = torch.fft.fft(k.float(), n=N)
    k_fT = k_f.reshape(H, N1, N1).transpose(-1, -2)
    return (
        to_bf16_cuda(dy.reshape(B, H, N1, N1)),
        to_bf16_cuda(k_fT.conj().real), to_bf16_cuda(k_fT.conj().imag),
        to_bf16_cuda(f_mat.real), to_bf16_cuda(f_mat.imag),
        to_bf16_cuda(finv_mat.real), to_bf16_cuda(finv_mat.imag),
        to_bf16_cuda(twinv.conj().real), to_bf16_cuda(twinv.conj().imag),
        to_bf16_cuda(tw_fwd.conj().real), to_bf16_cuda(tw_fwd.conj().imag),
    )

def prepare_dkf_inputs(u, dy, B, H, N, N1):
    f_mat = fft_matrix(N1)
    tw = compute_twiddle_factors_fft(N1, N1)
    return (
        to_bf16_cuda(dy.reshape(B, H, N1, N1)),
        to_bf16_cuda(u.reshape(B, H, N1, N1)),
        to_bf16_cuda(f_mat.real), to_bf16_cuda(f_mat.imag),
        to_bf16_cuda(tw.real), to_bf16_cuda(tw.imag),
    )

def prepare_fused_inputs(u, k, dy, B, H, N, N1):
    f_mat = fft_matrix(N1); finv_mat = ifft_matrix(N1)
    tw = compute_twiddle_factors_fft(N1, N1)
    tw_fwd = tw / N
    twinv_bwd = tw_fwd.conj()
    k_f = torch.fft.fft(k.float(), n=N)
    k_fT = k_f.reshape(H, N1, N1).transpose(-1, -2)
    kf_conj = k_fT.conj()
    return (
        to_bf16_cuda(dy.reshape(B, H, N1, N1)),
        to_bf16_cuda(u.reshape(B, H, N1, N1)),
        to_bf16_cuda(kf_conj.real), to_bf16_cuda(kf_conj.imag),
        to_bf16_cuda(f_mat.real), to_bf16_cuda(f_mat.imag),
        to_bf16_cuda(finv_mat.real), to_bf16_cuda(finv_mat.imag),
        to_bf16_cuda(tw.real), to_bf16_cuda(tw.imag),
        to_bf16_cuda(twinv_bwd.real), to_bf16_cuda(twinv_bwd.imag),
    )


# ============================================================================
# Benchmark functions
# ============================================================================

def bench_pytorch_autograd(B, H, N, num_iters=50, warmup=10):
    """PyTorch FFT conv forward + backward via autograd."""
    u_data = torch.randn(B, H, N, device='cuda', dtype=torch.float32) / H
    k_data = torch.randn(H, N, device='cuda', dtype=torch.float32) / H
    dy = torch.randn(B, H, N, device='cuda', dtype=torch.float32) / H

    def run():
        u = u_data.detach().requires_grad_(True)
        k = k_data.detach().requires_grad_(True)
        u_f = torch.fft.fft(u, n=N)
        k_f = torch.fft.fft(k, n=N)
        y = torch.fft.ifft(u_f * k_f, n=N).real
        loss = (y * dy).sum()
        loss.backward()

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        run()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return np.median(times)


def bench_separate(B, H, N, N1, num_iters=50, warmup=10):
    """TK separate kernels: du + dk_f as two launches."""
    u = torch.randn(B, H, N, device='cuda', dtype=torch.bfloat16).float() / H
    k = torch.randn(H, N, device='cuda', dtype=torch.bfloat16).float() / H
    dy = torch.randn(B, H, N, device='cuda', dtype=torch.bfloat16).float() / H

    du_args = prepare_du_inputs(u, k, dy, B, H, N, N1)
    dkf_args = prepare_dkf_inputs(u, dy, B, H, N, N1)

    def run():
        fftconv_bwd(*du_args, B, H, N, N1)
        dkr, dki = fftconv_bwd_dkf(*dkf_args, B, H, N, N1)
        # Batch reduction (would be in Python)
        _ = dkr.sum(dim=0)
        _ = dki.sum(dim=0)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        run()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return np.median(times)


def bench_flashfftconv(B, H, N, num_iters=50, warmup=10):
    """FlashFFTConv forward + backward."""
    u = torch.randn(B, H, N, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(H, N, device='cuda', dtype=torch.float32)

    conv = FlashFFTConv(N, dtype=u.dtype).to(u.device)
    dy = torch.randn(B, H, N, device='cuda', dtype=u.dtype)

    def run():
        u_in = u.detach().requires_grad_(True)
        y = conv(u_in, k)
        loss = (y * dy).sum()
        loss.backward()

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        run()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return np.median(times)


def bench_fused(B, H, N, N1, num_iters=50, warmup=10):
    """TK fused kernel: du + dk_f in one launch."""
    u = torch.randn(B, H, N, device='cuda', dtype=torch.bfloat16).float() / H
    k = torch.randn(H, N, device='cuda', dtype=torch.bfloat16).float() / H
    dy = torch.randn(B, H, N, device='cuda', dtype=torch.bfloat16).float() / H

    fused_args = prepare_fused_inputs(u, k, dy, B, H, N, N1)

    def run():
        fftconv_bwd_fused(*fused_args, B, H, N, N1)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        run()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return np.median(times)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["separate", "fused", "all"], default="all")
    args = parser.parse_args()

    N = 4096
    N1 = int(np.sqrt(N))

    configs = [
        (2,   4),
        (4,   8),
        (8,  16),
        (16, 16),
        (32,  4),
        (32, 16),
    ]

    print(f"\nFFTConv Backward Benchmark (N={N})")
    print(f"All times are median of 50 iterations, in milliseconds.")
    print(f"Speedup is relative to PyTorch autograd baseline.\n")

    # Header
    cols = [f"{'B':>4} {'H':>4}", f"{'PyTorch':>10}"]
    if HAS_FLASHFFT:
        cols.append(f"{'FlashFFT':>10} {'spd':>6}")
    if args.mode in ("separate", "all") and HAS_SEPARATE:
        cols.append(f"{'Separate':>10} {'spd':>6}")
    if args.mode in ("fused", "all") and HAS_FUSED:
        cols.append(f"{'Fused':>10} {'spd':>6}")
    print(" | ".join(cols))
    print("-" * (len(" | ".join(cols)) + 5))

    for B, H in configs:
        pytorch_ms = bench_pytorch_autograd(B, H, N)
        row = [f"{B:>4} {H:>4}", f"{pytorch_ms:>10.3f}"]

        if HAS_FLASHFFT:
            try:
                flash_ms = bench_flashfftconv(B, H, N)
                row.append(f"{flash_ms:>10.3f} {pytorch_ms/flash_ms:>5.1f}x")
            except Exception as e:
                row.append(f"{'err':>10} {'':>6}")
                print(f"  FlashFFTConv error: {e}", flush=True)

        if args.mode in ("separate", "all") and HAS_SEPARATE:
            sep_ms = bench_separate(B, H, N, N1)
            row.append(f"{sep_ms:>10.3f} {pytorch_ms/sep_ms:>5.1f}x")

        if args.mode in ("fused", "all") and HAS_FUSED:
            fused_ms = bench_fused(B, H, N, N1)
            row.append(f"{fused_ms:>10.3f} {pytorch_ms/fused_ms:>5.1f}x")

        print(" | ".join(row))
