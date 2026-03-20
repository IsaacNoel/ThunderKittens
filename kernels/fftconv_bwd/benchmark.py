"""
Benchmark: FFTConv backward pass

Compares TK kernel against PyTorch autograd baseline (torch.fft forward + .backward())

Usage:
  python benchmark.py
"""

import torch
import numpy as np

from _C import fftconv_bwd_fused

print("Loaded: TK fftconv_bwd kernel")


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

to_bf16_cuda = lambda t: t.to(torch.bfloat16).contiguous().cuda()


def prepare_inputs(u, k, dy, B, H, N, N1):
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

def bench_pytorch(B, H, N, num_iters=50, warmup=10):
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


def bench_tk(B, H, N, N1, num_iters=50, warmup=10):
    """TK fftconv backward kernel."""
    u = torch.randn(B, H, N, device='cuda', dtype=torch.bfloat16).float() / H
    k = torch.randn(H, N, device='cuda', dtype=torch.bfloat16).float() / H
    dy = torch.randn(B, H, N, device='cuda', dtype=torch.bfloat16).float() / H

    args = prepare_inputs(u, k, dy, B, H, N, N1)

    def run():
        fftconv_bwd_fused(*args, B, H, N, N1)

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
print(f"Median of 50 iterations, in milliseconds.\n")
print(f"{'B':>4} {'H':>4} | {'PyTorch':>10} | {'TK':>10} | {'Speedup':>8}")
print("-" * 50)

for B, H in configs:
    pytorch_ms = bench_pytorch(B, H, N)
    tk_ms = bench_tk(B, H, N, N1)
    speedup = pytorch_ms / tk_ms
    print(f"{B:>4} {H:>4} | {pytorch_ms:>10.3f} | {tk_ms:>10.3f} | {speedup:>7.1f}x")
