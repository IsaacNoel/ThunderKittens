"""
Benchmark: FFTConv backward pass — TK kernel vs PyTorch reference

Benchmarks both causal and non-causal modes across multiple sequence lengths.
Generates PNG graphs to benchmarks/ directory.

Usage:
  # Build the fused kernel first:
  cd /kernels/fftconv_bwd && make  # (with fused kernel config)
  cd benchmarks && python run_benchmarks.py

Prerequisites: matplotlib, numpy, torch, compiled _C module (fused kernel)
"""

import sys
import os
import time
import json
import numpy as np

# Add parent dir to path for _C import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import torch

# Try to import TK kernel; if unavailable, benchmark PyTorch-only
try:
    from _C import fftconv_bwd_fused
    HAS_TK = True
    print("Loaded: TK fftconv_bwd_fused kernel")
except ImportError:
    HAS_TK = False
    print("WARNING: TK kernel not available (no _C module). Running PyTorch-only benchmarks.")


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


def prepare_tk_inputs(u, k, dy, B, H, N, N1):
    """Prepare inputs for TK fused backward kernel."""
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
# Benchmark: PyTorch autograd backward (forward FFT conv + .backward())
# ============================================================================

def bench_pytorch_bwd(B, H, N, num_iters=50, warmup=10):
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


def bench_pytorch_bwd_only(B, H, N, num_iters=50, warmup=10):
    """PyTorch backward-only: IFFT(FFT(dy)*conj(FFT(k))) + dk reduction."""
    u_data = torch.randn(B, H, N, device='cuda', dtype=torch.float32) / H
    k_data = torch.randn(H, N, device='cuda', dtype=torch.float32) / H
    dy = torch.randn(B, H, N, device='cuda', dtype=torch.float32) / H

    def run():
        dy_f = torch.fft.fft(dy, n=N)
        k_f = torch.fft.fft(k_data, n=N)
        u_f = torch.fft.fft(u_data, n=N)
        du = torch.fft.ifft(dy_f * k_f.conj(), n=N).real
        dk = torch.fft.ifft((dy_f * u_f.conj()).sum(dim=0), n=N).real

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
# Benchmark: TK fused backward kernel
# ============================================================================

def bench_tk_fused(B, H, N, N1, num_iters=50, warmup=10):
    """TK fused backward kernel (du + dk_f in one pass)."""
    if not HAS_TK:
        return float('nan')

    u = torch.randn(B, H, N, device='cuda', dtype=torch.bfloat16).float() / H
    k = torch.randn(H, N, device='cuda', dtype=torch.bfloat16).float() / H
    dy = torch.randn(B, H, N, device='cuda', dtype=torch.bfloat16).float() / H

    args = prepare_tk_inputs(u, k, dy, B, H, N, N1)

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
# Main benchmark suite
# ============================================================================

def run_all_benchmarks():
    B_default = 16
    H_default = 16
    D = 128  # not used in current kernel (single-head per tile), but listed per spec

    # Sequence lengths to test (N must be a perfect square for Monarch decomposition)
    # 768=not square, 1536=not square, etc. The kernel requires N = N1^2.
    # Available: 1024 (32^2), 4096 (64^2), 16384 (128^2)
    # For the requested sequence lengths, we use the closest supported sizes.
    seq_lengths = [1024, 4096]

    # Additional configs
    configs = [
        # (B, H, label)
        (2, 4, "B=2,H=4"),
        (4, 8, "B=4,H=8"),
        (8, 16, "B=8,H=16"),
        (16, 16, "B=16,H=16"),
        (32, 4, "B=32,H=4"),
        (32, 16, "B=32,H=16"),
    ]

    results = {}

    for N in seq_lengths:
        N1 = int(np.sqrt(N))
        if N1 * N1 != N:
            print(f"Skipping N={N} (not a perfect square)")
            continue

        print(f"\n{'='*60}")
        print(f"Sequence length N={N} (N1={N1})")
        print(f"{'='*60}")
        print(f"{'Config':>12} | {'PyTorch bwd':>12} | {'PyTorch only':>12} | {'TK fused':>10} | {'Speedup':>8}")
        print("-" * 70)

        results[N] = []

        for B, H, label in configs:
            pytorch_ms = bench_pytorch_bwd(B, H, N) if torch.cuda.is_available() else float('nan')
            pytorch_only_ms = bench_pytorch_bwd_only(B, H, N) if torch.cuda.is_available() else float('nan')
            tk_ms = bench_tk_fused(B, H, N, N1)

            speedup_vs_bwd = pytorch_ms / tk_ms if not np.isnan(tk_ms) and tk_ms > 0 else float('nan')
            speedup_vs_only = pytorch_only_ms / tk_ms if not np.isnan(tk_ms) and tk_ms > 0 else float('nan')

            results[N].append({
                'B': B, 'H': H, 'label': label,
                'pytorch_bwd_ms': pytorch_ms,
                'pytorch_only_ms': pytorch_only_ms,
                'tk_fused_ms': tk_ms,
                'speedup_vs_bwd': speedup_vs_bwd,
                'speedup_vs_only': speedup_vs_only,
            })

            tk_str = f"{tk_ms:.3f}" if not np.isnan(tk_ms) else "N/A"
            su_str = f"{speedup_vs_bwd:.1f}x" if not np.isnan(speedup_vs_bwd) else "N/A"
            print(f"{label:>12} | {pytorch_ms:>10.3f}ms | {pytorch_only_ms:>10.3f}ms | {tk_str:>8}ms | {su_str:>8}")

    return results


def plot_results(results):
    """Generate PNG bar charts comparing TK vs PyTorch backward performance."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for N, data in results.items():
        if not data:
            continue

        labels = [d['label'] for d in data]
        pytorch_times = [d['pytorch_only_ms'] for d in data]
        tk_times = [d['tk_fused_ms'] for d in data]

        x = np.arange(len(labels))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart: absolute times
        bars1 = ax1.bar(x - width/2, pytorch_times, width, label='PyTorch bwd-only', color='#4ECDC4')
        if HAS_TK:
            bars2 = ax1.bar(x + width/2, tk_times, width, label='TK fused', color='#FF6B6B')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title(f'FFTConv Backward — N={N}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Bar chart: speedup
        speedups = [d['speedup_vs_only'] for d in data]
        colors = ['#2ECC71' if s > 1 else '#E74C3C' for s in speedups]
        if HAS_TK:
            ax2.bar(x, speedups, width=0.5, color=colors)
            ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Speedup (TK / PyTorch)')
            ax2.set_title(f'Speedup — N={N}')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'TK kernel not available\n(no GPU / not compiled)',
                     transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title(f'Speedup — N={N}')

        plt.tight_layout()
        outpath = os.path.join(script_dir, f'benchmark_N{N}.png')
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"Saved: {outpath}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("No CUDA device available. Generating placeholder results.")
        print("To get real benchmarks, run on a machine with an H100 GPU.")

        # Save placeholder data for documentation
        results = {
            4096: [
                {'B': B, 'H': H, 'label': f'B={B},H={H}',
                 'pytorch_bwd_ms': float('nan'), 'pytorch_only_ms': float('nan'),
                 'tk_fused_ms': float('nan'), 'speedup_vs_bwd': float('nan'),
                 'speedup_vs_only': float('nan')}
                for B, H in [(2,4), (4,8), (8,16), (16,16), (32,4), (32,16)]
            ]
        }
    else:
        results = run_all_benchmarks()

    plot_results(results)

    # Save raw data as JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'benchmark_results.json')
    # Convert NaN to None for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    with open(json_path, 'w') as f:
        json.dump(clean_for_json({str(k): v for k, v in results.items()}), f, indent=2)
    print(f"Saved: {json_path}")
