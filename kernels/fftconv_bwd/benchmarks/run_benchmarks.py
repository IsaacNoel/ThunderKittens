"""
Benchmark: FFTConv backward pass — TK kernels vs PyTorch reference

Compares two implementations:
  1. PyTorch autograd backward (baseline)
  2. TK two-kernel: separate fftconv_bwd (du) + fftconv_bwd_dkf (dk_f)

Usage:
    cd kernels/fftconv_bwd && make
    python benchmarks/run_benchmarks.py
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

# ── Import TK kernels (graceful) ─────────────────────────────────────────────

try:
    from _C import fftconv_bwd, fftconv_bwd_dkf
    HAS_TK_SEPARATE = True
    print("Loaded: TK two-kernel (fftconv_bwd + fftconv_bwd_dkf)")
except ImportError:
    HAS_TK_SEPARATE = False
    print("INFO: TK two-kernel not in _C (build with 'make' to enable)")


# ── Matrix / twiddle helpers ──────────────────────────────────────────────────

KERNEL_TILE = 64

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

def to_tile_block_diag(mat):
    """Lift N1xN1 matrix to 64x64 for left/right matrix multiplies."""
    if mat.shape[-1] == KERNEL_TILE:
        return mat
    return torch.block_diag(mat, mat)

def to_tile_pointwise(mat):
    """Lift N1xN1 (or HxN1xN1) matrix to 64x64 (or Hx64x64) for pointwise ops."""
    if mat.shape[-1] == KERNEL_TILE:
        return mat
    return mat.repeat(2, 2) if mat.dim() == 2 else mat.repeat(1, 2, 2)

bf16_cuda = lambda t: t.to(torch.bfloat16).contiguous().cuda()


def prepare_du_inputs(u, k, dy, B, H, N, N1):
    """Inputs for the du backward kernel."""
    f_mat    = to_tile_block_diag(fft_matrix(N1))
    finv_mat = to_tile_block_diag(ifft_matrix(N1))
    tw       = to_tile_pointwise(compute_twiddle_factors_fft(N1, N1) / N)
    twinv    = to_tile_pointwise(compute_twiddle_factors_ifft(N1, N1))
    tw_bwd        = twinv.conj()
    twinv_t_bwd   = tw.conj()
    k_f  = torch.fft.fft(k.float(), n=N)
    k_fT = k_f.reshape(H, N1, N1).transpose(-1, -2)
    kf_conj = to_tile_pointwise(k_fT.conj())
    dy_r = bf16_cuda(dy.reshape(B, H, N1, N1))
    return (
        dy_r,
        bf16_cuda(kf_conj.real), bf16_cuda(kf_conj.imag),
        bf16_cuda(f_mat.real),   bf16_cuda(f_mat.imag),
        bf16_cuda(finv_mat.real),bf16_cuda(finv_mat.imag),
        bf16_cuda(tw_bwd.real),  bf16_cuda(tw_bwd.imag),
        bf16_cuda(twinv_t_bwd.real), bf16_cuda(twinv_t_bwd.imag),
    )

def prepare_dkf_inputs(u, dy, B, H, N, N1):
    """Inputs for the dk_f backward kernel."""
    f_mat = to_tile_block_diag(fft_matrix(N1))
    tw    = to_tile_pointwise(compute_twiddle_factors_fft(N1, N1))
    return (
        bf16_cuda(dy.reshape(B, H, N1, N1)),
        bf16_cuda(u.reshape(B, H, N1, N1)),
        bf16_cuda(f_mat.real), bf16_cuda(f_mat.imag),
        bf16_cuda(tw.real),    bf16_cuda(tw.imag),
    )


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def _time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(iters)]
    for s, e in evs:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return np.median([s.elapsed_time(e) for s, e in evs])


def bench_pytorch(B, H, N):
    u = torch.randn(B, H, N, device='cuda', dtype=torch.float32) / H
    k = torch.randn(H, N,    device='cuda', dtype=torch.float32) / H
    dy = torch.randn(B, H, N, device='cuda', dtype=torch.float32) / H
    def run():
        dy_f = torch.fft.fft(dy, n=N)
        k_f  = torch.fft.fft(k,  n=N)
        u_f  = torch.fft.fft(u,  n=N)
        _ = torch.fft.ifft(dy_f * k_f.conj(), n=N).real
        _ = torch.fft.ifft((dy_f * u_f.conj()).sum(dim=0), n=N).real
    return _time_fn(run)


def bench_tk(B, H, N, N1):
    if not HAS_TK_SEPARATE:
        return float('nan')
    u  = torch.randn(B, H, N, device='cuda').float() / H
    k  = torch.randn(H, N,    device='cuda').float() / H
    dy = torch.randn(B, H, N, device='cuda').float() / H
    du_args  = prepare_du_inputs(u, k, dy, B, H, N, N1)
    dkf_args = prepare_dkf_inputs(u, dy, B, H, N, N1)
    def run():
        fftconv_bwd(*du_args, B, H, N, N1)
        fftconv_bwd_dkf(*dkf_args, B, H, N, N1)
    return _time_fn(run)


# ── Main suite ────────────────────────────────────────────────────────────────

CONFIGS = [
    (2,  4,  "B=2,H=4"),
    (2,  16, "B=2,H=16"),
    (4,  8,  "B=4,H=8"),
    (4,  64, "B=4,H=64"),
    (8,  16, "B=8,H=16"),
    (8,  32, "B=8,H=32"),
    (16, 16, "B=16,H=16"),
    (16, 32, "B=16,H=32"),
    (32, 4,  "B=32,H=4"),
    (32, 16, "B=32,H=16"),
    (32, 32, "B=32,H=32"),
    (64, 4,  "B=64,H=4"),
    (64, 16, "B=64,H=16"),
    (64, 32, "B=64,H=32"),
]

SEQ_LENGTHS = [1024, 4096]  # N must be N1^2 for Monarch decomposition


def run_all():
    results = {}
    for N in SEQ_LENGTHS:
        N1 = int(N ** 0.5)
        assert N1 * N1 == N, f"N={N} is not a perfect square"
        print(f"\n{'='*60}\nN={N} (N1={N1})\n{'='*60}")
        hdr = f"{'Config':>12} | {'PyTorch':>10} | {'TK':>10} | {'Speedup':>8}"
        print(hdr); print("-" * len(hdr))
        results[N] = []
        for B, H, label in CONFIGS:
            pt = bench_pytorch(B, H, N)
            tk = bench_tk(B, H, N, N1)
            sp = pt / tk if not np.isnan(tk) and tk > 0 else float('nan')
            fmt = lambda v: f"{v:.3f}ms" if not np.isnan(v) else "   N/A  "
            fmx = lambda v: f"{v:.2f}x"  if not np.isnan(v) else "  N/A"
            print(f"{label:>12} | {fmt(pt):>10} | {fmt(tk):>10} | {fmx(sp):>8}")
            results[N].append(dict(label=label, B=B, H=H,
                                   pytorch_ms=pt, tk_ms=tk, speedup=sp))
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots"); return

    out_dir = os.path.dirname(os.path.abspath(__file__))

    for N, rows in results.items():
        labels = [r['label'] for r in rows]
        x = np.arange(len(labels)); w = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'FFTConv Backward — N={N}', fontsize=13, fontweight='bold')

        # Absolute times
        ax1.bar(x - w/2, [r['pytorch_ms'] for r in rows], w, label='PyTorch', color='#4ECDC4')
        ax1.bar(x + w/2, [r['tk_ms']      for r in rows], w, label='TK',      color='#FF6B6B')
        ax1.set_ylabel('Latency (ms)'); ax1.set_title('Absolute latency')
        ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha='right')
        ax1.legend(); ax1.grid(axis='y', alpha=0.3)

        # Speedup over PyTorch
        ax2.bar(x, [r['speedup'] for r in rows], w*2, color='#FF6B6B', label='TK speedup')
        ax2.axhline(1, color='gray', linestyle='--', alpha=0.5, label='PyTorch baseline')
        ax2.set_ylabel('Speedup vs PyTorch'); ax2.set_title('Speedup')
        ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha='right')
        ax2.legend(); ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out_dir, f'benchmark_N{N}.png')
        plt.savefig(path, dpi=150); plt.close()
        print(f"Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("No CUDA device — cannot benchmark. Run on an H100.")
        sys.exit(0)

    results = run_all()
    plot(results)

    # Save JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    def _clean(o):
        if isinstance(o, float) and np.isnan(o): return None
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list): return [_clean(v) for v in o]
        return o
    with open(os.path.join(out_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(_clean({str(k): v for k, v in results.items()}), f, indent=2)
    print(f"Saved: {out_dir}/benchmark_results.json")
