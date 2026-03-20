import torch
import sys

torch.set_grad_enabled(True)

N = 32 * 32 * 32
B = 1
H = 1

TESTNAME = sys.argv[1] if len(sys.argv) > 1 else 'randn'

if TESTNAME == 'ones':
    u = torch.ones((B, H, N), dtype=torch.float32, device='cpu', requires_grad=True)
    k = torch.ones((H, N), dtype=torch.float32, device='cpu', requires_grad=True)
    dy = torch.ones((B, H, N), dtype=torch.float32, device='cpu')
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    u = torch.randn((B, H, N), dtype=torch.float32, device='cpu', requires_grad=True)
    k = torch.randn((H, N), dtype=torch.float32, device='cpu', requires_grad=True)
    dy = torch.randn((B, H, N), dtype=torch.float32, device='cpu')
else:
    print('Invalid test name')
    sys.exit(1)


def ref_fftconv_fwd(u, k, N):
    """Forward pass: convolution via FFT."""
    L = u.shape[-1]
    u_f = torch.fft.fft(u, n=N)
    k_f = torch.fft.fft(k, n=N)
    y_f = u_f * k_f
    y = torch.fft.ifft(y_f, n=N).real[..., :L]
    return y


def ref_fftconv_bwd(u, k, dy, N):
    """
    Backward pass for FFT convolution. Self-contained — recomputes everything
    from the original inputs rather than relying on saved forward-pass state.

    Args:
        u:  (B, H, N) original input signal
        k:  (H, N)    original filter
        dy: (B, H, N) upstream gradient (dL/dy)
        N:  FFT size

    Returns:
        du: (B, H, N) gradient w.r.t. u
        dk: (H, N)    gradient w.r.t. k
    """
    L = u.shape[-1]

    # Recompute FFTs of original inputs (no dependence on forward pass)
    dy_f = torch.fft.fft(dy, n=N)
    k_f = torch.fft.fft(k, n=N)
    u_f = torch.fft.fft(u, n=N)

    # du: convolve upstream gradient with the conjugate filter
    du = torch.fft.ifft(dy_f * k_f.conj(), n=N).real[..., :L]

    # dk: convolve upstream gradient with the conjugate input, sum over batch
    dk = torch.fft.ifft(dy_f * u_f.conj(), n=N).real.sum(dim=0)[..., :L]

    return du, dk


# Validate against autograd
y = ref_fftconv_fwd(u, k, N)
y.backward(dy)

du_autograd = u.grad.clone()
dk_autograd = k.grad.clone()

du_manual, dk_manual = ref_fftconv_bwd(u.detach(), k.detach(), dy, N)

print(f"du allclose: {torch.allclose(du_autograd, du_manual, atol=1e-4, rtol=1e-3)}")
print(f"  max diff: {(du_autograd - du_manual).abs().max().item():.2e}")

print(f"dk allclose: {torch.allclose(dk_autograd, dk_manual, atol=1e-4, rtol=1e-3)}")
print(f"  max diff: {(dk_autograd - dk_manual).abs().max().item():.2e}")
