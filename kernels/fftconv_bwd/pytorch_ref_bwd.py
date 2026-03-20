import torch
import numpy as np
import sys

N = 32 * 32 * 32
B = 1
H = 1
N1 = 32
N2 = 32

TESTNAME = sys.argv[1] if len(sys.argv) > 1 else 'randn'

if TESTNAME in ['ones']:
    torch.random.manual_seed(41)
    u = (torch.ones((B, H, N), dtype=torch.cfloat, device='cpu'))
    k = (torch.ones((H, N), dtype=torch.cfloat, device='cpu'))
elif TESTNAME in ['randn']:
    torch.random.manual_seed(41)
    u = (torch.randn((B, H, N), dtype=torch.cfloat, device='cpu'))
    k = (torch.randn((H, N), dtype=torch.cfloat, device='cpu'))
else:
    print('Invalid test name')
    sys.exit(0)


############ CLOSED-FORM REFERENCES ############


def ref_fftconv(u, k, N):
    """Forward: Y = IFFT(FFT(u) * FFT(k))"""
    L = u.shape[-1]
    u_f = torch.fft.fft(u.float(), n=N)
    k_f = torch.fft.fft(k.float(), n=N)
    y_f = u_f * k_f
    y = torch.fft.ifft(y_f, n=N).real[..., :L].to(u.dtype).contiguous()
    return y


def ref_fftconv_bwd(u, k, dy, N):
    """
    Closed-form backward pass for FFT convolution. No forward pass values needed.

    Forward was: Y = IFFT(FFT(u) * FFT(k))
    Backward:
        du = IFFT(FFT(dy) * conj(FFT(k)))   -- convolve gradient with conjugate filter
        dk = IFFT(FFT(dy) * conj(FFT(u)))   -- convolve gradient with conjugate input, sum over batch

    All inputs/outputs are complex (matching the monarch decomposition).

    Args:
        u:  original input,  (B, H, N) complex
        k:  original filter, (H, N) complex
        dy: grad output,     (B, H, N) complex
        N:  FFT size

    Returns:
        du: grad w.r.t. u,  (B, H, N) complex
        dk: grad w.r.t. k,  (H, N) complex
    """
    L = u.shape[-1]

    dy_f = torch.fft.fft(dy, n=N)
    k_f  = torch.fft.fft(k, n=N)
    u_f  = torch.fft.fft(u, n=N)

    # Input gradient: another FFT convolution with conjugate filter
    du = torch.fft.ifft(dy_f * k_f.conj(), n=N)[..., :L]

    # Filter gradient: sum over batch dimension
    dk = torch.fft.ifft((dy_f * u_f.conj()).sum(dim=0), n=N)[..., :L]

    return du, dk


############ GENERATE KERNEL INPUTS ############


def fft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    M = torch.exp(-2j * torch.pi * n * k / N)
    return M

def compute_twiddle_factors_fft(n, m):
    """Compute the twiddle factors of size n x m"""
    n_a = torch.arange(n).view(-1, 1)
    m_a = torch.arange(m)
    N = n * m
    M = torch.exp(-2j * torch.pi * n_a * m_a / N)
    return M

def ifft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    M = torch.exp(2j * torch.pi * n * k / N)
    return M

def compute_twiddle_factors_ifft(n, m):
    """Compute the twiddle factors of size n x m"""
    n_a = torch.arange(n).view(-1, 1)
    m_a = torch.arange(m)
    N = n * m
    M = torch.exp(2j * torch.pi * n_a * m_a / N)
    return M


############## MONARCH BACKWARD ##############
#
# For each forward op, the adjoint (for a real-valued loss) is:
#   z = x @ A          =>  dx = dz @ A^H          (A^H = conj(A).T)
#   z = x * a          =>  dx = dz * conj(a)       (pointwise)
#   z = x.transpose()  =>  dx = dz.transpose()     (self-adjoint)
#
# We walk the forward ops in reverse, applying adjoints.
# Filter gradient: at the pointwise multiply x6 = x5 * k_f,
#   dk_f = (dz * conj(x5)).sum(dim=0)   (sum over batch B)
# We recompute x5 (FFT of input, before filter multiply) from the saved input.


def monarch_conv_fwd_x5(x, f_sqrt_N_fft, twiddle_factors_fft, sqrt_N):
    '''Recompute the FFT portion of monarch_conv to get x5 (pre-filter intermediate). This is done instead of saving the forward pass values.'''
    B, H, K, N = x.shape
    x = x.reshape(B, H, K, sqrt_N, sqrt_N)
    x = x.transpose(-1, -2)
    x = x @ f_sqrt_N_fft
    x = x.transpose(-1, -2)
    x = x * twiddle_factors_fft
    x = x @ f_sqrt_N_fft
    return x  # (B, H, K, sqrt_N, sqrt_N)


def monarch_conv_bwd(dy, x, k_f, f_sqrt_N_fft, twiddle_factors_fft, f_sqrt_N_ifft, twiddle_factors_ifft, N, sqrt_N):
    '''
    Backward pass through monarch_conv.

    Args:
        dy: grad w.r.t. output, shape (B, H, K, N)
        x:  original input (for recomputing FFT(x)), shape (B, H, K, N)
        k_f, f_sqrt_N_fft, twiddle_factors_fft, f_sqrt_N_ifft, twiddle_factors_ifft: same as forward
        N, sqrt_N: dimensions

    Returns:
        dx:   grad w.r.t. input, shape (B, H, K, N)
        dk_f: grad w.r.t. k_f, shape (H, K*sqrt_N*sqrt_N) — summed over batch
    '''
    B, H, K, N_dim = dy.shape

    # Recompute forward FFT of x to get x5 (needed for filter gradient)
    x5 = monarch_conv_fwd_x5(x, f_sqrt_N_fft, twiddle_factors_fft, sqrt_N)  # (B, H, K, sqrt_N, sqrt_N)
    k_f_r = k_f.reshape(H, K, sqrt_N, sqrt_N)

    # Reshape dy to tile form
    dy = dy.reshape(B, H, K, sqrt_N, sqrt_N)

    # --- Walk backward through IFFT + filter multiply ---

    # Adjoint of: x = x.transpose(-1, -2)      [final transpose]
    d = dy.transpose(-1, -2)

    # Adjoint of: x = x @ f_sqrt_N_ifft         [IFFT stage 2]
    d = d @ f_sqrt_N_ifft.conj().T

    # Adjoint of: x = x * twiddle_factors_ifft   [inverse twiddle]
    d = d * twiddle_factors_ifft.conj()

    # Adjoint of: x = x.transpose(-1, -2)
    d = d.transpose(-1, -2)

    # Adjoint of: x = x @ f_sqrt_N_ifft         [IFFT stage 1]
    d = d @ f_sqrt_N_ifft.conj().T

    # Adjoint of: x = x * k_f                   [pointwise filter multiply]
    dk_f_r = (d * x5.conj()).sum(dim=0)         # (H, K, sqrt_N, sqrt_N), summed over B
    d = d * k_f_r.conj()

    # --- Walk backward through FFT ---

    # Adjoint of: x = x @ f_sqrt_N_fft          [FFT stage 2]
    d = d @ f_sqrt_N_fft.conj().T

    # Adjoint of: x = x * twiddle_factors_fft    [twiddle]
    d = d * twiddle_factors_fft.conj()

    # Adjoint of: x = x.transpose(-1, -2)
    d = d.transpose(-1, -2)

    # Adjoint of: x = x @ f_sqrt_N_fft          [FFT stage 1]
    d = d @ f_sqrt_N_fft.conj().T

    # Adjoint of: x = x.transpose(-1, -2)       [initial transpose]
    d = d.transpose(-1, -2)

    dx = d.reshape(B, H, K, N_dim)
    dk_f = dk_f_r.reshape(H, K * sqrt_N * sqrt_N)

    return dx, dk_f


def monarch_conv_full_bwd(
    dy, x, k_f_permuted,
    f_32_fft,
    twiddle_factors_fft_32_1K, twiddle_factors_fft_32_32,
    f_32_ifft,
    twiddle_factors_ifft_32_1K, twiddle_factors_ifft_32_32,
    N, sqrt_N_32, N_1024
):
    '''
    Backward pass through monarch_conv_full.

    Args:
        dy: grad w.r.t. output, shape (B, H, sqrt_N_32, N_1024)
        x:  original input, shape (B, H, N)
        k_f_permuted: permuted frequency-domain filter, shape (H, N)
        [other args same as forward]

    Returns:
        dx:   grad w.r.t. input, shape (B, H, N)
        dk_f: grad w.r.t. k_f_permuted, shape (H, N) — summed over batch
    '''
    # Recompute forward intermediate: x4 = input to inner monarch_conv
    x_r = x.reshape(B, H, sqrt_N_32, N_1024)
    x_r = x_r.transpose(-1, -2)
    x_r = x_r @ f_32_fft
    x_r = x_r.transpose(-1, -2)
    x4 = x_r * twiddle_factors_fft_32_1K   # (B, H, 32, 1024)

    d = dy  # (B, H, 32, 1024)

    # --- Walk backward through outer IFFT ---

    # Adjoint of: x = x.transpose(-1, -2)       [final transpose]
    d = d.transpose(-1, -2)

    # Adjoint of: x = x @ f_32_ifft
    d = d @ f_32_ifft.conj().T

    # Adjoint of: x = x.transpose(-1, -2)
    d = d.transpose(-1, -2)

    # Adjoint of: x = x * twiddle_factors_ifft_32_1K
    d = d * twiddle_factors_ifft_32_1K.conj()

    # --- Backward through inner monarch_conv ---
    dx4, dk_f = monarch_conv_bwd(
        d, x4, k_f_permuted,
        f_32_fft, twiddle_factors_fft_32_32,
        f_32_ifft, twiddle_factors_ifft_32_32,
        N_1024, 32
    )

    # --- Walk backward through outer FFT ---

    # Adjoint of: x = x * twiddle_factors_fft_32_1K
    d = dx4 * twiddle_factors_fft_32_1K.conj()

    # Adjoint of: x = x.transpose(-1, -2)
    d = d.transpose(-1, -2)

    # Adjoint of: x = x @ f_32_fft
    d = d @ f_32_fft.conj().T

    # Adjoint of: x = x.transpose(-1, -2)       [initial transpose]
    d = d.transpose(-1, -2)

    dx = d.reshape(B, H, N)
    return dx, dk_f


############## CHUNKED MONARCH BACKWARD ##############
# Mirrors the chunked forward in fftconv/pytorch_ref.py.
# Each loop iteration = one tile/warpgroup's work in the CUDA kernel.
#
# Forward chunking order:
#   1. Outer FFT:  32 chunks of 32 along dim=-1 (the 1024 dim)
#   2. Inner:       8 chunks of  4 along dim= 2 (the K=32 dim)
#      per chunk: FFT → pointwise * k_f → IFFT
#   3. Outer IFFT:  4 chunks of 256 along dim=-1
#
# Backward (reversed):
#   1. Adjoint of outer IFFT:  4 chunks of 256 along dim=-1
#   2. Adjoint of inner:       8 chunks of  4 along dim=2
#      per chunk: adjoint IFFT → adjoint pointwise → adjoint FFT
#      (recompute x5 per chunk for filter gradient)
#   3. Adjoint of outer FFT:  32 chunks of 32 along dim=-1
#
# Plus: recompute outer FFT of u (same as forward step 1) to get x4,
#       needed for x5 recomputation inside the inner loop.


def monarch_conv_full_bwd_chunked(
    dy, x, k_f_permuted,
    f_mat, finv_mat,
    tw_32_1k, tw_32_32,
    tw_32_1k_inv, tw_32_32_inv,
    N, N1, N2
):
    d = dy.clone().reshape(B, H, N1, 1024).to(torch.cfloat)

    # ---- Recompute x4 = outer_FFT(u) (same as forward step 1) ----
    x4 = x.clone().reshape(B, H, 32, 1024).to(torch.cfloat)
    chunk_size = 32
    chunks = 32
    for i in range(chunks):
        block = x4[:, :, :, i*chunk_size:(i+1)*chunk_size]
        block = block.transpose(-1, -2)
        block = block @ f_mat
        block = block.transpose(-1, -2)
        block = block * tw_32_1k[:, i*chunk_size:(i+1)*chunk_size]
        x4[:, :, :, i*chunk_size:(i+1)*chunk_size] = block

    # ---- Step 1: Adjoint of outer IFFT (4 chunks of 256) ----
    # Forward outer IFFT was: *tw_inv → transpose → @Finv → transpose
    # Adjoint (reversed):     transpose → @conj(Finv).T → transpose → *conj(tw_inv)
    chunk_size = 256
    chunks = 1024 // chunk_size
    for i in range(chunks):
        block = d[:, :, :, i*chunk_size:(i+1)*chunk_size]
        block = block.transpose(-1, -2)
        block = block @ finv_mat.conj().T
        block = block.transpose(-1, -2)
        block = block * tw_32_1k_inv[:, i*chunk_size:(i+1)*chunk_size].conj()
        d[:, :, :, i*chunk_size:(i+1)*chunk_size] = block

    # ---- Step 2: Adjoint of inner monarch_conv (8 chunks of 4) ----
    d = d.reshape(B, H, 32, 32, 32)
    x4 = x4.reshape(B, H, 32, 32, 32)
    k_f_r = k_f_permuted.reshape(H, 32, 32, 32)
    dk_f_acc = torch.zeros_like(k_f_r)  # accumulate filter gradient

    chunk_size = 4
    chunks = 32 // chunk_size
    for i in range(chunks):
        # Recompute x5 for this chunk (FFT of x4 chunk)
        x5_chunk = x4[:, :, i*chunk_size:(i+1)*chunk_size]
        x5_chunk = x5_chunk.transpose(-1, -2)
        x5_chunk = x5_chunk @ f_mat
        x5_chunk = x5_chunk.transpose(-1, -2)
        x5_chunk = x5_chunk * tw_32_32
        x5_chunk = x5_chunk @ f_mat  # (B, H, 4, 32, 32)

        block = d[:, :, i*chunk_size:(i+1)*chunk_size]

        # Adjoint of IFFT: transpose → @conj(Finv).T → *conj(tw_inv) → transpose → @conj(Finv).T
        block = block.transpose(-1, -2)
        block = block @ finv_mat.conj().T
        block = block * tw_32_32_inv.conj()
        block = block.transpose(-1, -2)
        block = block @ finv_mat.conj().T

        # Adjoint of pointwise * k_f
        dk_f_acc[:, i*chunk_size:(i+1)*chunk_size] = (block * x5_chunk.conj()).sum(dim=0)
        block = block * k_f_r[:, i*chunk_size:(i+1)*chunk_size].conj()

        # Adjoint of FFT: @conj(F).T → *conj(tw) → transpose → @conj(F).T → transpose
        block = block @ f_mat.conj().T
        block = block * tw_32_32.conj()
        block = block.transpose(-1, -2)
        block = block @ f_mat.conj().T
        block = block.transpose(-1, -2)

        d[:, :, i*chunk_size:(i+1)*chunk_size] = block

    dk_f = dk_f_acc.reshape(H, N)

    # ---- Step 3: Adjoint of outer FFT (32 chunks of 32) ----
    # Forward outer FFT was: transpose → @F → transpose → *tw
    # Adjoint (reversed):    *conj(tw) → transpose → @conj(F).T → transpose
    d = d.reshape(B, H, 32, 1024)
    chunk_size = 32
    chunks = 32
    for i in range(chunks):
        block = d[:, :, :, i*chunk_size:(i+1)*chunk_size]
        block = block * tw_32_1k[:, i*chunk_size:(i+1)*chunk_size].conj()
        block = block.transpose(-1, -2)
        block = block @ f_mat.conj().T
        block = block.transpose(-1, -2)
        d[:, :, :, i*chunk_size:(i+1)*chunk_size] = block

    dx = d.reshape(B, H, N)
    return dx, dk_f


############## VERIFICATION ##############
# Source of truth: ref_fftconv_bwd (closed-form, no forward pass dependence)
# Under test:     monarch_conv_full_bwd (tile-decomposed, what the CUDA kernel will implement)


def verify():
    print("=== Backward Pass Verification ===\n")

    # --- Source of truth: closed-form backward ---
    torch.random.manual_seed(99)
    dy_flat = torch.randn((B, H, N), dtype=torch.cfloat)
    du_ref, dk_ref = ref_fftconv_bwd(u, k, dy_flat, N)

    print(f"ref du shape:  {du_ref.shape}")
    print(f"ref dk shape:  {dk_ref.shape}")

    # --- Under test: monarch backward ---
    # Prepare all matrices (same as forward)
    f_mat = fft_matrix(N1)
    finv_mat = ifft_matrix(N1)
    tw_32_1k = compute_twiddle_factors_fft(N1, 1024)
    tw_32_1k_inv = compute_twiddle_factors_ifft(N1, 1024) / N
    tw_32_32 = compute_twiddle_factors_fft(N2, N2)
    tw_32_32_inv = compute_twiddle_factors_ifft(N2, N2)

    # Prepare filter in permuted frequency domain
    k_f = torch.fft.fft(k, n=N)
    k_f_permuted = (k_f.reshape(H, 1024, N1)
                      .transpose(-1, -2)
                      .reshape(H, N1, N2, N2)
                      .transpose(-1, -2)
                      .reshape(H, N)
                      .contiguous())

    # dy must be reshaped to match monarch_conv_full output shape (B, H, 32, 1024)
    dy_tiled = dy_flat.reshape(B, H, N1, 1024)

    du_monarch, dkf_monarch = monarch_conv_full_bwd(
        dy_tiled, u, k_f_permuted,
        f_mat, tw_32_1k, tw_32_32,
        finv_mat, tw_32_1k_inv, tw_32_32_inv,
        N, 32, 1024
    )

    # --- Compare du (direct comparison, same domain) ---
    du_err = (du_monarch - du_ref).abs().max().item()
    du_rel = (du_monarch - du_ref).abs().max().item() / du_ref.abs().max().item()
    print(f"\ndu  max abs error: {du_err:.6e}")
    print(f"du  max rel error: {du_rel:.6e}")
    print(f"du  allclose (atol=1e-2): {torch.allclose(du_monarch, du_ref, atol=1e-2)}")

    # --- Compare dk_f ---
    # ref_fftconv_bwd returns dk in time domain.
    # monarch_conv_full_bwd returns dk_f_permuted in the permuted frequency domain.
    #
    # To compare, we convert dk_ref to the permuted frequency domain:
    #   dk -> FFT(dk) -> permute -> dk_f_permuted_ref
    # The monarch backward's dk_f includes a 1/N factor from the IFFT adjoint
    # (the forward absorbs 1/N into tw_32_1k_inv, and conj() preserves it).
    dk_f_ref = torch.fft.fft(dk_ref, n=N) / N
    dk_f_permuted_ref = (dk_f_ref.reshape(H, 1024, N1)
                           .transpose(-1, -2)
                           .reshape(H, N1, N2, N2)
                           .transpose(-1, -2)
                           .reshape(H, N)
                           .contiguous())

    dkf_err = (dkf_monarch - dk_f_permuted_ref).abs().max().item()
    dkf_rel = (dkf_monarch - dk_f_permuted_ref).abs().max().item() / dk_f_permuted_ref.abs().max().item()
    print(f"\ndk_f max abs error: {dkf_err:.6e}")
    print(f"dk_f max rel error: {dkf_rel:.6e}")
    print(f"dk_f allclose (atol=1e-2): {torch.allclose(dkf_monarch, dk_f_permuted_ref, atol=1e-2)}")

    # --- Under test: chunked monarch backward ---
    print("\n\n=== Chunked Backward Verification ===\n")

    du_chunked, dkf_chunked = monarch_conv_full_bwd_chunked(
        dy_tiled, u, k_f_permuted,
        f_mat, finv_mat,
        tw_32_1k, tw_32_32,
        tw_32_1k_inv, tw_32_32_inv,
        N, N1, N2
    )

    # Compare chunked du against closed-form ref
    du_c_err = (du_chunked - du_ref).abs().max().item()
    du_c_rel = (du_chunked - du_ref).abs().max().item() / du_ref.abs().max().item()
    print(f"du  max abs error: {du_c_err:.6e}")
    print(f"du  max rel error: {du_c_rel:.6e}")
    print(f"du  allclose (atol=1e-2): {torch.allclose(du_chunked, du_ref, atol=1e-2)}")

    # Compare chunked dk_f against closed-form ref (same domain conversion)
    dkf_c_err = (dkf_chunked - dk_f_permuted_ref).abs().max().item()
    dkf_c_rel = (dkf_chunked - dk_f_permuted_ref).abs().max().item() / dk_f_permuted_ref.abs().max().item()
    print(f"\ndk_f max abs error: {dkf_c_err:.6e}")
    print(f"dk_f max rel error: {dkf_c_rel:.6e}")
    print(f"dk_f allclose (atol=1e-2): {torch.allclose(dkf_chunked, dk_f_permuted_ref, atol=1e-2)}")

    # Sanity: chunked should match whole-tensor monarch exactly
    print("\n\n=== Chunked vs Whole-Tensor Monarch ===\n")
    print(f"du  match: {torch.allclose(du_chunked, du_monarch, atol=1e-6)}")
    print(f"dk_f match: {torch.allclose(dkf_chunked, dkf_monarch, atol=1e-6)}")


verify()
