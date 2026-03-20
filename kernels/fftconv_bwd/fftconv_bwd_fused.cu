/**
 * fftconv_bwd_fused.cu — Fused backward pass (du + dk_f) without LCSF.
 *
 * This is a flat kernel using TK primitives directly. No producer-consumer
 * pipeline — loads are synchronous, but we get full control over shared memory
 * aliasing and can compute both du and dk_f in a single pass.
 *
 * Per-tile computation:
 *   1. Load dy, compute FFT(dy) = F @ dy → *= tw → @ F, save to smem
 *   2. Load u,  compute FFT(u)  = F @ u  → *= tw → @ F
 *   3. dk_f_acc += FFT(dy) * conj(FFT(u))    [accumulated in smem across batch]
 *   4. Reload FFT(dy), continue du backward:
 *      *= conj(kf) → @ Finv → *= conj(tw) → Finv @ d
 *   5. Store du to global memory
 *   6. On head change: write dk_f_acc to global memory
 *
 * Shared memory budget (~152KB, well within 227KB):
 *   - f, finv, tw, twinv_bwd, kf_conj: 5 × 16KB = 80KB  (constant matrices)
 *   - input tile:                       1 × 8KB  =  8KB  (dy or u, aliased)
 *   - fft_dy_save:                      1 × 16KB = 16KB  (save FFT(dy))
 *   - dk_f_acc:                         1 × 16KB = 16KB  (per-head accumulator)
 *   - tmp:                              2 × 16KB = 32KB  (workspace)
 *
 * Grid: 132 blocks (persistent), 128 threads (1 warpgroup = 4 warps)
 * Processes 1 batch element per iteration.
 */

#include "kittens.cuh"
#include "prototype.cuh"

#ifdef TORCH_COMPILE
#define TK_COMPILE_FFTCONV_BWD_FUSED
#endif

using namespace kittens;

// ============================================================================
// Shared memory layout
// ============================================================================

struct fused_bwd_smem {
    // Constant matrices (loaded once at kernel start)
    cst_bf<64, 64> f;           // FFT matrix
    cst_bf<64, 64> finv;        // IFFT matrix
    cst_bf<64, 64> tw;          // twiddle (unnormalized, for FFT of both dy and u)
    cst_bf<64, 64> twinv_bwd;   // conj(tw_fwd) = twinv/N (for du step 6)

    // Per-head data (reloaded on head change)
    cst_bf<64, 64> kf_conj;     // conj(kf)

    // Workspace (aliased/reused across phases)
    st_bf<64, 64> input_tile;   // for loading dy or u from global memory
    cst_bf<64, 64> fft_dy_save; // save FFT(dy) while computing FFT(u)
    cst_bf<64, 64> dk_f_acc;    // dk_f accumulator (summed across batch)
    cst_bf<64, 64> tmp;         // workspace for left-multiply smem roundtrip
};

// ============================================================================
// Global memory layout types
// ============================================================================

using seq_layout    = gl<bf16, -1, -1, 64, 64, st_bf<64,64>>;
using filter_layout = cgl<gl<bf16, 1, -1, 64, 64, st_bf<64,64>>>;
using fft_layout    = cgl<gl<bf16, 1, 1, 64, 64>>;
using dkf_layout    = cgl<gl<bf16, 1, -1, 64, 64, st_bf<64,64>>>; // (1, H, 64, 64) for dk_f output

struct fused_bwd_globals {
    seq_layout dy, u, du;                    // (B, H, 64, 64) inputs/output
    filter_layout kf_conj;                   // (1, H, 64, 64) complex
    dkf_layout dk_f_out;                     // (1, H, 64, 64) complex dk_f output
    fft_layout f, finv, tw, twinv_bwd;       // (1, 1, 64, 64) complex constants
};

// ============================================================================
// Kernel
// ============================================================================

static constexpr int BLOCK_THREADS = 128; // 1 warpgroup = 4 warps

__global__ __launch_bounds__(BLOCK_THREADS)
void fftconv_bwd_fused_kernel(fused_bwd_globals g, int B, int H) {
    extern __shared__ char smem_raw[];
    auto &s = *(fused_bwd_smem*)smem_raw;

    // ---- Load constant matrices (once per kernel) ----
    using all_threads = group<4>; // 4 warps
    all_threads::load(s.f,         g.f,         {0, 0, 0, 0});
    all_threads::load(s.finv,      g.finv,      {0, 0, 0, 0});
    all_threads::load(s.tw,        g.tw,        {0, 0, 0, 0});
    all_threads::load(s.twinv_bwd, g.twinv_bwd, {0, 0, 0, 0});
    __syncthreads();

    int total_heads = H;

    // ---- Persistent grid: iterate over heads ----
    for (int head = blockIdx.x; head < total_heads; head += gridDim.x) {

        // Load filter for this head
        all_threads::load(s.kf_conj, g.kf_conj, {0, head, 0, 0});

        // Zero dk_f accumulator
        warpgroup::zero(s.dk_f_acc.real);
        warpgroup::zero(s.dk_f_acc.imag);
        __syncthreads();

        for (int b = 0; b < B; b++) {

            // Register tiles
            crt_fl<16, 64> mma_reg;
            crt_bf<16, 64> accum, fft_dy, fft_u, tw_tmp;

            // ================================================
            // Phase 1: Load dy, compute FFT(dy), save to smem
            // ================================================

            all_threads::load(s.input_tile, g.dy, {b, head, 0, 0});
            __syncthreads();

            // FFT stage 1: F @ dy (left multiply, dy is real)
            warpgroup::mm_AB(mma_reg.real, s.f.real, s.input_tile);
            warpgroup::mm_AB(mma_reg.imag, s.f.imag, s.input_tile);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);

            // Twiddle: *= tw
            warpgroup::load(tw_tmp, s.tw);
            warp::mul(accum, accum, tw_tmp);
            __syncthreads();

            // FFT stage 2: @ F (right multiply, complex)
            warpgroup::mm_AB(mma_reg, accum, s.f);
            warpgroup::mma_async_wait();
            warp::copy(fft_dy, mma_reg);

            // Save FFT(dy) to shared memory
            warpgroup::store(s.fft_dy_save, fft_dy);
            __syncthreads();

            // ================================================
            // Phase 2: Load u, compute FFT(u)
            // ================================================

            all_threads::load(s.input_tile, g.u, {b, head, 0, 0});
            __syncthreads();

            // FFT stage 1: F @ u
            warpgroup::mm_AB(mma_reg.real, s.f.real, s.input_tile);
            warpgroup::mm_AB(mma_reg.imag, s.f.imag, s.input_tile);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);

            // Twiddle: *= tw
            warpgroup::load(tw_tmp, s.tw);
            warp::mul(accum, accum, tw_tmp);
            __syncthreads();

            // FFT stage 2: @ F
            warpgroup::mm_AB(mma_reg, accum, s.f);
            warpgroup::mma_async_wait();
            warp::copy(fft_u, mma_reg);

            // ================================================
            // Phase 3: Accumulate dk_f += FFT(dy) * conj(FFT(u))
            // ================================================

            // Reload FFT(dy) from smem
            warpgroup::load(fft_dy, s.fft_dy_save);

            // Complex multiply: (a+bi)(c-di) = (ac+bd) + (bc-ad)i
            rt_bf<16, 64> t1, t2;

            warp::mul(t1, fft_dy.real, fft_u.real);       // ac
            warp::mul(t2, fft_dy.imag, fft_u.imag);       // bd
            warp::add(accum.real, t1, t2);                 // ac + bd

            warp::mul(t1, fft_dy.imag, fft_u.real);       // bc
            warp::mul(t2, fft_dy.real, fft_u.imag);       // ad
            warp::sub(accum.imag, t1, t2);                 // bc - ad

            // Add to dk_f accumulator in smem
            // Load current accumulator, add partial, store back
            crt_bf<16, 64> dk_acc_reg;
            warpgroup::load(dk_acc_reg, s.dk_f_acc);
            warp::add(dk_acc_reg.real, dk_acc_reg.real, accum.real);
            warp::add(dk_acc_reg.imag, dk_acc_reg.imag, accum.imag);
            warpgroup::store(s.dk_f_acc, dk_acc_reg);
            __syncthreads();

            // ================================================
            // Phase 4: Continue du backward from FFT(dy)
            // ================================================

            // Reload FFT(dy) (accum was overwritten by dk_f computation)
            warpgroup::load(fft_dy, s.fft_dy_save);

            // Step 4: *= conj(kf)
            warpgroup::load(tw_tmp, s.kf_conj);
            warp::mul(fft_dy, fft_dy, tw_tmp);

            // Step 5: @ Finv (right multiply)
            warpgroup::mm_AB(mma_reg, fft_dy, s.finv);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);

            // Step 6: *= twinv_bwd (= conj(tw_fwd))
            warpgroup::load(tw_tmp, s.twinv_bwd);
            warp::mul(accum, accum, tw_tmp);

            // Step 7-8: Finv @ d (left multiply via smem roundtrip)
            warpgroup::store(s.tmp, accum);
            __syncthreads();
            warpgroup::mm_AB(mma_reg, s.finv, s.tmp);
            warpgroup::mma_async_wait();

            // ================================================
            // Phase 5: Store du to global memory
            // ================================================

            warpgroup::store(s.input_tile, mma_reg.real); // reuse input_tile
            __syncthreads();
            all_threads::store(g.du, s.input_tile, {b, head, 0, 0});
            __syncthreads();
        }

        // ================================================
        // Write accumulated dk_f for this head to global memory
        // ================================================
        all_threads::store(g.dk_f_out.real, s.dk_f_acc.real, {0, head, 0, 0});
        all_threads::store(g.dk_f_out.imag, s.dk_f_acc.imag, {0, head, 0, 0});
        __syncthreads();
    }
}


// ============================================================================
// Launch
// ============================================================================

#ifdef TK_COMPILE_FFTCONV_BWD_FUSED
#include <ATen/cuda/CUDAContext.h>
#include "pyutils/torchutils.cuh"
#include <ATen/Functions.h>

void launch_fused(fused_bwd_globals G, int B, int H) {
    unsigned long mem_size = sizeof(fused_bwd_smem);
    cudaFuncSetAttribute(
        fftconv_bwd_fused_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(132);
    dim3 block(BLOCK_THREADS);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fftconv_bwd_fused_kernel<<<grid, block, mem_size, stream>>>(G, B, H);
}

/**
 * fftconv_bwd_fused — Computes both du and dk_f in a single kernel launch.
 *
 * Inputs (all bf16):
 *   dy:       (B, H, N1, N1) — grad output
 *   u:        (B, H, N1, N1) — original input
 *   kf_conj:  (H, N1, N1) real + imag — conj(kf), precomputed
 *   f, finv:  (N1, N1) real + imag — FFT/IFFT matrices
 *   tw:       (N1, N1) real + imag — twiddle (unnormalized, no /N)
 *   twinv_bwd:(N1, N1) real + imag — conj(tw_fwd) = twinv/N
 *
 * Returns: (du, dk_f_real, dk_f_imag)
 *   du:       (B, H, N1, N1) — input gradient
 *   dk_f_*:   (H, N1, N1) — filter gradient (complex, already summed over B)
 */
std::vector<at::Tensor> fftconv_bwd_fused(
    const at::Tensor dy,
    const at::Tensor u,
    const at::Tensor kf_conj_real,
    const at::Tensor kf_conj_imag,
    const at::Tensor f_real,
    const at::Tensor f_imag,
    const at::Tensor finv_real,
    const at::Tensor finv_imag,
    const at::Tensor tw_real,
    const at::Tensor tw_imag,
    const at::Tensor twinv_bwd_real,
    const at::Tensor twinv_bwd_imag,
    int B, int H, int N, int N1
) {
    CHECK_INPUT(dy); CHECK_INPUT(u);
    CHECK_INPUT(kf_conj_real); CHECK_INPUT(kf_conj_imag);
    CHECK_INPUT(f_real); CHECK_INPUT(f_imag);
    CHECK_INPUT(finv_real); CHECK_INPUT(finv_imag);
    CHECK_INPUT(tw_real); CHECK_INPUT(tw_imag);
    CHECK_INPUT(twinv_bwd_real); CHECK_INPUT(twinv_bwd_imag);

    TORCH_CHECK(N == 4096, "fused backward kernel only supports N=4096, got ", N);
    TORCH_CHECK(N1 == 64, "fused backward kernel requires N1=64, got ", N1);
    TORCH_CHECK(dy.size(0) == B && dy.size(1) == H, "dy shape mismatch");
    TORCH_CHECK(u.size(0) == B && u.size(1) == H, "u shape mismatch");
    TORCH_CHECK(f_real.size(0) == 64 && f_real.size(1) == 64, "f_real must be 64x64");
    TORCH_CHECK(finv_real.size(0) == 64 && finv_real.size(1) == 64, "finv_real must be 64x64");
    TORCH_CHECK(tw_real.size(0) == 64 && tw_real.size(1) == 64, "tw_real must be 64x64");

    at::Tensor du_out     = at::empty({B, H, N1, N1}, dy.options());
    at::Tensor dk_f_r_out = at::zeros({1, H, N1, N1}, dy.options()); // zeros — kernel accumulates
    at::Tensor dk_f_i_out = at::zeros({1, H, N1, N1}, dy.options());

    auto ptr = [](const at::Tensor &t) { return reinterpret_cast<bf16*>(t.data_ptr<c10::BFloat16>()); };

    seq_layout    dy_gl {ptr(dy),  (unsigned long)B, (unsigned long)H, nullptr, nullptr};
    seq_layout    u_gl  {ptr(u),   (unsigned long)B, (unsigned long)H, nullptr, nullptr};
    seq_layout    du_gl {ptr(du_out), (unsigned long)B, (unsigned long)H, nullptr, nullptr};

    filter_layout kf_gl {
        typename filter_layout::component{ptr(kf_conj_real), nullptr, (unsigned long)H, nullptr, nullptr},
        typename filter_layout::component{ptr(kf_conj_imag), nullptr, (unsigned long)H, nullptr, nullptr}
    };
    dkf_layout dk_f_gl {
        typename dkf_layout::component{ptr(dk_f_r_out), nullptr, (unsigned long)H, nullptr, nullptr},
        typename dkf_layout::component{ptr(dk_f_i_out), nullptr, (unsigned long)H, nullptr, nullptr}
    };

    auto make_fft_gl = [](bf16* r, bf16* i) {
        return fft_layout{
            typename fft_layout::component{r, nullptr, nullptr, nullptr, nullptr},
            typename fft_layout::component{i, nullptr, nullptr, nullptr, nullptr}
        };
    };

    fft_layout f_gl        = make_fft_gl(ptr(f_real),        ptr(f_imag));
    fft_layout finv_gl     = make_fft_gl(ptr(finv_real),     ptr(finv_imag));
    fft_layout tw_gl       = make_fft_gl(ptr(tw_real),       ptr(tw_imag));
    fft_layout twinv_bwd_gl= make_fft_gl(ptr(twinv_bwd_real),ptr(twinv_bwd_imag));

    fused_bwd_globals G{dy_gl, u_gl, du_gl, kf_gl, dk_f_gl, f_gl, finv_gl, tw_gl, twinv_bwd_gl};
    launch_fused(G, B, H);

    CHECK_CUDA_ERROR(cudaGetLastError());
    return {du_out, dk_f_r_out.squeeze(0), dk_f_i_out.squeeze(0)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fftconv_bwd_fused", fftconv_bwd_fused,
        "Fused FFTConv backward (du + dk_f). Single kernel, no LCSF. "
        "Returns (du, dk_f_real, dk_f_imag). dk_f is already summed over batch.");
}
#else
// TODO: standalone harness
#endif
