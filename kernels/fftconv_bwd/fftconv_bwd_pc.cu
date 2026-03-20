/**
 * fftconv_bwd_pc.cu — Backward pass for FFT convolution using ThunderKittens.
 *
 * KEY INSIGHT: The du (input gradient) kernel has identical compute structure
 * to the forward kernel. The adjoint of the Monarch FFT convolution maps to
 * the same sequence of matrix multiplies (F, F, Finv, Finv), just with
 * conjugated pointwise operands:
 *
 *   Forward:             F@X → *tw      → @F → *kf      → @Finv → *twinv_t     → Finv@X
 *   Backward (du):       F@dy → *conj(twinv_t) → @F → *conj(kf) → @Finv → *conj(tw) → Finv@X
 *
 * So the kernel is the same — we just pass conjugated data into the tw/kf/twinv_t slots.
 * Conjugation = negate imaginary part, done on the Python side before kernel launch.
 *
 * The dk_f (filter gradient) kernel is separate because it has a different
 * access pattern: two inputs (dy, u), only the FFT half of ops, and per-head
 * output (accumulated over batch). It outputs per-batch partials; the batch
 * reduction (sum over B) is done in Python.
 */

#include "kittens.cuh"
#include "prototype.cuh"

#ifdef TORCH_COMPILE
#define TK_COMPILE_FFTCONV_BWD
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;


// ============================================================================
// Layout and template for du backward (1024 variant)
// Identical to forward fftconv_1024_layout — same tiles, same scratch.
// ============================================================================

template<int _wg> struct fftconv_bwd_1024_layout {
    static constexpr int wg = _wg;
    using seq_tile      = st_bf<64, 64>;
    using seq_layout    =     gl<bf16, -1, -1, 32, 32>;
    using filter_layout = cgl<gl<bf16,  1, -1, 64, 64>>;
    using fft_layout    = cgl<gl<bf16,  1,  1, 64, 64>>;
    struct globals {
        seq_layout du, dy;               // output=du, input=dy
        filter_layout kf_conj;           // conj(kf) in the filter slot
        fft_layout f, finv, tw_bwd, twinv_t_bwd;
        // tw_bwd = conj(twinv_t),  twinv_t_bwd = conj(tw)
    };
    struct input_block    { seq_tile dy[wg]; };
    struct output_block   { seq_tile du[wg]; };
    struct scratch_block  {
        cst_bf<64, 64> kf_conj, f, finv, tw_bwd, twinv_t_bwd, tmp[2];
    };
    struct consumer_state { int current_head; };
};

struct fft_bwd_1024_template {
    static constexpr int NUM_CONSUMER_WARPS=8, NUM_CONSUMER_WARPGROUPS=NUM_CONSUMER_WARPS/4, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=3, INPUT_PIPE_STAGES=3;
    using layout = fftconv_bwd_1024_layout<NUM_CONSUMER_WARPGROUPS>;

    __device__ static inline void load_head_data(typename layout::scratch_block &scratch, const layout::globals &g, int head) {
        using consumers = group<NUM_CONSUMER_WARPS>;
        consumers::sync(3);
        consumers::load(scratch.kf_conj, g.kf_conj, {0, head, 0, 0});
        consumers::sync(3);
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int heads_handled = (args.globals.dy.depth()+131-blockIdx.x) / 132;
        int iters_per_head = (args.globals.dy.batch() + (NUM_CONSUMER_WARPGROUPS*4)-1) / (NUM_CONSUMER_WARPGROUPS*4);
        args.num_iters = args.task_iter == 0 ? heads_handled * iters_per_head : -1;
    }

    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            int iters_per_head = (args.globals.dy.batch() + (NUM_CONSUMER_WARPGROUPS*4)-1) / (NUM_CONSUMER_WARPGROUPS*4);
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * (NUM_CONSUMER_WARPGROUPS*4);
            if(warpgroup::warpid() == args.iter%4) {
                for(int b = batch; b < batch+(NUM_CONSUMER_WARPGROUPS*4) && b < args.globals.dy.batch(); b++) {
                    int diff = b-batch;
                    auto st = args.input.dy[diff/4].template subtile<32,32>({(diff%4)/2, diff%2});
                    warp::load_async(st, args.globals.dy, { b, head, 0, 0 });
                }
                load_async_wait();
                if(laneid() == 0) arrive(args.inputs_arrived, 4);
                __syncwarp();
            }
        }
        __device__ static inline void store(producer_store_args<layout> args) {
            int iters_per_head = (args.globals.dy.batch() + (NUM_CONSUMER_WARPGROUPS*4)-1) / (NUM_CONSUMER_WARPGROUPS*4);
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * (NUM_CONSUMER_WARPGROUPS*4);
            if(warpgroup::warpid() == args.iter%4) {
                for(int b = batch; b < batch+(NUM_CONSUMER_WARPGROUPS*4) && b < args.globals.dy.batch(); b++) {
                    int diff = b-batch;
                    auto st = args.output.du[diff/4].subtile<32,32>({(diff%4)/2, diff%2});
                    warp::store(args.globals.du, st, { b, head, 0, 0 });
                }
                __syncwarp();
                if(laneid() == 0) arrive(args.outputs_finished, 4);
                __syncwarp();
            }
        }
    };

    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            int iters_per_head = (args.globals.dy.batch() + (NUM_CONSUMER_WARPGROUPS*4)-1) / (NUM_CONSUMER_WARPGROUPS*4);
            args.state.current_head = (0 / iters_per_head)*132 + blockIdx.x;
            using consumers = group<NUM_CONSUMER_WARPS>;
            consumers::load(args.scratch.f,           args.globals.f,           {0, 0, 0, 0});
            consumers::load(args.scratch.finv,        args.globals.finv,        {0, 0, 0, 0});
            consumers::load(args.scratch.tw_bwd,      args.globals.tw_bwd,      {0, 0, 0, 0});
            consumers::load(args.scratch.twinv_t_bwd, args.globals.twinv_t_bwd, {0, 0, 0, 0});
            load_head_data(args.scratch, args.globals, args.state.current_head);
        }

        __device__ static inline void compute(consumer_compute_args<layout> args) {
            int warpgroupid = warpgroup::groupid();
            int default_barrer_id = warpgroupid+4;

            // The compute is structurally identical to the forward.
            // The adjoint just swaps which data sits in each slot:
            //   scratch.tw_bwd      = conj(twinv_t)   [was tw in forward]
            //   scratch.kf_conj     = conj(kf)         [was kf in forward]
            //   scratch.twinv_t_bwd = conj(tw)         [was twinv_t in forward]

            crt_fl<16, 64> mma_reg;
            crt_bf<16, 64> accum, tmp;

            // Adjoint of IFFT stage 2: F @ dy  (Finv^H = F)
            warpgroup::mm_AB(mma_reg.real, args.scratch.f.real, args.input.dy[warpgroup::groupid()]);
            warpgroup::mm_AB(mma_reg.imag, args.scratch.f.imag, args.input.dy[warpgroup::groupid()]);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);

            // Adjoint of inverse twiddle: *= conj(twinv_t)
            warpgroup::load(tmp, args.scratch.tw_bwd);
            warp::mul(accum, accum, tmp);

            group<NUM_CONSUMER_WARPS>::sync(2);

            // Adjoint of IFFT stage 1: @ F  (Finv^H = F)
            warpgroup::mm_AB(mma_reg, accum, args.scratch.f);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);

            // Adjoint of filter multiply: *= conj(kf)
            warpgroup::load(tmp, args.scratch.kf_conj);
            warp::mul(accum, accum, tmp);

            // Adjoint of FFT stage 2: @ Finv  (F^H = Finv)
            warpgroup::mm_AB(mma_reg, accum, args.scratch.finv);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);

            // Adjoint of twiddle: *= conj(tw)
            warpgroup::load(tmp, args.scratch.twinv_t_bwd);
            warp::mul(accum, accum, tmp);

            // Adjoint of FFT stage 1: Finv @ d  (F^H = Finv)
            warpgroup::store(args.scratch.tmp[warpgroup::groupid()], accum);
            warpgroup::sync(default_barrer_id);

            warpgroup::mm_AB(mma_reg, args.scratch.finv, args.scratch.tmp[warpgroup::groupid()]);
            warpgroup::mma_async_wait();

            warpgroup::store(args.output.du[warpgroup::groupid()], mma_reg.real);
            warpgroup::sync(default_barrer_id);

            if(laneid() == 0) {
                arrive(args.inputs_finished);
                arrive(args.outputs_arrived);
            }
            __syncwarp();

            // persistent grid: reload filter for next head
            // NOTE: must use *4 divisor to match common_setup and producer,
            // since 1024 variant packs 4 subtiles (32x32) per 64x64 tile.
            int iters_per_head = (args.globals.dy.batch() + (NUM_CONSUMER_WARPGROUPS*4)-1) / (NUM_CONSUMER_WARPGROUPS*4);
            int next_head = ((args.iter+1) / iters_per_head)*132 + blockIdx.x;
            if(next_head != args.state.current_head) {
                load_head_data(args.scratch, args.globals, next_head);
                args.state.current_head = next_head;
            }
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) { if(laneid() == 0) arrive(args.finish_finished); }
    };
};


// ============================================================================
// Layout and template for du backward (4096 variant)
// Identical structure, uses TMA loads.
// ============================================================================

template<int _wg> struct fftconv_bwd_4096_layout {
    static constexpr int wg = _wg;
    using seq_tile      = st_bf<64, 64>;
    using seq_layout    =     gl<bf16, -1, -1, 64, 64, seq_tile>;
    using filter_layout = cgl<gl<bf16,  1, -1, 64, 64, seq_tile>>;
    using fft_layout    = cgl<gl<bf16,  1,  1, 64, 64>>;
    struct globals {
        seq_layout du, dy;
        filter_layout kf_conj;
        fft_layout f, finv, tw_bwd, twinv_t_bwd;
    };
    struct input_block    { seq_tile dy[wg]; };
    struct output_block   { seq_tile du[wg]; };
    struct scratch_block  {
        cst_bf<64, 64> kf_conj, f, finv, tw_bwd, twinv_t_bwd, tmp[2];
    };
    struct consumer_state { int current_head; };
};

struct fft_bwd_4096_template {
    static constexpr int NUM_CONSUMER_WARPS=8, NUM_CONSUMER_WARPGROUPS=NUM_CONSUMER_WARPS/4, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=2, INPUT_PIPE_STAGES=4;
    using layout = fftconv_bwd_4096_layout<NUM_CONSUMER_WARPGROUPS>;

    __device__ static inline void load_head_data(typename layout::scratch_block &scratch, const layout::globals &g, int head) {
        using consumers = group<NUM_CONSUMER_WARPS>;
        consumers::sync(3);
        consumers::load(scratch.kf_conj, g.kf_conj, {0, head, 0, 0});
        consumers::sync(3);
    }

    __device__ static void common_setup(common_setup_args<layout> args) {
        int heads_handled = (args.globals.dy.depth()+131-blockIdx.x) / 132;
        int iters_per_head = (args.globals.dy.batch() + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
        args.num_iters = args.task_iter == 0 ? heads_handled * iters_per_head : -1;
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static void load(producer_load_args<layout> args) {
            int iters_per_head = (args.globals.dy.batch() + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * NUM_CONSUMER_WARPGROUPS;
            if(warpgroup::warpid() == args.iter%4) {
                warp::tma::expect_bytes(args.inputs_arrived, sizeof(args.input.dy[0]) * min((int)NUM_CONSUMER_WARPGROUPS, (int)(args.globals.dy.batch() - batch)));
                for(int b = batch; b < batch+NUM_CONSUMER_WARPGROUPS && b < args.globals.dy.batch(); b++) {
                    warp::tma::load_async(args.input.dy[b-batch], args.globals.dy, { b, head, 0, 0 }, args.inputs_arrived);
                }
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
                __syncwarp();
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            int iters_per_head = (args.globals.dy.batch() + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * NUM_CONSUMER_WARPGROUPS;
            if(warpgroup::warpid() == args.iter%4) {
                for(int b = batch; b < batch+NUM_CONSUMER_WARPGROUPS && b < args.globals.dy.batch(); b++) {
                    warp::tma::store_async(args.globals.du, args.output.du[b-batch], { b, head, 0, 0 });
                }
                warp::tma::store_async_read_wait();
                if(laneid() == 0) arrive(args.outputs_finished, 4);
                __syncwarp();
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>();
            int iters_per_head = (args.globals.dy.batch() + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            args.state.current_head = (0 / iters_per_head)*132 + blockIdx.x;
            using consumers = group<NUM_CONSUMER_WARPS>;
            consumers::load(args.scratch.f,           args.globals.f,           {0, 0, 0, 0});
            consumers::load(args.scratch.finv,        args.globals.finv,        {0, 0, 0, 0});
            consumers::load(args.scratch.tw_bwd,      args.globals.tw_bwd,      {0, 0, 0, 0});
            consumers::load(args.scratch.twinv_t_bwd, args.globals.twinv_t_bwd, {0, 0, 0, 0});
            load_head_data(args.scratch, args.globals, args.state.current_head);
        }

        __device__ static void compute(consumer_compute_args<layout> args) {
            int warpgroupid = warpgroup::warpid()/kittens::WARPGROUP_WARPS;
            int default_barrer_id = warpgroupid + 4;

            crt_fl<16, 64> mma_reg;
            crt_bf<16, 64> accum, tmp;

            // Adjoint of IFFT stage 2: F @ dy
            warpgroup::mm_AB(mma_reg.real, args.scratch.f.real, args.input.dy[warpgroup::groupid()]);
            warpgroup::mm_AB(mma_reg.imag, args.scratch.f.imag, args.input.dy[warpgroup::groupid()]);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);

            // Adjoint of inverse twiddle: *= conj(twinv_t)
            warpgroup::load(tmp, args.scratch.tw_bwd);
            warp::mul(accum, accum, tmp);

            group<NUM_CONSUMER_WARPS>::sync(2);

            // Adjoint of IFFT stage 1: @ F
            warpgroup::mm_AB(mma_reg, accum, args.scratch.f);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);

            // Adjoint of filter multiply: *= conj(kf)
            warpgroup::load(tmp, args.scratch.kf_conj);
            warp::mul(accum, accum, tmp);

            // Adjoint of FFT stage 2: @ Finv
            warpgroup::mm_AB(mma_reg, accum, args.scratch.finv);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);

            // Adjoint of twiddle: *= conj(tw)
            warpgroup::load(tmp, args.scratch.twinv_t_bwd);
            warp::mul(accum, accum, tmp);

            // Adjoint of FFT stage 1: Finv @ d
            warpgroup::store(args.scratch.tmp[warpgroup::groupid()], accum);
            warpgroup::sync(default_barrer_id);

            warpgroup::mm_AB(mma_reg, args.scratch.finv, args.scratch.tmp[warpgroup::groupid()]);
            warpgroup::mma_async_wait();

            warpgroup::store(args.output.du[warpgroup::groupid()], mma_reg.real);
            warpgroup::sync(default_barrer_id);

            if(laneid() == 0) {
                arrive(args.inputs_finished);
                arrive(args.outputs_arrived);
            }
            __syncwarp();

            int iters_per_head = (args.globals.dy.batch() + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            int next_head = ((args.iter+1) / iters_per_head)*132 + blockIdx.x;
            if(next_head != args.state.current_head) {
                load_head_data(args.scratch, args.globals, next_head);
                args.state.current_head = next_head;
            }
        }
        __device__ static void finish(consumer_finish_args<layout> args) { if(laneid() == 0) arrive(args.finish_finished); }
    };
};


// ============================================================================
// dk_f kernel (filter gradient)
//
// Computes dk_f_partial[b,h] = FFT(dy[b,h]) * conj(FFT(u[b,h]))
// per batch element. Batch reduction (sum over B) is done in Python.
//
// The FFT is the first half of the forward: F @ X → *= tw → @ F
// Both dy and u go through the same FFT.
//
// Layout: input has both dy and u tiles; output has complex dk_f partial.
// ============================================================================

template<int _wg> struct fftconv_dkf_4096_layout {
    static constexpr int wg = _wg;
    using seq_tile      = st_bf<64, 64>;
    using seq_layout    =     gl<bf16, -1, -1, 64, 64, seq_tile>;
    using fft_layout    = cgl<gl<bf16,  1,  1, 64, 64>>;
    // complex output layout: separate real/imag global buffers, same shape as seq
    using out_layout    =     gl<bf16, -1, -1, 64, 64, seq_tile>;
    struct globals {
        out_layout dk_f_real, dk_f_imag;   // output: per-batch dk_f partial (complex)
        seq_layout dy, u;                   // inputs: grad output + original input
        fft_layout f, tw;                   // FFT matrix + twiddle (not conjugated — actual forward FFT)
    };
    struct input_block  { seq_tile dy[wg]; seq_tile u[wg]; };
    struct output_block { seq_tile dk_f_real[wg]; seq_tile dk_f_imag[wg]; };
    struct scratch_block {
        cst_bf<64, 64> f, tw, tmp_dy[2], tmp_u[2];
    };
    struct consumer_state { };
};

struct fft_dkf_4096_template {
    static constexpr int NUM_CONSUMER_WARPS=8, NUM_CONSUMER_WARPGROUPS=NUM_CONSUMER_WARPS/4, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=2, INPUT_PIPE_STAGES=2;
    using layout = fftconv_dkf_4096_layout<NUM_CONSUMER_WARPGROUPS>;

    __device__ static void common_setup(common_setup_args<layout> args) {
        int heads_handled = (args.globals.dy.depth()+131-blockIdx.x) / 132;
        int iters_per_head = (args.globals.dy.batch() + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
        args.num_iters = args.task_iter == 0 ? heads_handled * iters_per_head : -1;
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        }
        __device__ static void load(producer_load_args<layout> args) {
            int iters_per_head = (args.globals.dy.batch() + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * NUM_CONSUMER_WARPGROUPS;
            if(warpgroup::warpid() == args.iter%4) {
                int num_to_load = min((int)NUM_CONSUMER_WARPGROUPS, (int)(args.globals.dy.batch() - batch));
                warp::tma::expect_bytes(args.inputs_arrived,
                    (sizeof(args.input.dy[0]) + sizeof(args.input.u[0])) * num_to_load);
                for(int b = batch; b < batch+NUM_CONSUMER_WARPGROUPS && b < args.globals.dy.batch(); b++) {
                    int idx = b - batch;
                    warp::tma::load_async(args.input.dy[idx], args.globals.dy, {b, head, 0, 0}, args.inputs_arrived);
                    warp::tma::load_async(args.input.u[idx],  args.globals.u,  {b, head, 0, 0}, args.inputs_arrived);
                }
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
                __syncwarp();
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            int iters_per_head = (args.globals.dy.batch() + NUM_CONSUMER_WARPGROUPS-1) / NUM_CONSUMER_WARPGROUPS;
            int head  = (args.iter / iters_per_head)*132 + blockIdx.x;
            int batch = (args.iter % iters_per_head) * NUM_CONSUMER_WARPGROUPS;
            if(warpgroup::warpid() == args.iter%4) {
                for(int b = batch; b < batch+NUM_CONSUMER_WARPGROUPS && b < args.globals.dy.batch(); b++) {
                    int idx = b - batch;
                    warp::tma::store_async(args.globals.dk_f_real, args.output.dk_f_real[idx], {b, head, 0, 0});
                    warp::tma::store_async(args.globals.dk_f_imag, args.output.dk_f_imag[idx], {b, head, 0, 0});
                }
                warp::tma::store_async_read_wait();
                if(laneid() == 0) arrive(args.outputs_finished, 4);
                __syncwarp();
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>();
            using consumers = group<NUM_CONSUMER_WARPS>;
            consumers::load(args.scratch.f,  args.globals.f,  {0, 0, 0, 0});
            consumers::load(args.scratch.tw, args.globals.tw, {0, 0, 0, 0});
        }

        // Helper: apply Monarch FFT to a tile. Returns result in accum.
        // Computes: F @ X → *= tw → @ F
        __device__ static inline void monarch_fft(
            crt_bf<16,64> &accum, crt_fl<16,64> &mma_reg, crt_bf<16,64> &tw_tmp,
            const cst_bf<64,64> &f_smem, const cst_bf<64,64> &tw_smem,
            const st_bf<64,64> &input_tile
        ) {
            // Stage 1: F @ X (left multiply, input is real)
            warpgroup::mm_AB(mma_reg.real, f_smem.real, input_tile);
            warpgroup::mm_AB(mma_reg.imag, f_smem.imag, input_tile);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);

            // Twiddle: *= tw
            warpgroup::load(tw_tmp, tw_smem);
            warp::mul(accum, accum, tw_tmp);

            // Stage 2: @ F (right multiply, complex)
            group<NUM_CONSUMER_WARPS>::sync(2);
            warpgroup::mm_AB(mma_reg, accum, f_smem);
            warpgroup::mma_async_wait();
            warp::copy(accum, mma_reg);
        }

        __device__ static void compute(consumer_compute_args<layout> args) {
            int wgid = warpgroup::warpid()/kittens::WARPGROUP_WARPS;
            int default_barrer_id = wgid + 4;

            crt_fl<16, 64> mma_reg;
            crt_bf<16, 64> fft_dy, fft_u, tw_tmp;

            // FFT(dy)
            monarch_fft(fft_dy, mma_reg, tw_tmp,
                        args.scratch.f, args.scratch.tw,
                        args.input.dy[warpgroup::groupid()]);

            // Save FFT(dy) to scratch (need registers for FFT(u))
            warpgroup::store(args.scratch.tmp_dy[warpgroup::groupid()], fft_dy);
            warpgroup::sync(default_barrer_id);

            // FFT(u)
            monarch_fft(fft_u, mma_reg, tw_tmp,
                        args.scratch.f, args.scratch.tw,
                        args.input.u[warpgroup::groupid()]);

            // Reload FFT(dy)
            warpgroup::load(fft_dy, args.scratch.tmp_dy[warpgroup::groupid()]);

            // dk_f_partial = FFT(dy) * conj(FFT(u))
            // conj(fft_u): negate imaginary part
            // (a+bi)(c-di) = (ac+bd) + (bc-ad)i
            // real = fft_dy.real * fft_u.real + fft_dy.imag * fft_u.imag
            // imag = fft_dy.imag * fft_u.real - fft_dy.real * fft_u.imag
            crt_bf<16, 64> dk_partial;
            rt_bf<16, 64> t1, t2;

            warp::mul(t1, fft_dy.real, fft_u.real);       // ac
            warp::mul(t2, fft_dy.imag, fft_u.imag);       // bd
            warp::add(dk_partial.real, t1, t2);            // ac + bd

            warp::mul(t1, fft_dy.imag, fft_u.real);       // bc
            warp::mul(t2, fft_dy.real, fft_u.imag);       // ad
            warp::sub(dk_partial.imag, t1, t2);            // bc - ad

            // Store dk_f partial (complex) to output
            warpgroup::store(args.output.dk_f_real[warpgroup::groupid()], dk_partial.real);
            warpgroup::store(args.output.dk_f_imag[warpgroup::groupid()], dk_partial.imag);
            warpgroup::sync(default_barrer_id);

            if(laneid() == 0) {
                arrive(args.inputs_finished);
                arrive(args.outputs_arrived);
            }
            __syncwarp();
        }
        __device__ static void finish(consumer_finish_args<layout> args) { if(laneid() == 0) arrive(args.finish_finished); }
    };
};


// ============================================================================
// Template dispatch
// ============================================================================

template<int N> struct fft_bwd_template_internal  { using type = fft_bwd_1024_template; };
template<> struct fft_bwd_template_internal<4096> { using type = fft_bwd_4096_template; };
template<int N> using fft_bwd_template = fft_bwd_template_internal<N>::type;


template<int SEQ> typename fft_bwd_template<SEQ>::layout::globals setup_du_globals(
    bf16 *d_dy, bf16 *d_du,
    bf16 *d_kf_conj_real, bf16 *d_kf_conj_imag,
    bf16 *d_f_real, bf16 *d_f_imag,
    bf16 *d_finv_real, bf16 *d_finv_imag,
    bf16 *d_tw_bwd_real, bf16 *d_tw_bwd_imag,
    bf16 *d_twinv_t_bwd_real, bf16 *d_twinv_t_bwd_imag,
    int B, int H, int N, int N1
) {
    using fftst = fft_bwd_template<SEQ>;
    using globals       = fftst::layout::globals;
    using fft_layout    = fftst::layout::fft_layout;
    using filter_layout = fftst::layout::filter_layout;
    using seq_layout    = fftst::layout::seq_layout;

    seq_layout du_gl{d_du, B, H, nullptr, nullptr};
    seq_layout dy_gl{d_dy, B, H, nullptr, nullptr};

    filter_layout kf_conj_gl{
        typename filter_layout::component{d_kf_conj_real, nullptr, H, nullptr, nullptr},
        typename filter_layout::component{d_kf_conj_imag, nullptr, H, nullptr, nullptr}
    };

    fft_layout f_gl{
        typename fft_layout::component{d_f_real, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::component{d_f_imag, nullptr, nullptr, nullptr, nullptr}
    };
    fft_layout finv_gl{
        typename fft_layout::component{d_finv_real, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::component{d_finv_imag, nullptr, nullptr, nullptr, nullptr}
    };
    fft_layout tw_bwd_gl{
        typename fft_layout::component{d_tw_bwd_real, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::component{d_tw_bwd_imag, nullptr, nullptr, nullptr, nullptr}
    };
    fft_layout twinv_t_bwd_gl{
        typename fft_layout::component{d_twinv_t_bwd_real, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::component{d_twinv_t_bwd_imag, nullptr, nullptr, nullptr, nullptr}
    };

    globals G{
        du_gl,          // output
        dy_gl,          // input
        kf_conj_gl,     // conj(kf)
        f_gl,
        finv_gl,
        tw_bwd_gl,      // conj(twinv_t)
        twinv_t_bwd_gl  // conj(tw)
    };
    return G;
}

#ifdef TK_COMPILE_FFTCONV_BWD
#include <ATen/cuda/CUDAContext.h>
#endif

template<int SEQ>
void launch_du(typename fft_bwd_template<SEQ>::layout::globals G) {
    using fftst = fft_bwd_template<SEQ>;
    unsigned long mem_size = (MAX_SHARED_MEMORY-1024);
    cudaFuncSetAttribute(
        prototype::lcsf::kernel<fftst>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(132);
    dim3 block(prototype::detail::NUM_THREADS_v<fftst>);
#ifdef TK_COMPILE_FFTCONV_BWD
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    prototype::lcsf::kernel<fftst><<<grid, block, mem_size, stream>>>(G);
#else
    prototype::lcsf::kernel<fftst><<<grid, block, mem_size>>>(G);
#endif
}


// ============================================================================
// PyTorch binding
// ============================================================================

#ifdef TK_COMPILE_FFTCONV_BWD
#include "pyutils/torchutils.cuh"
#include <ATen/Functions.h>
#include <iostream>

void dispatch_fftconv_bwd_du(
    bf16 *dy, bf16 *du,
    bf16 *kf_conj_real, bf16 *kf_conj_imag,
    bf16 *f, bf16 *f_imag,
    bf16 *finv, bf16 *finv_imag,
    bf16 *tw_bwd_real, bf16 *tw_bwd_imag,
    bf16 *twinv_t_bwd_real, bf16 *twinv_t_bwd_imag,
    int B, int H, int N, int N1
) {
    if (N == 4096) {
        auto G = setup_du_globals<4096>(
            dy, du,
            kf_conj_real, kf_conj_imag,
            f, f_imag, finv, finv_imag,
            tw_bwd_real, tw_bwd_imag,
            twinv_t_bwd_real, twinv_t_bwd_imag,
            B, H, N, N1
        );
        launch_du<4096>(G);
    } else {
        auto G = setup_du_globals<1024>(
            dy, du,
            kf_conj_real, kf_conj_imag,
            f, f_imag, finv, finv_imag,
            tw_bwd_real, tw_bwd_imag,
            twinv_t_bwd_real, twinv_t_bwd_imag,
            B, H, N, N1
        );
        launch_du<1024>(G);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

/**
 * fftconv_bwd — Backward pass for FFT convolution.
 *
 * The caller is responsible for precomputing the conjugated inputs:
 *   kf_conj     = conj(kf)      = (kf_real, -kf_imag)
 *   tw_bwd      = conj(twinv_t) = (twinv_real, -twinv_imag)  [note: twinv_t was pre-transposed]
 *   twinv_t_bwd = conj(tw)      = (tw_real, -tw_imag)
 *
 * Returns du (input gradient). dk_f (filter gradient) is TODO.
 */
at::Tensor fftconv_bwd(
    const at::Tensor dy_real,
    const at::Tensor kf_conj_real,
    const at::Tensor kf_conj_imag,
    const at::Tensor f_real,
    const at::Tensor f_imag,
    const at::Tensor finv_real,
    const at::Tensor finv_imag,
    const at::Tensor tw_bwd_real,
    const at::Tensor tw_bwd_imag,
    const at::Tensor twinv_t_bwd_real,
    const at::Tensor twinv_t_bwd_imag,
    int B,
    int H,
    int N,
    int N1
) {
    CHECK_INPUT(dy_real);
    CHECK_INPUT(kf_conj_real);
    CHECK_INPUT(kf_conj_imag);
    CHECK_INPUT(f_real);
    CHECK_INPUT(f_imag);
    CHECK_INPUT(finv_real);
    CHECK_INPUT(finv_imag);
    CHECK_INPUT(tw_bwd_real);
    CHECK_INPUT(tw_bwd_imag);
    CHECK_INPUT(twinv_t_bwd_real);
    CHECK_INPUT(twinv_t_bwd_imag);

    TORCH_CHECK(dy_real.size(0) == B, "dy_real has incompatible batch shape");
    TORCH_CHECK(dy_real.size(1) == H, "dy_real has incompatible head shape");
    TORCH_CHECK(dy_real.size(2) == N1, "dy_real has incompatible sequence shape");

    at::Tensor du = at::empty({B, H, N1, N1}, dy_real.options());

    bf16 *d_dy   = reinterpret_cast<bf16*>(dy_real.data_ptr<c10::BFloat16>());
    bf16 *d_du   = reinterpret_cast<bf16*>(du.data_ptr<c10::BFloat16>());
    bf16 *d_kfc_r = reinterpret_cast<bf16*>(kf_conj_real.data_ptr<c10::BFloat16>());
    bf16 *d_kfc_i = reinterpret_cast<bf16*>(kf_conj_imag.data_ptr<c10::BFloat16>());
    bf16 *d_f_r  = reinterpret_cast<bf16*>(f_real.data_ptr<c10::BFloat16>());
    bf16 *d_f_i  = reinterpret_cast<bf16*>(f_imag.data_ptr<c10::BFloat16>());
    bf16 *d_fi_r = reinterpret_cast<bf16*>(finv_real.data_ptr<c10::BFloat16>());
    bf16 *d_fi_i = reinterpret_cast<bf16*>(finv_imag.data_ptr<c10::BFloat16>());
    bf16 *d_twb_r  = reinterpret_cast<bf16*>(tw_bwd_real.data_ptr<c10::BFloat16>());
    bf16 *d_twb_i  = reinterpret_cast<bf16*>(tw_bwd_imag.data_ptr<c10::BFloat16>());
    bf16 *d_twib_r = reinterpret_cast<bf16*>(twinv_t_bwd_real.data_ptr<c10::BFloat16>());
    bf16 *d_twib_i = reinterpret_cast<bf16*>(twinv_t_bwd_imag.data_ptr<c10::BFloat16>());

    dispatch_fftconv_bwd_du(
        d_dy, d_du,
        d_kfc_r, d_kfc_i,
        d_f_r, d_f_i, d_fi_r, d_fi_i,
        d_twb_r, d_twb_i, d_twib_r, d_twib_i,
        B, H, N, N1
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    return du;
}

// ---- dk_f kernel launch ----

void launch_dkf(
    typename fft_dkf_4096_template::layout::globals G
) {
    using fftst = fft_dkf_4096_template;
    unsigned long mem_size = (MAX_SHARED_MEMORY-1024);
    cudaFuncSetAttribute(
        prototype::lcsf::kernel<fftst>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(132);
    dim3 block(prototype::detail::NUM_THREADS_v<fftst>);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    prototype::lcsf::kernel<fftst><<<grid, block, mem_size, stream>>>(G);
}

/**
 * fftconv_bwd_dkf — Filter gradient for FFT convolution.
 *
 * Computes dk_f_partial[b,h] = FFT(dy[b,h]) * conj(FFT(u[b,h]))
 * Returns (dk_f_real, dk_f_imag), each of shape (B, H, N1, N1) in bf16.
 * Caller sums over batch dim B to get final dk_f.
 */
std::vector<at::Tensor> fftconv_bwd_dkf(
    const at::Tensor dy_real,
    const at::Tensor u_real,
    const at::Tensor f_real,
    const at::Tensor f_imag,
    const at::Tensor tw_real,
    const at::Tensor tw_imag,
    int B, int H, int N, int N1
) {
    CHECK_INPUT(dy_real);
    CHECK_INPUT(u_real);
    CHECK_INPUT(f_real);
    CHECK_INPUT(f_imag);
    CHECK_INPUT(tw_real);
    CHECK_INPUT(tw_imag);

    at::Tensor dk_f_real = at::empty({B, H, N1, N1}, dy_real.options());
    at::Tensor dk_f_imag = at::empty({B, H, N1, N1}, dy_real.options());

    using fftst = fft_dkf_4096_template;
    using globals       = fftst::layout::globals;
    using fft_layout    = fftst::layout::fft_layout;
    using seq_layout    = fftst::layout::seq_layout;
    using out_layout    = fftst::layout::out_layout;

    bf16 *d_dy  = reinterpret_cast<bf16*>(dy_real.data_ptr<c10::BFloat16>());
    bf16 *d_u   = reinterpret_cast<bf16*>(u_real.data_ptr<c10::BFloat16>());
    bf16 *d_dkr = reinterpret_cast<bf16*>(dk_f_real.data_ptr<c10::BFloat16>());
    bf16 *d_dki = reinterpret_cast<bf16*>(dk_f_imag.data_ptr<c10::BFloat16>());
    bf16 *d_f_r = reinterpret_cast<bf16*>(f_real.data_ptr<c10::BFloat16>());
    bf16 *d_f_i = reinterpret_cast<bf16*>(f_imag.data_ptr<c10::BFloat16>());
    bf16 *d_tw_r = reinterpret_cast<bf16*>(tw_real.data_ptr<c10::BFloat16>());
    bf16 *d_tw_i = reinterpret_cast<bf16*>(tw_imag.data_ptr<c10::BFloat16>());

    out_layout dkr_gl{d_dkr, (unsigned long)B, (unsigned long)H, nullptr, nullptr};
    out_layout dki_gl{d_dki, (unsigned long)B, (unsigned long)H, nullptr, nullptr};
    seq_layout dy_gl{d_dy, (unsigned long)B, (unsigned long)H, nullptr, nullptr};
    seq_layout u_gl{d_u, (unsigned long)B, (unsigned long)H, nullptr, nullptr};
    fft_layout f_gl{
        typename fft_layout::component{d_f_r, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::component{d_f_i, nullptr, nullptr, nullptr, nullptr}
    };
    fft_layout tw_gl{
        typename fft_layout::component{d_tw_r, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::component{d_tw_i, nullptr, nullptr, nullptr, nullptr}
    };

    globals G{ dkr_gl, dki_gl, dy_gl, u_gl, f_gl, tw_gl };
    launch_dkf(G);

    CHECK_CUDA_ERROR(cudaGetLastError());
    return {dk_f_real, dk_f_imag};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fftconv_bwd", fftconv_bwd,
        "FFTConv backward (du). Takes (dy_real, kf_conj_real, kf_conj_imag, "
        "f_real, f_imag, finv_real, finv_imag, tw_bwd_real, tw_bwd_imag, "
        "twinv_t_bwd_real, twinv_t_bwd_imag, B, H, N, N1). "
        "Caller precomputes conjugated inputs. Returns du in bf16.");
    m.def("fftconv_bwd_dkf", fftconv_bwd_dkf,
        "FFTConv backward (dk_f). Takes (dy_real, u_real, f_real, f_imag, "
        "tw_real, tw_imag, B, H, N, N1). Returns (dk_f_real, dk_f_imag), "
        "each (B,H,N1,N1) in bf16. Caller sums over B for final dk_f.");
}
#else
// TODO: standalone harness
#endif
