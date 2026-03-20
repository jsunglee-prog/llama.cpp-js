#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef GGML_USE_HEXAGON
int main() {
    std::fprintf(stderr, "pi0_infer: GGML was built without Hexagon backend support (GGML_USE_HEXAGON).\n");
    return 1;
}
#else
namespace {

struct pi0_config {
    int seq_len    = 784;
    int dim        = 2048;
    int n_layer    = 18;
    int gqa_ratio  = 8;     // Q heads : KV heads
    int hidden_mul = 8;
    int head_dim   = 64;
    int arena_mb   = 256;

    int n_head     = 0;
    int n_head_kv  = 0;
    int hidden_dim = 0;

    void finalize() {
        if (dim % head_dim != 0) {
            throw std::runtime_error("embedding dim must be divisible by head dim");
        }
        n_head = dim / head_dim;
        n_head_kv = std::max(1, n_head / gqa_ratio);
        hidden_dim = dim * hidden_mul;
    }

    size_t activation_elems() const {
        return static_cast<size_t>(dim) * static_cast<size_t>(seq_len);
    }

    size_t kv_dim() const {
        return static_cast<size_t>(head_dim) * static_cast<size_t>(n_head_kv);
    }

    float attn_scale() const {
        return 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    size_t workspace_bytes() const {
        return static_cast<size_t>(arena_mb) * 1024ULL * 1024ULL;
    }
};

struct layer_weights {
    std::vector<ggml_fp16_t> wq;
    std::vector<ggml_fp16_t> wk;
    std::vector<ggml_fp16_t> wv;
    std::vector<ggml_fp16_t> wo;
    std::vector<ggml_fp16_t> w_up;
    std::vector<ggml_fp16_t> w_down;

    explicit layer_weights(const pi0_config & cfg, std::mt19937 & rng) {
        auto fill = [&](std::vector<ggml_fp16_t> & dst, size_t rows, size_t cols, float scale) {
            std::normal_distribution<float> dist(0.0f, scale);
            dst.resize(rows * cols);
            for (auto & v : dst) {
                v = ggml_fp32_to_fp16(dist(rng));
            }
        };

        const float init_scale = 0.02f;
        fill(wq, cfg.dim, cfg.dim, init_scale);
        fill(wk, cfg.kv_dim(), cfg.dim, init_scale);
        fill(wv, cfg.kv_dim(), cfg.dim, init_scale);
        fill(wo, cfg.dim, cfg.dim, init_scale);
        fill(w_up, cfg.hidden_dim, cfg.dim, init_scale);
        fill(w_down, cfg.dim, cfg.hidden_dim, init_scale);
    }
};

struct layer_runtime {
    explicit layer_runtime(const pi0_config & cfg) {
        arena.resize(cfg.workspace_bytes());
    }

    std::vector<uint8_t> arena;
};

static void usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n\n"
        "Options:\n"
        "  --layers N        Number of transformer layers (default 18)\n"
        "  --seq N           Sequence length / tokens (default 784)\n"
        "  --dim N           Embedding dimension (default 2048)\n"
        "  --gqa R           GQA ratio Q:KV (default 8)\n"
        "  --head-dim N      Per-head dimension (default 64)\n"
        "  --hidden-mult N   MLP hidden multiplier (default 8)\n"
        "  --arena-mb N      Scratch size per layer in MB (default 256)\n"
        "  --seed N          RNG seed (default 42)\n"
        "  --help            Show this message\n",
        prog);
}

static bool parse_int_arg(const char * label, const char * value, int & dst) {
    try {
        dst = std::stoi(value);
        return true;
    } catch (...) {
        std::fprintf(stderr, "pi0_infer: invalid value for %s -> %s\n", label, value);
        return false;
    }
}

static void fill_random(std::vector<ggml_fp16_t> & dst, std::mt19937 & rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & v : dst) {
        v = ggml_fp32_to_fp16(dist(rng));
    }
}

static ggml_tensor * make_weight(
        ggml_context * ctx,
        const std::vector<ggml_fp16_t> & data,
        int64_t rows,
        int64_t cols,
        const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cols, rows);
    std::memcpy(ggml_get_data(t), data.data(), data.size() * sizeof(ggml_fp16_t));
    ggml_set_name(t, name);
    return t;
}

static ggml_tensor * build_attention(
        ggml_context * ctx,
        const pi0_config & cfg,
        ggml_tensor * x,
        ggml_tensor * wq,
        ggml_tensor * wk,
        ggml_tensor * wv,
        ggml_tensor * wo) {
    ggml_tensor * q = ggml_mul_mat(ctx, wq, x);
    ggml_tensor * k = ggml_mul_mat(ctx, wk, x);
    ggml_tensor * v = ggml_mul_mat(ctx, wv, x);

    q = ggml_cast(ctx, q, GGML_TYPE_F16);
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
    v = ggml_cast(ctx, v, GGML_TYPE_F16);

    ggml_tensor * q_heads = ggml_reshape_4d(ctx, q, cfg.head_dim, cfg.seq_len, cfg.n_head, 1);
    ggml_tensor * k_heads = ggml_reshape_4d(ctx, k, cfg.head_dim, cfg.seq_len, cfg.n_head_kv, 1);
    ggml_tensor * v_heads = ggml_reshape_4d(ctx, v, cfg.head_dim, cfg.seq_len, cfg.n_head_kv, 1);

    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q_heads, k_heads, v_heads, nullptr, cfg.attn_scale(), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    attn = ggml_permute(ctx, attn, 0, 2, 1, 3);
    attn = ggml_cont(ctx, attn);
    attn = ggml_reshape_2d(ctx, attn, cfg.dim, cfg.seq_len);
    attn = ggml_cast(ctx, attn, GGML_TYPE_F16);

    ggml_tensor * proj = ggml_mul_mat(ctx, wo, attn);
    proj = ggml_cast(ctx, proj, GGML_TYPE_F16);
    proj = ggml_add(ctx, proj, x);
    ggml_set_name(proj, "attn_out");
    return proj;
}

static ggml_tensor * build_mlp(
        ggml_context * ctx,
        const pi0_config & cfg,
        ggml_tensor * x,
        ggml_tensor * w_up,
        ggml_tensor * w_down) {
    ggml_tensor * up = ggml_mul_mat(ctx, w_up, x);
    up = ggml_cast(ctx, up, GGML_TYPE_F16);
    up = ggml_silu(ctx, up);

    ggml_tensor * down = ggml_mul_mat(ctx, w_down, up);
    down = ggml_cast(ctx, down, GGML_TYPE_F16);

    ggml_tensor * out = ggml_add(ctx, x, down);
    ggml_set_name(out, "mlp_out");
    ggml_set_output(out);
    return out;
}

static bool run_layer(
        const pi0_config & cfg,
        ggml_backend_t backend,
        layer_runtime & workspace,
        const layer_weights & weights,
        std::vector<ggml_fp16_t> & state) {
    ggml_init_params params {
        /*.mem_size   =*/ workspace.arena.size(),
        /*.mem_buffer =*/ workspace.arena.data(),
        /*.no_alloc   =*/ false,
    };

    ggml_time_init();

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "pi0_infer: failed to init ggml context for layer\n");
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cfg.dim, cfg.seq_len);
    std::memcpy(ggml_get_data(x), state.data(), state.size() * sizeof(ggml_fp16_t));
    ggml_set_name(x, "activations_in");

    ggml_tensor * wq = make_weight(ctx, weights.wq, cfg.dim, cfg.dim, "wq");
    ggml_tensor * wk = make_weight(ctx, weights.wk, cfg.kv_dim(), cfg.dim, "wk");
    ggml_tensor * wv = make_weight(ctx, weights.wv, cfg.kv_dim(), cfg.dim, "wv");
    ggml_tensor * wo = make_weight(ctx, weights.wo, cfg.dim, cfg.dim, "wo");
    ggml_tensor * w_up = make_weight(ctx, weights.w_up, cfg.hidden_dim, cfg.dim, "w_up");
    ggml_tensor * w_down = make_weight(ctx, weights.w_down, cfg.dim, cfg.hidden_dim, "w_down");

    ggml_tensor * attn = build_attention(ctx, cfg, x, wq, wk, wv, wo);
    ggml_tensor * out = build_mlp(ctx, cfg, attn, w_up, w_down);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    const int n_nodes = graph->n_nodes;
    int out_index = -1;
    for (int i = 0; i < n_nodes; ++i) {
        if (graph->nodes[i] == out) {
            out_index = i;
            break;
        }
    }
    if (out_index == -1) {
        ggml_free(ctx);
        std::fprintf(stderr, "pi0_infer: failed to locate output node\n");
        return false;
    }

    auto graph_copy = ggml_backend_graph_copy(backend, graph);
    if (graph_copy.graph == nullptr) {
        ggml_free(ctx);
        std::fprintf(stderr, "pi0_infer: backend graph copy failed\n");
        return false;
    }

    ggml_tensor * out_dev = graph_copy.graph->nodes[out_index];

    if (ggml_backend_graph_compute(backend, graph_copy.graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_graph_copy_free(graph_copy);
        ggml_free(ctx);
        std::fprintf(stderr, "pi0_infer: backend compute failed\n");
        return false;
    }

    ggml_backend_synchronize(backend);

    ggml_backend_tensor_get(out_dev, state.data(), 0, state.size() * sizeof(ggml_fp16_t));

    ggml_backend_graph_copy_free(graph_copy);
    ggml_free(ctx);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    pi0_config cfg;
    uint64_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--layers" && i + 1 < argc) {
            if (!parse_int_arg("--layers", argv[++i], cfg.n_layer)) {
                return 1;
            }
        } else if (arg == "--seq" && i + 1 < argc) {
            if (!parse_int_arg("--seq", argv[++i], cfg.seq_len)) {
                return 1;
            }
        } else if (arg == "--dim" && i + 1 < argc) {
            if (!parse_int_arg("--dim", argv[++i], cfg.dim)) {
                return 1;
            }
        } else if (arg == "--gqa" && i + 1 < argc) {
            if (!parse_int_arg("--gqa", argv[++i], cfg.gqa_ratio)) {
                return 1;
            }
        } else if (arg == "--head-dim" && i + 1 < argc) {
            if (!parse_int_arg("--head-dim", argv[++i], cfg.head_dim)) {
                return 1;
            }
        } else if (arg == "--hidden-mult" && i + 1 < argc) {
            if (!parse_int_arg("--hidden-mult", argv[++i], cfg.hidden_mul)) {
                return 1;
            }
        } else if (arg == "--arena-mb" && i + 1 < argc) {
            if (!parse_int_arg("--arena-mb", argv[++i], cfg.arena_mb)) {
                return 1;
            }
        } else if (arg == "--seed" && i + 1 < argc) {
            try {
                seed = static_cast<uint64_t>(std::stoll(argv[++i]));
            } catch (...) {
                std::fprintf(stderr, "pi0_infer: invalid seed value\n");
                return 1;
            }
        } else if (arg == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "pi0_infer: unknown option %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    try {
        cfg.finalize();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "pi0_infer: invalid configuration - %s\n", e.what());
        return 1;
    }

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_hexagon_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "pi0_infer: failed to initialize Hexagon backend. Check DSP availability.\n");
        return 1;
    }

    std::mt19937 rng(static_cast<uint32_t>(seed));

    std::vector<ggml_fp16_t> state(cfg.activation_elems());
    fill_random(state, rng);

    layer_runtime runtime(cfg);

    double total_us = 0.0;

    for (int il = 0; il < cfg.n_layer; ++il) {
        layer_weights weights(cfg, rng);
        const int64_t t_start = ggml_time_us();
        if (!run_layer(cfg, backend, runtime, weights, state)) {
            ggml_backend_free(backend);
            return 1;
        }
        const int64_t t_end = ggml_time_us();
        total_us += static_cast<double>(t_end - t_start);
        std::printf("layer %02d finished in %.2f ms\n", il, (t_end - t_start) / 1000.0);
    }

    std::vector<float> preview(8);
    for (size_t i = 0; i < preview.size() && i < state.size(); ++i) {
        preview[i] = ggml_fp16_to_fp32(state[i]);
    }

    double rms = 0.0;
    for (ggml_fp16_t v : state) {
        const float f = ggml_fp16_to_fp32(v);
        rms += static_cast<double>(f) * static_cast<double>(f);
    }
    rms = std::sqrt(rms / static_cast<double>(state.size()));

    std::printf("\nSummary:\n");
    std::printf("  layers          : %d\n", cfg.n_layer);
    std::printf("  sequence length : %d\n", cfg.seq_len);
    std::printf("  embedding dim   : %d\n", cfg.dim);
    std::printf("  heads (Q / KV)  : %d / %d\n", cfg.n_head, cfg.n_head_kv);
    std::printf("  hidden dim      : %d\n", cfg.hidden_dim);
    std::printf("  runtime         : %.2f ms total (%.2f ms/layer)\n",
                total_us / 1000.0,
                (total_us / 1000.0) / static_cast<double>(cfg.n_layer));
    std::printf("  output RMS      : %.6f\n", rms);
    std::printf("  sample values   : ");
    for (float v : preview) {
        std::printf("%.4f ", v);
    }
    std::printf("\n");

    ggml_backend_free(backend);
    return 0;
}
#endif // GGML_USE_HEXAGON
