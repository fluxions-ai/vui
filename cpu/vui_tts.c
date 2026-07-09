/* VUI TTS: Full C inference - backbone, RQ, codec decoder, tokenizer, WAV output.
 *
 * Build: gcc -O3 -march=native -ffast-math -fopenmp -o vui_tts cpu/vui_tts.c -lm -lopenblas
 * Usage: ./vui_tts vui_full.bin --kv-cache prompt_cache.bin --text "Hello world." -o out.wav
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cblas.h>
#include <pthread.h>

static int g_codec_quiet = 0;  /* suppress per-stage codec timing prints (overlap mode) */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ================================================================== */
/* Config                                                              */
/* ================================================================== */

typedef struct {
    int bb_dim, bb_hidden, bb_layers, bb_heads, bb_kv_heads, bb_max_seq;
    int rq_dim, rq_hidden, rq_layers, rq_heads, rq_n_q, rq_cs;
    int vocab_size, audio_emb_size;
    float rope_theta, eos_bias;
    int sc_token_id;
} VuiConfig;

/* ================================================================== */
/* Weight pointers (all fp32, mmap'd)                                  */
/* ================================================================== */

typedef struct {
    /* backbone per-layer */
    float **bb_attn_norm, **bb_wqkv, **bb_wo;
    float **bb_mlp_norm, **bb_w1, **bb_w2, **bb_w3;
    float *bb_final_norm, *bb_freqs_cis;
    float *codec_head, *eos_head;
    float *token_emb, *audio_emb, *cond_bias;
    /* RQ per-layer */
    float **rq_attn_norm, **rq_wqkv, **rq_wo;
    float **rq_mlp_norm, **rq_w1, **rq_w2, **rq_w3;
    float *rq_final_norm, *rq_code_emb, *rq_pos_emb, *rq_head_W;
    /* Codec decoder */
    float *sem_codebook;        /* [2048, 256] */
    float *sem_out_proj;        /* [512, 256, 1] */
    float *acou_codebooks;      /* [15*2048, 256] */
    float *acou_out_proj;       /* [512, 256, 1] */
    float *pre_conv_w, *pre_conv_b;
    float *pt_input_proj_w, *pt_input_proj_b;
    float **pt_input_ln, **pt_q_proj, **pt_k_proj, **pt_v_proj, **pt_o_proj;
    float **pt_attn_scale, **pt_post_ln;
    float **pt_gate_proj, **pt_up_proj, **pt_down_proj, **pt_mlp_scale;
    float *pt_norm, *pt_output_proj_w, *pt_output_proj_b;
    float *codec_rope;          /* [1024, 32, 2] */
    /* Upsample (2 stages) */
    float *up_tconv_w[2], *up_tconv_b[2];
    float *up_dw_w[2], *up_dw_b[2];
    float *up_ln_w[2], *up_ln_b[2];
    float *up_pw1_w[2], *up_pw1_b[2];
    float *up_pw2_w[2], *up_pw2_b[2];
    float *up_gamma[2];
    /* Waveform decoder */
    float *dec_init_w, *dec_init_b;
    /* 4 decoder blocks: snake + transconv + 3 res units */
    float *dec_snake_a[4], *dec_snake_b[4];
    float *dec_tconv_w[4], *dec_tconv_b[4];
    float *dec_ru_a1[4][3], *dec_ru_b1[4][3];  /* res unit act1 */
    float *dec_ru_c1w[4][3], *dec_ru_c1b[4][3]; /* res unit conv1 */
    float *dec_ru_a2[4][3], *dec_ru_b2[4][3];  /* res unit act2 */
    float *dec_ru_c2w[4][3], *dec_ru_c2b[4][3]; /* res unit conv2 */
    float *dec_final_snake_a, *dec_final_snake_b;
    float *dec_final_conv_w, *dec_final_conv_b;
} Weights;

/* ================================================================== */
/* Runtime buffers                                                     */
/* ================================================================== */

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits;
    float *key_cache, *value_cache;
    int pos;
} BackboneState;

#define RQ_MAX_SEQ 32

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *att;
    float *key_cache, *value_cache;
} RQState;

/* Tokenizer */
typedef struct {
    char **vocab;       /* token strings (byte strings) */
    float *scores;      /* merge priority scores */
    int *lengths;       /* string lengths */
    int vocab_size;
    int max_token_len;
    int byte_offset;
    int special_offset;
    int n_specials;
    char **special_names;
    int *special_ids;
} Tokenizer;

/* Streaming codec decoder state */
#define CODEC_TF_LAYERS 8
#define CODEC_MAX_SEQ 1024
#define CODEC_D_MODEL 512
#define CODEC_ATTN_DIM 1024
#define CODEC_N_HEADS 16
#define CODEC_HEAD_DIM 64
#define CODEC_LATENT 1024

typedef struct {
    /* Pre-conv context buffer: last 2 frames of quantizer output [512, 2] */
    float quant_buf[512 * 2];
    int quant_buf_frames;  /* 0 at start, up to 2 */

    /* Transformer KV cache: [8 layers, n_heads*head_dim, max_seq] */
    float *tf_k_cache;  /* [layers, max_seq, attn_dim] */
    float *tf_v_cache;
    int tf_pos;  /* number of frames processed */

    /* Transformer output history [latent_dim, max_frames] for vocoder context */
    float *tf_out_buf;  /* [CODEC_LATENT, max_frames] */
    int tf_out_frames;
    int tf_out_alloc;
} CodecStreamState;

typedef struct {
    VuiConfig cfg;
    Weights w;
    BackboneState bb;
    RQState rq;
    Tokenizer tok;
    CodecStreamState codec_stream;
    int fd;
    float *data;
    size_t file_size;
} VuiModel;

/* ================================================================== */
/* Neural net primitives                                               */
/* ================================================================== */

static void rmsnorm(float *o, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void layernorm(float *o, const float *x, const float *weight, const float *bias, int size) {
    float mean = 0.0f;
    for (int j = 0; j < size; j++) mean += x[j];
    mean /= size;
    float var = 0.0f;
    for (int j = 0; j < size; j++) { float d = x[j] - mean; var += d * d; }
    var = 1.0f / sqrtf(var / size + 1e-6f);
    for (int j = 0; j < size; j++) o[j] = weight[j] * ((x[j] - mean) * var) + bias[j];
}

static void matmul(float *xout, const float *x, const float *w, int n, int d) {
    /* w is row-major [d, n], compute xout[d] = w @ x[n] */
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d, n, 1.0f, w, n, x, 1, 0.0f, xout, 1);
}

/* Matrix-vector multiply with bias: xout = W @ x + b */
static void matmul_bias(float *xout, const float *x, const float *w, const float *b, int n, int d) {
    memcpy(xout, b, d * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d, n, 1.0f, w, n, x, 1, 1.0f, xout, 1);
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void add_vec(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

/* ================================================================== */
/* Conv1d primitives                                                   */
/* ================================================================== */

/* Causal Conv1d: left-pad, then conv.
 * Input:  [in_ch, T_in]  (row-major, contiguous)
 * Output: [out_ch, T_out] where T_out = T_in (stride=1)
 * Weight: [out_ch, in_ch/groups, kernel]
 * Bias:   [out_ch] or NULL
 *
 * For non-depthwise: uses GEMM approach - reshape conv as matrix multiply.
 * For depthwise (groups==in_ch): direct loop (each channel independent).
 */
static void conv1d_causal(float *out, const float *in, const float *w, const float *b,
                           int in_ch, int out_ch, int kernel, int stride, int dilation,
                           int groups, int T_in) {
    int pad_left = (kernel - 1) * dilation + 1 - stride;
    int T_out = (T_in + pad_left - dilation * (kernel - 1) - 1) / stride + 1;
    int ch_per_group_in = in_ch / groups;
    int ch_per_group_out = out_ch / groups;

    /* k=1, stride=1, no dilation: pure GEMM, no im2col needed */
    if (kernel == 1 && stride == 1 && groups == 1) {
        /* out[out_ch, T] = W[out_ch, in_ch] @ in[in_ch, T] + bias */
        if (b) {
            for (int c = 0; c < out_ch; c++)
                for (int t = 0; t < T_out; t++)
                    out[(size_t)c * T_out + t] = b[c];
        } else {
            memset(out, 0, (size_t)out_ch * T_out * sizeof(float));
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    out_ch, T_out, in_ch,
                    1.0f, w, in_ch, in, T_in,
                    1.0f, out, T_out);
        return;
    }

    if (groups == in_ch && groups == out_ch) {
        /* Depthwise conv: each channel is independent */
        int c;
        #pragma omp parallel for private(c) if(in_ch > 64)
        for (c = 0; c < in_ch; c++) {
            for (int t = 0; t < T_out; t++) {
                float val = b ? b[c] : 0.0f;
                const float *ww = w + (size_t)c * kernel;
                for (int k = 0; k < kernel; k++) {
                    int t_in = t * stride + k * dilation - pad_left;
                    if (t_in >= 0 && t_in < T_in)
                        val += ww[k] * in[(size_t)c * T_in + t_in];
                }
                out[(size_t)c * T_out + t] = val;
            }
        }
        return;
    }

    /* Standard conv via im2col + BLAS GEMM.
     * col layout: [col_K, T_out] (column-major for BLAS)
     * W: [out_ch_per_group, col_K]
     * out = W @ col => [out_ch_per_group, T_out] */
    size_t col_K = (size_t)ch_per_group_in * kernel;
    float *col = malloc(col_K * T_out * sizeof(float));

    for (int g = 0; g < groups; g++) {
        /* Build im2col: col[ic*kernel+k, t] = in[g*cpgi+ic, t*stride+k*dilation-pad] */
        for (int ic = 0; ic < ch_per_group_in; ic++) {
            int in_c = g * ch_per_group_in + ic;
            for (int k = 0; k < kernel; k++) {
                size_t row = (size_t)ic * kernel + k;
                for (int t = 0; t < T_out; t++) {
                    int t_in = t * stride + k * dilation - pad_left;
                    col[row * T_out + t] = (t_in >= 0 && t_in < T_in) ?
                        in[(size_t)in_c * T_in + t_in] : 0.0f;
                }
            }
        }

        /* GEMM: out_g[M, N] = W_g[M, K] @ col[K, N]
         * M = ch_per_group_out, K = col_K, N = T_out
         * Row-major: use CblasRowMajor */
        int out_offset = g * ch_per_group_out;
        const float *W_g = w + (size_t)out_offset * col_K;
        float *out_g = out + (size_t)out_offset * T_out;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    ch_per_group_out, T_out, (int)col_K,
                    1.0f, W_g, (int)col_K,
                    col, T_out,
                    0.0f, out_g, T_out);

        /* Add bias */
        if (b) {
            for (int o = 0; o < ch_per_group_out; o++) {
                int out_c = out_offset + o;
                float bv = b[out_c];
                for (int t = 0; t < T_out; t++)
                    out_g[(size_t)o * T_out + t] += bv;
            }
        }
    }
    free(col);
}

/* Causal Transposed Conv1d via GEMM.
 * Input:  [in_ch, T_in]
 * Output: [out_ch, T_out] where T_out = (T_in - 1) * stride + stride (= T_in * stride)
 * Weight: [in_ch, out_ch, kernel] (PyTorch ConvTranspose1d layout)
 *
 * Strategy: for each t_in, compute all contributions as GEMV, scatter to output.
 * Reformulated: input_T[T_in, in_ch] @ W_reshaped[in_ch, out_ch*kernel] -> patches[T_in, out_ch*kernel]
 * Then scatter patches to output.
 */
static int transconv1d_causal(float *out, const float *in, const float *w, const float *b,
                               int in_ch, int out_ch, int kernel, int stride, int T_in) {
    int T_raw = (T_in - 1) * stride + kernel;
    int trim_right = kernel - stride;
    int T_out = T_raw - trim_right;
    if (!out) return T_out;

    size_t patch_size = (size_t)out_ch * kernel;

    /* Init output with bias */
    for (int o = 0; o < out_ch; o++) {
        float bv = b ? b[o] : 0.0f;
        for (int t = 0; t < T_out; t++)
            out[(size_t)o * T_out + t] = bv;
    }

    /* Batch all timesteps as GEMM:
     * input_T[T_in, in_ch] @ W[in_ch, out_ch*kernel] -> patches[T_in, out_ch*kernel]
     * Then scatter patches to output. */
    float *input_T = malloc((size_t)T_in * in_ch * sizeof(float));
    for (int i = 0; i < in_ch; i++)
        for (int t = 0; t < T_in; t++)
            input_T[(size_t)t * in_ch + i] = in[(size_t)i * T_in + t];

    float *patches = malloc((size_t)T_in * patch_size * sizeof(float));

    /* GEMM: patches[T_in, patch_size] = input_T[T_in, in_ch] @ W[in_ch, patch_size] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                T_in, (int)patch_size, in_ch,
                1.0f, input_T, in_ch,
                w, (int)patch_size,
                0.0f, patches, (int)patch_size);

    /* Scatter patches to output */
    for (int t = 0; t < T_in; t++) {
        float *patch = patches + (size_t)t * patch_size;
        for (int o = 0; o < out_ch; o++) {
            for (int k = 0; k < kernel; k++) {
                int t_out = t * stride + k;
                if (t_out < T_out)
                    out[(size_t)o * T_out + t_out] += patch[(size_t)o * kernel + k];
            }
        }
    }
    free(input_T);
    free(patches);
    return T_out;
}

/* Fast sin approximation: Bhaskara I formula, good to ~0.2% max error.
 * Works for arbitrary x by reducing to [0, pi]. */
static inline float fast_sinf(float x) {
    /* Reduce to [0, 2*pi] */
    static const float TWO_PI = 6.2831853f;
    static const float PI = 3.1415926f;
    x = fmodf(x, TWO_PI);
    if (x < 0) x += TWO_PI;
    /* sin(x) for x in [0, pi]: 16x(pi-x) / (5*pi^2 - 4x(pi-x)) */
    float sign = 1.0f;
    if (x > PI) { x -= PI; sign = -1.0f; }
    float xp = x * (PI - x);
    return sign * 16.0f * xp / (49.348f - 4.0f * xp);  /* 5*pi^2 = 49.348 */
}

/* SnakeBeta activation: x + (1/(exp(beta)+1e-9)) * sin(x * exp(alpha))^2
 * Applied in-place on [dim, T] tensor */
static void snakebeta(float *x, const float *alpha, const float *beta, int dim, int T) {
    int c;
    #pragma omp parallel for private(c)
    for (c = 0; c < dim; c++) {
        float a = expf(alpha[c]);
        float inv_b = 1.0f / (expf(beta[c]) + 1e-9f);
        for (int t = 0; t < T; t++) {
            float v = x[(size_t)c * T + t];
            float s = fast_sinf(v * a);
            x[(size_t)c * T + t] = v + inv_b * s * s;
        }
    }
}

/* GELU activation (approximate) */
static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* ================================================================== */
/* Backbone forward (single token with KV cache)                       */
/* ================================================================== */

static void backbone_forward(VuiModel *m, const float *emb) {
    VuiConfig *c = &m->cfg;
    Weights *w = &m->w;
    BackboneState *s = &m->bb;
    int dim = c->bb_dim, heads = c->bb_heads, kv_heads = c->bb_kv_heads;
    int head_dim = dim / heads, kv_dim = kv_heads * head_dim;
    int hidden = c->bb_hidden, pos = s->pos;

    memcpy(s->x, emb, dim * sizeof(float));

    for (int l = 0; l < c->bb_layers; l++) {
        rmsnorm(s->xb, s->x, w->bb_attn_norm[l], dim);

        float *k_dest = s->key_cache + (size_t)l * c->bb_max_seq * kv_dim + pos * kv_dim;
        float *v_dest = s->value_cache + (size_t)l * c->bb_max_seq * kv_dim + pos * kv_dim;

        matmul(s->q, s->xb, w->bb_wqkv[l], dim, dim);
        matmul(k_dest, s->xb, w->bb_wqkv[l] + (size_t)dim * dim, dim, kv_dim);
        matmul(v_dest, s->xb, w->bb_wqkv[l] + (size_t)(dim + kv_dim) * dim, dim, kv_dim);

        /* RoPE (interleaved) */
        float *freq = w->bb_freqs_cis + pos * head_dim * 2;
        for (int i = 0; i < dim; i += 2) {
            int hd = i % head_dim;
            float fc = freq[hd * 2], fs = freq[hd * 2 + 1];
            float q0 = s->q[i], q1 = s->q[i + 1];
            s->q[i] = q0 * fc - q1 * fs;
            s->q[i + 1] = q0 * fs + q1 * fc;
            if (i < kv_dim) {
                float k0 = k_dest[i], k1 = k_dest[i + 1];
                k_dest[i] = k0 * fc - k1 * fs;
                k_dest[i + 1] = k0 * fs + k1 * fc;
            }
        }

        int kv_mul = heads / kv_heads;
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < heads; h++) {
            float *q_h = s->q + h * head_dim;
            float *att_h = s->att + h * c->bb_max_seq;
            size_t loff = (size_t)l * c->bb_max_seq * kv_dim;
            for (int t = 0; t <= pos; t++) {
                float *k_t = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) score += q_h[i] * k_t[i];
                att_h[t] = score / sqrtf((float)head_dim);
            }
            softmax(att_h, pos + 1);
            float *xb_h = s->xb + h * head_dim;
            memset(xb_h, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v_t = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float a = att_h[t];
                for (int i = 0; i < head_dim; i++) xb_h[i] += a * v_t[i];
            }
        }

        matmul(s->xb2, s->xb, w->bb_wo[l], dim, dim);
        add_vec(s->x, s->xb2, dim);

        rmsnorm(s->xb, s->x, w->bb_mlp_norm[l], dim);
        matmul(s->hb, s->xb, w->bb_w1[l], dim, hidden);
        matmul(s->hb2, s->xb, w->bb_w3[l], dim, hidden);
        for (int i = 0; i < hidden; i++) {
            float val = s->hb[i];
            val *= 1.0f / (1.0f + expf(-val));
            s->hb[i] = val * s->hb2[i];
        }
        matmul(s->xb, s->hb, w->bb_w2[l], hidden, dim);
        add_vec(s->x, s->xb, dim);
    }
    rmsnorm(s->x, s->x, w->bb_final_norm, dim);
    s->pos++;
}

/* ================================================================== */
/* RQ forward with KV cache                                            */
/* ================================================================== */

static void rq_forward_token(VuiModel *m, const float *token_emb, int rq_pos) {
    VuiConfig *c = &m->cfg;
    Weights *w = &m->w;
    RQState *s = &m->rq;
    int dim = c->rq_dim, heads = c->rq_heads;
    int head_dim = dim / heads, hidden = c->rq_hidden;

    memcpy(s->x, token_emb, dim * sizeof(float));

    for (int l = 0; l < c->rq_layers; l++) {
        rmsnorm(s->xb, s->x, w->rq_attn_norm[l], dim);
        float *k_dest = s->key_cache + (size_t)l * RQ_MAX_SEQ * dim + rq_pos * dim;
        float *v_dest = s->value_cache + (size_t)l * RQ_MAX_SEQ * dim + rq_pos * dim;

        matmul(s->q, s->xb, w->rq_wqkv[l], dim, dim);
        matmul(k_dest, s->xb, w->rq_wqkv[l] + (size_t)dim * dim, dim, dim);
        matmul(v_dest, s->xb, w->rq_wqkv[l] + (size_t)2 * dim * dim, dim, dim);

        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < heads; h++) {
            float *q_h = s->q + h * head_dim;
            float *att_h = s->att + h * RQ_MAX_SEQ;
            size_t loff = (size_t)l * RQ_MAX_SEQ * dim;
            for (int t = 0; t <= rq_pos; t++) {
                float *k_t = s->key_cache + loff + t * dim + h * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) score += q_h[i] * k_t[i];
                att_h[t] = score / sqrtf((float)head_dim);
            }
            softmax(att_h, rq_pos + 1);
            float *xb_h = s->xb + h * head_dim;
            memset(xb_h, 0, head_dim * sizeof(float));
            for (int t = 0; t <= rq_pos; t++) {
                float *v_t = s->value_cache + loff + t * dim + h * head_dim;
                float a = att_h[t];
                for (int i = 0; i < head_dim; i++) xb_h[i] += a * v_t[i];
            }
        }

        matmul(s->xb2, s->xb, w->rq_wo[l], dim, dim);
        add_vec(s->x, s->xb2, dim);

        rmsnorm(s->xb, s->x, w->rq_mlp_norm[l], dim);
        matmul(s->hb, s->xb, w->rq_w1[l], dim, hidden);
        matmul(s->hb2, s->xb, w->rq_w3[l], dim, hidden);
        for (int i = 0; i < hidden; i++) {
            float val = s->hb[i];
            val *= 1.0f / (1.0f + expf(-val));
            s->hb[i] = val * s->hb2[i];
        }
        matmul(s->xb, s->hb, w->rq_w2[l], hidden, dim);
        add_vec(s->x, s->xb, dim);
    }
    rmsnorm(s->x, s->x, w->rq_final_norm, dim);
}

static void rq_generate(VuiModel *m, const float *backbone_hidden, int code0,
                         float temperature, int n_quantizers, int *codes_out) {
    VuiConfig *c = &m->cfg;
    Weights *w = &m->w;
    int dim = c->rq_dim, cs = c->rq_cs;
    int Q = n_quantizers < c->rq_n_q ? n_quantizers : c->rq_n_q;

    codes_out[0] = code0;

    float *tok = (float *)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++)
        tok[i] = backbone_hidden[i] + w->rq_pos_emb[i];

    rq_forward_token(m, tok, 0);

    float *ce = w->rq_code_emb + (size_t)code0 * dim;
    for (int i = 0; i < dim; i++)
        tok[i] = ce[i] + w->rq_pos_emb[1 * dim + i];
    rq_forward_token(m, tok, 1);

    float *logits = (float *)malloc(cs * sizeof(float));

    for (int i = 0; i < Q - 1; i++) {
        float *head_w_i = w->rq_head_W + (size_t)i * cs * dim;
        matmul(logits, m->rq.x, head_w_i, dim, cs);

        for (int j = 0; j < cs; j++) logits[j] /= temperature;
        softmax(logits, cs);

        float coin = (float)rand() / (float)RAND_MAX;
        float cdf = 0.0f;
        int next_code = cs - 1;
        for (int j = 0; j < cs; j++) {
            cdf += logits[j];
            if (coin < cdf) { next_code = j; break; }
        }
        codes_out[i + 1] = next_code;

        if (i + 2 < Q) {
            int emb_idx = next_code + (i + 1) * cs;
            float *cem = w->rq_code_emb + (size_t)emb_idx * dim;
            for (int j = 0; j < dim; j++)
                tok[j] = cem[j] + w->rq_pos_emb[(i + 2) * dim + j];
            rq_forward_token(m, tok, i + 2);
        }
    }
    free(tok);
    free(logits);
}

/* ================================================================== */
/* Codec decoder forward                                               */
/* ================================================================== */

/* Quantizer decode: codes [n_frames, 16] -> features [512, n_frames] */
static void codec_quantizer_decode(VuiModel *m, const int *codes, int n_frames,
                                    int n_q, float *out) {
    Weights *w = &m->w;
    int cb_dim = 256, cb_size = 2048, out_dim = 512;
    /* out is [512, n_frames], init to zero */
    memset(out, 0, (size_t)out_dim * n_frames * sizeof(float));

    /* Accumulate codebook embeddings for each frame */
    /* temp buffer for codebook-domain accumulation: [256, n_frames] */
    float *sem_accum = calloc((size_t)cb_dim * n_frames, sizeof(float));
    float *acou_accum = calloc((size_t)cb_dim * n_frames, sizeof(float));

    for (int t = 0; t < n_frames; t++) {
        /* Semantic codebook (index 0) */
        int code0 = codes[t * 16 + 0];
        float *emb = w->sem_codebook + (size_t)code0 * cb_dim;
        for (int d = 0; d < cb_dim; d++)
            sem_accum[(size_t)d * n_frames + t] += emb[d];

        /* Acoustic codebooks (indices 1..n_q-1) */
        for (int q = 1; q < n_q && q < 16; q++) {
            int code = codes[t * 16 + q];
            float *aemb = w->acou_codebooks + (size_t)((q - 1) * cb_size + code) * cb_dim;
            for (int d = 0; d < cb_dim; d++)
                acou_accum[(size_t)d * n_frames + t] += aemb[d];
        }
    }

    /* Output projections: Conv1d(256->512, k=1) applied per timestep */
    /* sem_out_proj: [512, 256, 1] = effectively [512, 256] */
    float *sem_out = calloc((size_t)out_dim * n_frames, sizeof(float));
    float *acou_out = calloc((size_t)out_dim * n_frames, sizeof(float));

    for (int t = 0; t < n_frames; t++) {
        float sem_vec[256], acou_vec[256];
        for (int d = 0; d < cb_dim; d++) {
            sem_vec[d] = sem_accum[(size_t)d * n_frames + t];
            acou_vec[d] = acou_accum[(size_t)d * n_frames + t];
        }
        for (int o = 0; o < out_dim; o++) {
            float sv = 0, av = 0;
            for (int d = 0; d < cb_dim; d++) {
                sv += w->sem_out_proj[(size_t)o * cb_dim + d] * sem_vec[d];
                av += w->acou_out_proj[(size_t)o * cb_dim + d] * acou_vec[d];
            }
            out[(size_t)o * n_frames + t] = sv + av;
        }
    }

    free(sem_accum); free(acou_accum); free(sem_out); free(acou_out);
}

/* Codec pre-transformer: 8-layer transformer with non-interleaved RoPE.
 * Input/output: [n_frames, 512] (row-major, frame-first) */
static void codec_transformer(VuiModel *m, float *x, int T) {
    Weights *w = &m->w;
    int d_model = 512, n_heads = 16, head_dim = 64;
    int attn_dim = n_heads * head_dim;  /* 1024 */
    int mlp_dim = 1024;
    int half_hd = head_dim / 2;

    /* Scratch buffers */
    float *h = malloc((size_t)T * d_model * sizeof(float));
    float *q_buf = malloc((size_t)T * attn_dim * sizeof(float));
    float *k_buf = malloc((size_t)T * attn_dim * sizeof(float));
    float *v_buf = malloc((size_t)T * attn_dim * sizeof(float));
    float *attn_out = malloc((size_t)T * attn_dim * sizeof(float));
    float *proj_out = malloc((size_t)T * d_model * sizeof(float));
    float *gate = malloc((size_t)T * mlp_dim * sizeof(float));
    float *up = malloc((size_t)T * mlp_dim * sizeof(float));

    for (int l = 0; l < 8; l++) {
        /* RMSNorm */
        for (int t = 0; t < T; t++)
            rmsnorm(h + t * d_model, x + t * d_model, w->pt_input_ln[l], d_model);

        /* Q, K, V projections: [T, d_model] @ [d_model, attn_dim]^T => [T, attn_dim] */
        /* W is [attn_dim, d_model] row-major, h is [T, d_model] row-major */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, attn_dim, d_model,
                    1.0f, h, d_model, w->pt_q_proj[l], d_model,
                    0.0f, q_buf, attn_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, attn_dim, d_model,
                    1.0f, h, d_model, w->pt_k_proj[l], d_model,
                    0.0f, k_buf, attn_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, attn_dim, d_model,
                    1.0f, h, d_model, w->pt_v_proj[l], d_model,
                    0.0f, v_buf, attn_dim);

        /* RoPE (non-interleaved): first half and second half */
        for (int t = 0; t < T; t++) {
            float *rope_t = w->codec_rope + t * half_hd * 2;
            for (int nh = 0; nh < n_heads; nh++) {
                float *qt = q_buf + t * attn_dim + nh * head_dim;
                float *kt = k_buf + t * attn_dim + nh * head_dim;
                for (int i = 0; i < half_hd; i++) {
                    float cos_v = rope_t[i * 2];
                    float sin_v = rope_t[i * 2 + 1];
                    /* q_rot[i] = q[i]*cos - q[i+half]*sin */
                    /* q_rot[i+half] = q[i+half]*cos + q[i]*sin */
                    float q_lo = qt[i], q_hi = qt[i + half_hd];
                    qt[i] = q_lo * cos_v - q_hi * sin_v;
                    qt[i + half_hd] = q_hi * cos_v + q_lo * sin_v;
                    float k_lo = kt[i], k_hi = kt[i + half_hd];
                    kt[i] = k_lo * cos_v - k_hi * sin_v;
                    kt[i + half_hd] = k_hi * cos_v + k_lo * sin_v;
                }
            }
        }

        /* Multi-head attention with causal mask */
        float scale = 1.0f / sqrtf((float)head_dim);
        int nh;
        #pragma omp parallel for private(nh)
        for (nh = 0; nh < n_heads; nh++) {
            float *att = malloc(T * T * sizeof(float));
            for (int tq = 0; tq < T; tq++) {
                float *q_t = q_buf + tq * attn_dim + nh * head_dim;
                for (int tk = 0; tk <= tq; tk++) {
                    float *k_t = k_buf + tk * attn_dim + nh * head_dim;
                    float score = 0;
                    for (int i = 0; i < head_dim; i++) score += q_t[i] * k_t[i];
                    att[tq * T + tk] = score * scale;
                }
                for (int tk = tq + 1; tk < T; tk++) att[tq * T + tk] = -1e9f;
                softmax(att + tq * T, T);
                /* Compute attention output */
                float *out_t = attn_out + tq * attn_dim + nh * head_dim;
                memset(out_t, 0, head_dim * sizeof(float));
                for (int tk = 0; tk < T; tk++) {
                    float a = att[tq * T + tk];
                    float *v_t = v_buf + tk * attn_dim + nh * head_dim;
                    for (int i = 0; i < head_dim; i++) out_t[i] += a * v_t[i];
                }
            }
            free(att);
        }

        /* Output projection + attn_scale + residual */
        for (int t = 0; t < T; t++) {
            matmul(proj_out + t * d_model, attn_out + t * attn_dim, w->pt_o_proj[l], attn_dim, d_model);
            float *sc = w->pt_attn_scale[l];
            for (int i = 0; i < d_model; i++)
                x[t * d_model + i] += sc[i] * proj_out[t * d_model + i];
        }

        /* Post-attention norm + MLP */
        for (int t = 0; t < T; t++) {
            rmsnorm(h + t * d_model, x + t * d_model, w->pt_post_ln[l], d_model);
            matmul(gate + t * mlp_dim, h + t * d_model, w->pt_gate_proj[l], d_model, mlp_dim);
            matmul(up + t * mlp_dim, h + t * d_model, w->pt_up_proj[l], d_model, mlp_dim);
            for (int i = 0; i < mlp_dim; i++) {
                float g = gate[t * mlp_dim + i];
                g *= 1.0f / (1.0f + expf(-g)); /* silu */
                gate[t * mlp_dim + i] = g * up[t * mlp_dim + i];
            }
            matmul(proj_out + t * d_model, gate + t * mlp_dim, w->pt_down_proj[l], mlp_dim, d_model);
            float *ms = w->pt_mlp_scale[l];
            for (int i = 0; i < d_model; i++)
                x[t * d_model + i] += ms[i] * proj_out[t * d_model + i];
        }
    }

    /* Final norm */
    for (int t = 0; t < T; t++)
        rmsnorm(x + t * d_model, x + t * d_model, w->pt_norm, d_model);

    free(h); free(q_buf); free(k_buf); free(v_buf);
    free(attn_out); free(proj_out); free(gate); free(up);
}

/* ConvNeXt block: dwconv -> layernorm -> pwconv1 -> gelu -> pwconv2 * gamma + residual
 * Input/output: [dim, T] */
static void convnext_block(float *x, int dim, int T,
                            const float *dw_w, const float *dw_b,
                            const float *ln_w, const float *ln_b,
                            const float *pw1_w, const float *pw1_b,
                            const float *pw2_w, const float *pw2_b,
                            const float *gamma) {
    int mlp_dim = dim * 4;
    float *residual = malloc((size_t)dim * T * sizeof(float));
    memcpy(residual, x, (size_t)dim * T * sizeof(float));

    /* Depthwise causal conv1d (groups=dim, k=7) */
    float *dw_out = malloc((size_t)dim * T * sizeof(float));
    conv1d_causal(dw_out, x, dw_w, dw_b, dim, dim, 7, 1, 1, dim, T);

    /* Transpose to [T, dim], layernorm, pwconv1, gelu, pwconv2 * gamma */
    float *frame = malloc(dim * sizeof(float));
    float *ln_out = malloc(dim * sizeof(float));
    float *pw1_out = malloc(mlp_dim * sizeof(float));
    float *pw2_out = malloc(dim * sizeof(float));

    for (int t = 0; t < T; t++) {
        for (int d = 0; d < dim; d++) frame[d] = dw_out[(size_t)d * T + t];
        layernorm(ln_out, frame, ln_w, ln_b, dim);
        matmul_bias(pw1_out, ln_out, pw1_w, pw1_b, dim, mlp_dim);
        for (int i = 0; i < mlp_dim; i++) pw1_out[i] = gelu(pw1_out[i]);
        matmul_bias(pw2_out, pw1_out, pw2_w, pw2_b, mlp_dim, dim);
        for (int d = 0; d < dim; d++)
            x[(size_t)d * T + t] = residual[(size_t)d * T + t] + gamma[d] * pw2_out[d];
    }

    free(residual); free(dw_out); free(frame); free(ln_out);
    free(pw1_out); free(pw2_out);
}

/* Decoder residual unit: x + conv2(act2(conv1(act1(x)))) */
static void decoder_res_unit(float *x, int dim, int T, int dilation,
                              const float *a1_alpha, const float *a1_beta,
                              const float *c1_w, const float *c1_b,
                              const float *a2_alpha, const float *a2_beta,
                              const float *c2_w, const float *c2_b) {
    size_t sz = (size_t)dim * T;
    float *residual = malloc(sz * sizeof(float));
    memcpy(residual, x, sz * sizeof(float));

    snakebeta(x, a1_alpha, a1_beta, dim, T);
    float *c1_out = malloc(sz * sizeof(float));
    conv1d_causal(c1_out, x, c1_w, c1_b, dim, dim, 7, 1, dilation, 1, T);
    snakebeta(c1_out, a2_alpha, a2_beta, dim, T);
    /* conv2 is k=1 */
    conv1d_causal(x, c1_out, c2_w, c2_b, dim, dim, 1, 1, 1, 1, T);
    add_vec(x, residual, sz);

    free(residual); free(c1_out);
}

/* Full codec decoder: codes [n_frames, 16] -> audio [n_samples] */
static float *codec_decode(VuiModel *m, const int *codes, int n_frames, int n_q, int *out_samples) {
    Weights *w = &m->w;
    int T = n_frames;

    double _ct0, _ct1;
    _ct0 = time_ms();
    /* Step 1: Quantizer decode -> [512, T] */
    float *feat = malloc((size_t)512 * T * sizeof(float));
    codec_quantizer_decode(m, codes, T, n_q, feat);
    _ct1 = time_ms();
    if (!g_codec_quiet) fprintf(stderr, "Codec: quantizer %.0fms\n", _ct1 - _ct0);

    _ct0 = time_ms();
    /* Step 2: Pre-conv [512, T] -> [1024, T] */
    float *pre_conv_out = malloc((size_t)1024 * T * sizeof(float));
    conv1d_causal(pre_conv_out, feat, w->pre_conv_w, w->pre_conv_b, 512, 1024, 3, 1, 1, 1, T);
    free(feat);

    _ct1 = time_ms();
    if (!g_codec_quiet) fprintf(stderr, "Codec: pre_conv %.0fms\n", _ct1 - _ct0);
    _ct0 = time_ms();
    /* Step 3: Transpose [1024, T] -> [T, 1024], input_proj -> [T, 512] */
    float *tf_in = malloc((size_t)T * 512 * sizeof(float));
    for (int t = 0; t < T; t++) {
        float frame[1024];
        for (int d = 0; d < 1024; d++) frame[d] = pre_conv_out[(size_t)d * T + t];
        matmul_bias(tf_in + t * 512, frame, w->pt_input_proj_w, w->pt_input_proj_b, 1024, 512);
    }
    free(pre_conv_out);

    /* Transformer forward [T, 512] -> [T, 512] */
    codec_transformer(m, tf_in, T);

    /* Output proj [T, 512] -> [T, 1024], transpose -> [1024, T] */
    float *tf_out = malloc((size_t)1024 * T * sizeof(float));
    for (int t = 0; t < T; t++) {
        float out_frame[1024];
        matmul_bias(out_frame, tf_in + t * 512, w->pt_output_proj_w, w->pt_output_proj_b, 512, 1024);
        for (int d = 0; d < 1024; d++) tf_out[(size_t)d * T + t] = out_frame[d];
    }
    free(tf_in);

    /* Step 4: Upsample 2x2 = 4x total [1024, T] -> [1024, 4T] */
    _ct1 = time_ms();
    if (!g_codec_quiet) fprintf(stderr, "Codec: transformer %.0fms\n", _ct1 - _ct0);
    _ct0 = time_ms();
    float *h = tf_out;
    int cur_T = T;
    for (int i = 0; i < 2; i++) {
        int new_T = transconv1d_causal(NULL, NULL, NULL, NULL, 0, 0, 2, 2, cur_T);
        float *up_out = malloc((size_t)1024 * new_T * sizeof(float));
        new_T = transconv1d_causal(up_out, h, w->up_tconv_w[i], w->up_tconv_b[i], 1024, 1024, 2, 2, cur_T);
        free(h);
        h = up_out;
        cur_T = new_T;
        convnext_block(h, 1024, cur_T,
                       w->up_dw_w[i], w->up_dw_b[i],
                       w->up_ln_w[i], w->up_ln_b[i],
                       w->up_pw1_w[i], w->up_pw1_b[i],
                       w->up_pw2_w[i], w->up_pw2_b[i],
                       w->up_gamma[i]);
    }

    _ct1 = time_ms();
    if (!g_codec_quiet) fprintf(stderr, "Codec: upsample %.0fms\n", _ct1 - _ct0);
    _ct0 = time_ms();

    /* Initial conv [1024, T] -> [1536, T] */
    int dec_T = cur_T;
    float *dec_h = malloc((size_t)1536 * dec_T * sizeof(float));
    conv1d_causal(dec_h, h, w->dec_init_w, w->dec_init_b, 1024, 1536, 7, 1, 1, 1, dec_T);
    free(h);

    /* 4 decoder blocks */
    int dims_in[] = {1536, 768, 384, 192};
    int dims_out[] = {768, 384, 192, 96};
    int strides[] = {8, 5, 4, 3};
    int dilations[] = {1, 3, 9};

    h = dec_h;
    int cur_dim = 1536;
    for (int bi = 0; bi < 4; bi++) {
        int in_dim = dims_in[bi], out_dim = dims_out[bi], stride = strides[bi];
        int kernel = stride * 2;
        double _bt0 = time_ms();

        /* SnakeBeta activation */
        snakebeta(h, w->dec_snake_a[bi], w->dec_snake_b[bi], in_dim, dec_T);

        /* Transposed conv */
        int new_T = (dec_T - 1) * stride + kernel - (kernel - stride);
        float *tc_out = malloc((size_t)out_dim * new_T * sizeof(float));
        new_T = transconv1d_causal(tc_out, h, w->dec_tconv_w[bi], w->dec_tconv_b[bi],
                                    in_dim, out_dim, kernel, stride, dec_T);
        free(h);
        h = tc_out;
        dec_T = new_T;

        /* 3 residual units */
        for (int ri = 0; ri < 3; ri++) {
            decoder_res_unit(h, out_dim, dec_T, dilations[ri],
                             w->dec_ru_a1[bi][ri], w->dec_ru_b1[bi][ri],
                             w->dec_ru_c1w[bi][ri], w->dec_ru_c1b[bi][ri],
                             w->dec_ru_a2[bi][ri], w->dec_ru_b2[bi][ri],
                             w->dec_ru_c2w[bi][ri], w->dec_ru_c2b[bi][ri]);
        }
        cur_dim = out_dim;
        if (!g_codec_quiet) fprintf(stderr, "  Block %d: %d->%d T=%d %.0fms\n", bi, in_dim, out_dim, dec_T, time_ms() - _bt0);
    }

    /* Final SnakeBeta + conv [96, T] -> [1, T] */
    _ct1 = time_ms();
    if (!g_codec_quiet) fprintf(stderr, "Codec: waveform decoder %.0fms (T=%d)\n", _ct1 - _ct0, dec_T);
    _ct0 = time_ms();
    snakebeta(h, w->dec_final_snake_a, w->dec_final_snake_b, 96, dec_T);
    float *audio = malloc((size_t)dec_T * sizeof(float));
    conv1d_causal(audio, h, w->dec_final_conv_w, w->dec_final_conv_b, 96, 1, 7, 1, 1, 1, dec_T);
    free(h);

    /* Clamp to [-1, 1] */
    for (int i = 0; i < dec_T; i++) {
        if (audio[i] > 1.0f) audio[i] = 1.0f;
        if (audio[i] < -1.0f) audio[i] = -1.0f;
    }

    *out_samples = dec_T;
    return audio;
}

/* ================================================================== */
/* Streaming codec decoder                                             */
/* ================================================================== */

static void codec_stream_init(CodecStreamState *cs, int max_frames) {
    memset(cs->quant_buf, 0, sizeof(cs->quant_buf));
    cs->quant_buf_frames = 0;
    cs->tf_pos = 0;
    size_t kv_size = (size_t)CODEC_TF_LAYERS * CODEC_MAX_SEQ * CODEC_ATTN_DIM;
    cs->tf_k_cache = calloc(kv_size, sizeof(float));
    cs->tf_v_cache = calloc(kv_size, sizeof(float));
    cs->tf_out_alloc = max_frames + 64;
    cs->tf_out_buf = calloc((size_t)CODEC_LATENT * cs->tf_out_alloc, sizeof(float));
    cs->tf_out_frames = 0;
}

static void codec_stream_free(CodecStreamState *cs) {
    free(cs->tf_k_cache); free(cs->tf_v_cache); free(cs->tf_out_buf);
    cs->tf_k_cache = cs->tf_v_cache = cs->tf_out_buf = NULL;
}

/* Incremental transformer: process 1 new frame with KV cache.
 * frame_in: [512] input, frame_out: [512] output. */
static void codec_transformer_step(VuiModel *m, CodecStreamState *cs,
                                    const float *frame_in, float *frame_out) {
    Weights *w = &m->w;
    int d = CODEC_D_MODEL, ad = CODEC_ATTN_DIM, hd = CODEC_HEAD_DIM;
    int nh = CODEC_N_HEADS, half_hd = hd / 2;
    int mlp_dim = 1024;
    int pos = cs->tf_pos;

    float x[CODEC_D_MODEL];
    memcpy(x, frame_in, d * sizeof(float));

    float h[CODEC_D_MODEL], q[CODEC_ATTN_DIM];
    float xb[CODEC_D_MODEL], gate_buf[1024], up_buf[1024], proj[CODEC_D_MODEL];

    for (int l = 0; l < CODEC_TF_LAYERS; l++) {
        rmsnorm(h, x, w->pt_input_ln[l], d);

        /* Q, K, V projections */
        matmul(q, h, w->pt_q_proj[l], d, ad);
        float *k_dest = cs->tf_k_cache + (size_t)l * CODEC_MAX_SEQ * ad + pos * ad;
        float *v_dest = cs->tf_v_cache + (size_t)l * CODEC_MAX_SEQ * ad + pos * ad;
        matmul(k_dest, h, w->pt_k_proj[l], d, ad);
        matmul(v_dest, h, w->pt_v_proj[l], d, ad);

        /* RoPE (non-interleaved) */
        float *rope_t = w->codec_rope + pos * half_hd * 2;
        for (int ih = 0; ih < nh; ih++) {
            float *qt = q + ih * hd;
            float *kt = k_dest + ih * hd;
            for (int i = 0; i < half_hd; i++) {
                float cv = rope_t[i * 2], sv = rope_t[i * 2 + 1];
                float q_lo = qt[i], q_hi = qt[i + half_hd];
                qt[i] = q_lo * cv - q_hi * sv;
                qt[i + half_hd] = q_hi * cv + q_lo * sv;
                float k_lo = kt[i], k_hi = kt[i + half_hd];
                kt[i] = k_lo * cv - k_hi * sv;
                kt[i + half_hd] = k_hi * cv + k_lo * sv;
            }
        }

        /* Causal attention: query at pos, keys/values at 0..pos */
        float attn_out[CODEC_ATTN_DIM];
        float scale = 1.0f / sqrtf((float)hd);
        for (int ih = 0; ih < nh; ih++) {
            float *q_h = q + ih * hd;
            float att[CODEC_MAX_SEQ];
            size_t loff = (size_t)l * CODEC_MAX_SEQ * ad;
            for (int t = 0; t <= pos; t++) {
                float *k_t = cs->tf_k_cache + loff + t * ad + ih * hd;
                float score = 0;
                for (int i = 0; i < hd; i++) score += q_h[i] * k_t[i];
                att[t] = score * scale;
            }
            /* Softmax over 0..pos */
            float mx = att[0];
            for (int t = 1; t <= pos; t++) if (att[t] > mx) mx = att[t];
            float sum = 0;
            for (int t = 0; t <= pos; t++) { att[t] = expf(att[t] - mx); sum += att[t]; }
            float inv_sum = 1.0f / sum;
            float *out_h = attn_out + ih * hd;
            memset(out_h, 0, hd * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float a = att[t] * inv_sum;
                float *v_t = cs->tf_v_cache + loff + t * ad + ih * hd;
                for (int i = 0; i < hd; i++) out_h[i] += a * v_t[i];
            }
        }

        /* Output projection + attn_scale + residual */
        matmul(proj, attn_out, w->pt_o_proj[l], ad, d);
        float *sc = w->pt_attn_scale[l];
        for (int i = 0; i < d; i++) x[i] += sc[i] * proj[i];

        /* Post-attention norm + MLP */
        rmsnorm(h, x, w->pt_post_ln[l], d);
        matmul(gate_buf, h, w->pt_gate_proj[l], d, mlp_dim);
        matmul(up_buf, h, w->pt_up_proj[l], d, mlp_dim);
        for (int i = 0; i < mlp_dim; i++) {
            float g = gate_buf[i];
            g *= 1.0f / (1.0f + expf(-g));
            gate_buf[i] = g * up_buf[i];
        }
        matmul(proj, gate_buf, w->pt_down_proj[l], mlp_dim, d);
        float *ms = w->pt_mlp_scale[l];
        for (int i = 0; i < d; i++) x[i] += ms[i] * proj[i];
    }

    /* Final norm */
    rmsnorm(x, x, w->pt_norm, d);
    memcpy(frame_out, x, d * sizeof(float));
    cs->tf_pos++;
}

/* Run vocoder (upsample + waveform decoder) on a window of transformer outputs.
 * tf_out: [CODEC_LATENT, n_frames] column-major. Returns audio and n_samples. */
static float *vocoder_window(VuiModel *m, const float *tf_out, int n_frames, int *out_samples) {
    Weights *w = &m->w;
    int T = n_frames;

    /* Step 1: Upsample 2x2 = 4x */
    float *h = malloc((size_t)CODEC_LATENT * T * sizeof(float));
    memcpy(h, tf_out, (size_t)CODEC_LATENT * T * sizeof(float));
    int cur_T = T;
    for (int i = 0; i < 2; i++) {
        int new_T = transconv1d_causal(NULL, NULL, NULL, NULL, 0, 0, 2, 2, cur_T);
        float *up_out = malloc((size_t)1024 * new_T * sizeof(float));
        new_T = transconv1d_causal(up_out, h, w->up_tconv_w[i], w->up_tconv_b[i], 1024, 1024, 2, 2, cur_T);
        free(h);
        h = up_out;
        cur_T = new_T;
        convnext_block(h, 1024, cur_T,
                       w->up_dw_w[i], w->up_dw_b[i],
                       w->up_ln_w[i], w->up_ln_b[i],
                       w->up_pw1_w[i], w->up_pw1_b[i],
                       w->up_pw2_w[i], w->up_pw2_b[i],
                       w->up_gamma[i]);
    }

    /* Step 2: Waveform decoder */
    int dec_T = cur_T;
    float *dec_h = malloc((size_t)1536 * dec_T * sizeof(float));
    conv1d_causal(dec_h, h, w->dec_init_w, w->dec_init_b, 1024, 1536, 7, 1, 1, 1, dec_T);
    free(h);

    int dims_in[] = {1536, 768, 384, 192};
    int dims_out[] = {768, 384, 192, 96};
    int strides[] = {8, 5, 4, 3};
    int dilations[] = {1, 3, 9};

    h = dec_h;
    for (int bi = 0; bi < 4; bi++) {
        int in_dim = dims_in[bi], out_dim = dims_out[bi], stride = strides[bi];
        int kernel = stride * 2;
        snakebeta(h, w->dec_snake_a[bi], w->dec_snake_b[bi], in_dim, dec_T);
        int new_T = (dec_T - 1) * stride + kernel - (kernel - stride);
        float *tc_out = malloc((size_t)out_dim * new_T * sizeof(float));
        new_T = transconv1d_causal(tc_out, h, w->dec_tconv_w[bi], w->dec_tconv_b[bi],
                                    in_dim, out_dim, kernel, stride, dec_T);
        free(h);
        h = tc_out;
        dec_T = new_T;
        for (int ri = 0; ri < 3; ri++) {
            decoder_res_unit(h, out_dim, dec_T, dilations[ri],
                             w->dec_ru_a1[bi][ri], w->dec_ru_b1[bi][ri],
                             w->dec_ru_c1w[bi][ri], w->dec_ru_c1b[bi][ri],
                             w->dec_ru_a2[bi][ri], w->dec_ru_b2[bi][ri],
                             w->dec_ru_c2w[bi][ri], w->dec_ru_c2b[bi][ri]);
        }
    }

    snakebeta(h, w->dec_final_snake_a, w->dec_final_snake_b, 96, dec_T);
    float *audio = malloc((size_t)dec_T * sizeof(float));
    conv1d_causal(audio, h, w->dec_final_conv_w, w->dec_final_conv_b, 96, 1, 7, 1, 1, 1, dec_T);
    free(h);

    for (int i = 0; i < dec_T; i++) {
        if (audio[i] > 1.0f) audio[i] = 1.0f;
        if (audio[i] < -1.0f) audio[i] = -1.0f;
    }
    *out_samples = dec_T;
    return audio;
}

#define DOWNSAMPLE_RATE 1920
#define VOCODER_CTX 25

/* Process one new frame through the streaming codec pipeline.
 * codes: [16] int codes for this frame.
 * Returns new audio samples (caller must free), sets *n_samples. */
static float *codec_stream_frame(VuiModel *m, const int *codes, int n_q, int *n_samples) {
    Weights *w = &m->w;
    CodecStreamState *cs = &m->codec_stream;
    int cb_dim = 256, cb_size = 2048, out_dim = 512;

    /* Step 1: Quantizer decode for this single frame -> [512] */
    float feat[512];
    memset(feat, 0, sizeof(feat));
    {
        /* Semantic codebook (index 0) */
        float *emb = w->sem_codebook + (size_t)codes[0] * cb_dim;
        float sem_proj[512];
        cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, cb_dim, 1.0f,
                    w->sem_out_proj, cb_dim, emb, 1, 0.0f, sem_proj, 1);
        for (int d = 0; d < out_dim; d++) feat[d] += sem_proj[d];

        /* Acoustic codebooks */
        float acou_accum[256];
        memset(acou_accum, 0, sizeof(acou_accum));
        for (int q = 1; q < n_q && q < 16; q++) {
            float *aemb = w->acou_codebooks + (size_t)((q - 1) * cb_size + codes[q]) * cb_dim;
            for (int d = 0; d < cb_dim; d++) acou_accum[d] += aemb[d];
        }
        float acou_proj[512];
        cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, cb_dim, 1.0f,
                    w->acou_out_proj, cb_dim, acou_accum, 1, 0.0f, acou_proj, 1);
        for (int d = 0; d < out_dim; d++) feat[d] += acou_proj[d];
    }

    /* Step 2: Pre-conv with context buffer.
     * pre_conv is Conv1d(512, 1024, k=3, causal) => needs 2 prior frames.
     * Build [512, ctx+1] from quant_buf + new frame, convolve, take last frame. */
    int ctx = cs->quant_buf_frames;  /* 0, 1, or 2 */
    int conv_T = ctx + 1;
    float *conv_in = malloc((size_t)512 * conv_T * sizeof(float));
    /* Copy buffered frames (column-major: [512, ctx]) */
    for (int t = 0; t < ctx; t++)
        for (int d = 0; d < 512; d++)
            conv_in[(size_t)d * conv_T + t] = cs->quant_buf[d * 2 + (2 - ctx + t)];
    /* New frame as last column */
    for (int d = 0; d < 512; d++)
        conv_in[(size_t)d * conv_T + ctx] = feat[d];

    /* Update quant_buf: shift left, append new frame */
    if (cs->quant_buf_frames < 2) {
        for (int d = 0; d < 512; d++)
            cs->quant_buf[d * 2 + cs->quant_buf_frames] = feat[d];
        cs->quant_buf_frames++;
    } else {
        for (int d = 0; d < 512; d++) {
            cs->quant_buf[d * 2 + 0] = cs->quant_buf[d * 2 + 1];
            cs->quant_buf[d * 2 + 1] = feat[d];
        }
    }

    float *conv_out = malloc((size_t)1024 * conv_T * sizeof(float));
    conv1d_causal(conv_out, conv_in, w->pre_conv_w, w->pre_conv_b, 512, 1024, 3, 1, 1, 1, conv_T);
    free(conv_in);

    /* Take last frame of conv output: [1024] */
    float conv_frame[1024];
    for (int d = 0; d < 1024; d++)
        conv_frame[d] = conv_out[(size_t)d * conv_T + (conv_T - 1)];
    free(conv_out);

    /* Step 3: input_proj [1024] -> [512], then transformer step */
    float tf_in[512];
    matmul_bias(tf_in, conv_frame, w->pt_input_proj_w, w->pt_input_proj_b, 1024, 512);

    float tf_out_frame[512];
    codec_transformer_step(m, cs, tf_in, tf_out_frame);

    /* output_proj [512] -> [1024] */
    float latent_frame[1024];
    matmul_bias(latent_frame, tf_out_frame, w->pt_output_proj_w, w->pt_output_proj_b, 512, 1024);

    /* Append to transformer output buffer [CODEC_LATENT, n_frames] column-major */
    int fi = cs->tf_out_frames;
    for (int d = 0; d < CODEC_LATENT; d++)
        cs->tf_out_buf[(size_t)d * cs->tf_out_alloc + fi] = latent_frame[d];
    cs->tf_out_frames++;

    /* Step 4: Run vocoder on context window + new frame */
    int voc_ctx = VOCODER_CTX;
    if (voc_ctx > fi) voc_ctx = fi;  /* less context at start */
    int voc_T = voc_ctx + 1;
    int voc_start = cs->tf_out_frames - voc_T;

    /* Build vocoder input [CODEC_LATENT, voc_T] column-major */
    float *voc_in = malloc((size_t)CODEC_LATENT * voc_T * sizeof(float));
    for (int d = 0; d < CODEC_LATENT; d++)
        for (int t = 0; t < voc_T; t++)
            voc_in[(size_t)d * voc_T + t] = cs->tf_out_buf[(size_t)d * cs->tf_out_alloc + (voc_start + t)];

    int total_samples = 0;
    float *full_audio = vocoder_window(m, voc_in, voc_T, &total_samples);
    free(voc_in);

    /* Strip context samples, keep only new frame's audio */
    int ctx_samples = voc_ctx * DOWNSAMPLE_RATE;
    int new_samples = total_samples - ctx_samples;
    if (new_samples <= 0) {
        free(full_audio);
        *n_samples = 0;
        return NULL;
    }
    float *new_audio = malloc(new_samples * sizeof(float));
    memcpy(new_audio, full_audio + ctx_samples, new_samples * sizeof(float));
    free(full_audio);

    *n_samples = new_samples;
    return new_audio;
}

/* ================================================================== */
/* Tokenizer                                                           */
/* ================================================================== */

/* GPT2-style bytes_to_unicode mapping */
static int byte_to_unicode[256];
static int unicode_to_byte[65536];  /* sparse: only 256 entries used */

static void init_byte_unicode_map(void) {
    int bs[256], cs[256], n = 0, count = 0;
    for (int b = '!'; b <= '~'; b++) { bs[count] = b; cs[count] = b; count++; }
    for (int b = 0xA1; b <= 0xAC; b++) { bs[count] = b; cs[count] = b; count++; }
    for (int b = 0xAE; b <= 0xFF; b++) { bs[count] = b; cs[count] = b; count++; }
    int used[256]; memset(used, 0, sizeof(used));
    for (int i = 0; i < count; i++) used[bs[i]] = 1;
    for (int b = 0; b < 256; b++) {
        if (!used[b]) {
            bs[count] = b;
            cs[count] = 256 + n;
            count++;
            n++;
        }
    }
    memset(unicode_to_byte, -1, sizeof(unicode_to_byte));
    for (int i = 0; i < count; i++) {
        byte_to_unicode[bs[i]] = cs[i];
        if (cs[i] < 65536) unicode_to_byte[cs[i]] = bs[i];
    }
}

/* Encode text to token IDs using BPE */
static int *tokenize(Tokenizer *tok, const char *text, int *n_tokens) {
    int text_len = strlen(text);
    /* Start with byte-level tokens */
    int *tokens = malloc((text_len + 16) * sizeof(int));
    int ntok = 0;

    /* Convert each byte to its initial token via bytes_to_unicode lookup */
    for (int i = 0; i < text_len; i++) {
        unsigned char byte = (unsigned char)text[i];
        int unicode_char = byte_to_unicode[byte];
        /* Find this single-char token in vocab */
        int found = -1;
        for (int v = 0; v < tok->vocab_size; v++) {
            if (tok->lengths[v] == 1) {
                /* Token is stored as byte string. For single-char tokens from
                 * bytes_to_unicode, the stored byte might be multi-byte UTF-8. */
                /* Actually, we stored the BYTE string (decoded from unicode).
                 * So for byte 0x20 (space), we stored byte 0x20 as the token string. */
                /* The token string is the actual byte(s) this token represents. */
                if ((unsigned char)tok->vocab[v][0] == byte && tok->lengths[v] == 1) {
                    found = v;
                    break;
                }
            }
        }
        if (found >= 0) {
            tokens[ntok++] = found;
        } else {
            /* Fallback: use byte token */
            tokens[ntok++] = tok->byte_offset + byte;
        }
    }

    /* BPE merge loop: repeatedly find and apply highest-priority merge */
    while (ntok > 1) {
        float best_score = -1e9f;
        int best_idx = -1;
        int best_id = -1;

        for (int i = 0; i < ntok - 1; i++) {
            /* Concatenate token strings */
            int id1 = tokens[i], id2 = tokens[i + 1];
            if (id1 >= tok->vocab_size || id2 >= tok->vocab_size) continue;
            int len1 = tok->lengths[id1], len2 = tok->lengths[id2];
            int total = len1 + len2;
            if (total > tok->max_token_len) continue;

            char merged[256];
            memcpy(merged, tok->vocab[id1], len1);
            memcpy(merged + len1, tok->vocab[id2], len2);

            /* Look up merged string in vocab (linear scan - OK for short texts) */
            for (int v = 0; v < tok->vocab_size; v++) {
                if (tok->lengths[v] == total && tok->scores[v] > best_score) {
                    if (memcmp(tok->vocab[v], merged, total) == 0) {
                        best_score = tok->scores[v];
                        best_idx = i;
                        best_id = v;
                        break;
                    }
                }
            }
        }

        if (best_idx == -1) break;  /* No more merges possible */

        /* Apply merge: replace tokens[best_idx] and tokens[best_idx+1] with best_id */
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < ntok - 1; i++)
            tokens[i] = tokens[i + 1];
        ntok--;
    }

    *n_tokens = ntok;
    return tokens;
}

/* ================================================================== */
/* WAV writer                                                          */
/* ================================================================== */

static void write_wav(const char *path, const float *audio, int n_samples, int sample_rate) {
    FILE *fp = fopen(path, "wb");
    int16_t *pcm = malloc(n_samples * sizeof(int16_t));
    for (int i = 0; i < n_samples; i++) {
        float v = audio[i] * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm[i] = (int16_t)v;
    }

    int data_size = n_samples * 2;
    int file_size = 36 + data_size;
    /* RIFF header */
    fwrite("RIFF", 1, 4, fp);
    int32_t tmp = file_size; fwrite(&tmp, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);
    /* fmt chunk */
    fwrite("fmt ", 1, 4, fp);
    tmp = 16; fwrite(&tmp, 4, 1, fp);
    int16_t s16 = 1; fwrite(&s16, 2, 1, fp); /* PCM */
    s16 = 1; fwrite(&s16, 2, 1, fp);         /* mono */
    tmp = sample_rate; fwrite(&tmp, 4, 1, fp);
    tmp = sample_rate * 2; fwrite(&tmp, 4, 1, fp);
    s16 = 2; fwrite(&s16, 2, 1, fp);         /* block align */
    s16 = 16; fwrite(&s16, 2, 1, fp);        /* bits per sample */
    /* data chunk */
    fwrite("data", 1, 4, fp);
    tmp = data_size; fwrite(&tmp, 4, 1, fp);
    fwrite(pcm, 2, n_samples, fp);
    fclose(fp);
    free(pcm);
}

/* ================================================================== */
/* Model loading                                                       */
/* ================================================================== */

static float **alloc_layer_ptrs(int n) { return (float **)calloc(n, sizeof(float *)); }

static void load_model(VuiModel *m, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(fp, 0, SEEK_END);
    m->file_size = ftell(fp);
    fclose(fp);

    m->fd = open(path, O_RDONLY);
    m->data = mmap(NULL, m->file_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (m->data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }

    uint32_t magic = *(uint32_t *)m->data;
    if (magic != 0x76756932) { fprintf(stderr, "Bad magic: 0x%08x (need vui2)\n", magic); exit(1); }

    int *ihdr = (int *)m->data;
    float *fhdr = (float *)m->data;
    VuiConfig *c = &m->cfg;
    c->bb_dim = ihdr[2]; c->bb_hidden = ihdr[3]; c->bb_layers = ihdr[4];
    c->bb_heads = ihdr[5]; c->bb_kv_heads = ihdr[6]; c->bb_max_seq = ihdr[7];
    c->rq_dim = ihdr[8]; c->rq_hidden = ihdr[9]; c->rq_layers = ihdr[10];
    c->rq_heads = ihdr[11]; c->rq_n_q = ihdr[12]; c->rq_cs = ihdr[13];
    c->vocab_size = ihdr[14]; c->audio_emb_size = ihdr[15];
    c->rope_theta = fhdr[16]; c->eos_bias = fhdr[17];
    c->sc_token_id = ihdr[18];

    fprintf(stderr, "Backbone: d=%d h=%d L=%d heads=%d kv=%d\n",
            c->bb_dim, c->bb_hidden, c->bb_layers, c->bb_heads, c->bb_kv_heads);
    fprintf(stderr, "RQ: d=%d h=%d L=%d heads=%d Q=%d CS=%d\n",
            c->rq_dim, c->rq_hidden, c->rq_layers, c->rq_heads, c->rq_n_q, c->rq_cs);

    float *ptr = (float *)((char *)m->data + 256);
    Weights *w = &m->w;

    int dim = c->bb_dim, hidden = c->bb_hidden;
    int heads = c->bb_heads, kv_heads = c->bb_kv_heads;
    int head_dim = dim / heads, kv_dim = kv_heads * head_dim;
    int qkv_dim = (heads + 2 * kv_heads) * head_dim;

    /* ===== Backbone layers ===== */
    w->bb_attn_norm = alloc_layer_ptrs(c->bb_layers);
    w->bb_wqkv = alloc_layer_ptrs(c->bb_layers);
    w->bb_wo = alloc_layer_ptrs(c->bb_layers);
    w->bb_mlp_norm = alloc_layer_ptrs(c->bb_layers);
    w->bb_w1 = alloc_layer_ptrs(c->bb_layers);
    w->bb_w2 = alloc_layer_ptrs(c->bb_layers);
    w->bb_w3 = alloc_layer_ptrs(c->bb_layers);

    for (int l = 0; l < c->bb_layers; l++) {
        w->bb_attn_norm[l] = ptr; ptr += dim;
        w->bb_wqkv[l] = ptr; ptr += qkv_dim * dim;
        w->bb_wo[l] = ptr; ptr += dim * dim;
        w->bb_mlp_norm[l] = ptr; ptr += dim;
        w->bb_w1[l] = ptr; ptr += hidden * dim;
        w->bb_w2[l] = ptr; ptr += dim * hidden;
        w->bb_w3[l] = ptr; ptr += hidden * dim;
    }
    w->bb_final_norm = ptr; ptr += dim;
    w->bb_freqs_cis = ptr; ptr += c->bb_max_seq * head_dim * 2;

    w->codec_head = ptr; ptr += c->rq_cs * dim;
    w->eos_head = ptr; ptr += 1 * dim;

    w->token_emb = ptr; ptr += c->vocab_size * dim;
    w->audio_emb = ptr; ptr += c->audio_emb_size * dim;
    w->cond_bias = ptr; ptr += dim;

    /* ===== RQ layers ===== */
    int rq_dim = c->rq_dim, rq_hidden = c->rq_hidden;

    w->rq_attn_norm = alloc_layer_ptrs(c->rq_layers);
    w->rq_wqkv = alloc_layer_ptrs(c->rq_layers);
    w->rq_wo = alloc_layer_ptrs(c->rq_layers);
    w->rq_mlp_norm = alloc_layer_ptrs(c->rq_layers);
    w->rq_w1 = alloc_layer_ptrs(c->rq_layers);
    w->rq_w2 = alloc_layer_ptrs(c->rq_layers);
    w->rq_w3 = alloc_layer_ptrs(c->rq_layers);

    for (int l = 0; l < c->rq_layers; l++) {
        w->rq_attn_norm[l] = ptr; ptr += rq_dim;
        w->rq_wqkv[l] = ptr; ptr += 3 * rq_dim * rq_dim;
        w->rq_wo[l] = ptr; ptr += rq_dim * rq_dim;
        w->rq_mlp_norm[l] = ptr; ptr += rq_dim;
        w->rq_w1[l] = ptr; ptr += rq_hidden * rq_dim;
        w->rq_w2[l] = ptr; ptr += rq_dim * rq_hidden;
        w->rq_w3[l] = ptr; ptr += rq_hidden * rq_dim;
    }
    w->rq_final_norm = ptr; ptr += rq_dim;
    w->rq_code_emb = ptr; ptr += (c->rq_n_q - 1) * c->rq_cs * rq_dim;
    w->rq_pos_emb = ptr; ptr += c->rq_n_q * rq_dim;
    w->rq_head_W = ptr; ptr += (c->rq_n_q - 1) * c->rq_cs * rq_dim;

    /* ===== Codec decoder ===== */
    fprintf(stderr, "Loading codec weights...\n");
    int cb_dim = 256, cb_size = 2048;
    w->sem_codebook = ptr; ptr += cb_size * cb_dim;
    w->sem_out_proj = ptr; ptr += 512 * cb_dim * 1;
    w->acou_codebooks = ptr; ptr += 15 * cb_size * cb_dim;
    w->acou_out_proj = ptr; ptr += 512 * cb_dim * 1;

    /* Pre-conv */
    w->pre_conv_w = ptr; ptr += 1024 * 512 * 3;
    w->pre_conv_b = ptr; ptr += 1024;

    /* Pre-transformer */
    w->pt_input_proj_w = ptr; ptr += 512 * 1024;
    w->pt_input_proj_b = ptr; ptr += 512;

    w->pt_input_ln = alloc_layer_ptrs(8);
    w->pt_q_proj = alloc_layer_ptrs(8);
    w->pt_k_proj = alloc_layer_ptrs(8);
    w->pt_v_proj = alloc_layer_ptrs(8);
    w->pt_o_proj = alloc_layer_ptrs(8);
    w->pt_attn_scale = alloc_layer_ptrs(8);
    w->pt_post_ln = alloc_layer_ptrs(8);
    w->pt_gate_proj = alloc_layer_ptrs(8);
    w->pt_up_proj = alloc_layer_ptrs(8);
    w->pt_down_proj = alloc_layer_ptrs(8);
    w->pt_mlp_scale = alloc_layer_ptrs(8);

    for (int l = 0; l < 8; l++) {
        w->pt_input_ln[l] = ptr; ptr += 512;
        w->pt_q_proj[l] = ptr; ptr += 1024 * 512;
        w->pt_k_proj[l] = ptr; ptr += 1024 * 512;
        w->pt_v_proj[l] = ptr; ptr += 1024 * 512;
        w->pt_o_proj[l] = ptr; ptr += 512 * 1024;
        w->pt_attn_scale[l] = ptr; ptr += 512;
        w->pt_post_ln[l] = ptr; ptr += 512;
        w->pt_gate_proj[l] = ptr; ptr += 1024 * 512;
        w->pt_up_proj[l] = ptr; ptr += 1024 * 512;
        w->pt_down_proj[l] = ptr; ptr += 512 * 1024;
        w->pt_mlp_scale[l] = ptr; ptr += 512;
    }
    w->pt_norm = ptr; ptr += 512;
    w->pt_output_proj_w = ptr; ptr += 1024 * 512;
    w->pt_output_proj_b = ptr; ptr += 1024;

    /* Codec RoPE */
    w->codec_rope = ptr; ptr += 1024 * 32 * 2;

    /* Upsample (2 stages) */
    for (int i = 0; i < 2; i++) {
        w->up_tconv_w[i] = ptr; ptr += 1024 * 1024 * 2;
        w->up_tconv_b[i] = ptr; ptr += 1024;
        w->up_dw_w[i] = ptr; ptr += 1024 * 1 * 7;
        w->up_dw_b[i] = ptr; ptr += 1024;
        w->up_ln_w[i] = ptr; ptr += 1024;
        w->up_ln_b[i] = ptr; ptr += 1024;
        w->up_pw1_w[i] = ptr; ptr += 4096 * 1024;
        w->up_pw1_b[i] = ptr; ptr += 4096;
        w->up_pw2_w[i] = ptr; ptr += 1024 * 4096;
        w->up_pw2_b[i] = ptr; ptr += 1024;
        w->up_gamma[i] = ptr; ptr += 1024;
    }

    /* Waveform decoder */
    w->dec_init_w = ptr; ptr += 1536 * 1024 * 7;
    w->dec_init_b = ptr; ptr += 1536;

    int dims_in[] = {1536, 768, 384, 192};
    int dims_out[] = {768, 384, 192, 96};
    int strides[] = {8, 5, 4, 3};

    for (int bi = 0; bi < 4; bi++) {
        int in_dim = dims_in[bi], out_dim = dims_out[bi], stride = strides[bi];
        int kernel = stride * 2;
        w->dec_snake_a[bi] = ptr; ptr += in_dim;
        w->dec_snake_b[bi] = ptr; ptr += in_dim;
        w->dec_tconv_w[bi] = ptr; ptr += in_dim * out_dim * kernel;
        w->dec_tconv_b[bi] = ptr; ptr += out_dim;
        for (int ri = 0; ri < 3; ri++) {
            w->dec_ru_a1[bi][ri] = ptr; ptr += out_dim;
            w->dec_ru_b1[bi][ri] = ptr; ptr += out_dim;
            w->dec_ru_c1w[bi][ri] = ptr; ptr += out_dim * out_dim * 7;
            w->dec_ru_c1b[bi][ri] = ptr; ptr += out_dim;
            w->dec_ru_a2[bi][ri] = ptr; ptr += out_dim;
            w->dec_ru_b2[bi][ri] = ptr; ptr += out_dim;
            w->dec_ru_c2w[bi][ri] = ptr; ptr += out_dim * out_dim * 1;
            w->dec_ru_c2b[bi][ri] = ptr; ptr += out_dim;
        }
    }
    w->dec_final_snake_a = ptr; ptr += 96;
    w->dec_final_snake_b = ptr; ptr += 96;
    w->dec_final_conv_w = ptr; ptr += 1 * 96 * 7;
    w->dec_final_conv_b = ptr; ptr += 1;

    size_t model_bytes = (char *)ptr - (char *)m->data;
    fprintf(stderr, "Model weights: %.1fMB\n", model_bytes / 1e6);

    /* ===== Tokenizer ===== */
    fprintf(stderr, "Loading tokenizer...\n");
    char *tptr = (char *)ptr;
    Tokenizer *tok = &m->tok;

    tok->vocab_size = *(int *)tptr; tptr += 4;
    tok->max_token_len = *(int *)tptr; tptr += 4;

    tok->vocab = calloc(tok->vocab_size, sizeof(char *));
    tok->scores = calloc(tok->vocab_size, sizeof(float));
    tok->lengths = calloc(tok->vocab_size, sizeof(int));

    for (int v = 0; v < tok->vocab_size; v++) {
        tok->scores[v] = *(float *)tptr; tptr += 4;
        int len = *(int *)tptr; tptr += 4;
        tok->vocab[v] = tptr;
        tok->lengths[v] = len;
        tptr += len;
    }

    tok->byte_offset = *(int *)tptr; tptr += 4;
    tok->special_offset = *(int *)tptr; tptr += 4;
    tok->n_specials = *(int *)tptr; tptr += 4;

    tok->special_names = calloc(tok->n_specials, sizeof(char *));
    tok->special_ids = calloc(tok->n_specials, sizeof(int));

    for (int i = 0; i < tok->n_specials; i++) {
        tok->special_ids[i] = *(int *)tptr; tptr += 4;
        int len = *(int *)tptr; tptr += 4;
        tok->special_names[i] = tptr;
        tptr += len;
    }

    size_t total_bytes = (char *)tptr - (char *)m->data;
    fprintf(stderr, "Total loaded: %.1fMB (file: %.1fMB)\n", total_bytes / 1e6, m->file_size / 1e6);

    init_byte_unicode_map();

    /* ===== Allocate runtime buffers ===== */
    BackboneState *bb = &m->bb;
    bb->x = calloc(dim, sizeof(float));
    bb->xb = calloc(dim, sizeof(float));
    bb->xb2 = calloc(dim, sizeof(float));
    bb->hb = calloc(hidden, sizeof(float));
    bb->hb2 = calloc(hidden, sizeof(float));
    bb->q = calloc(dim, sizeof(float));
    bb->att = calloc(heads * c->bb_max_seq, sizeof(float));
    bb->logits = calloc(c->rq_cs, sizeof(float));
    bb->key_cache = calloc((size_t)c->bb_layers * c->bb_max_seq * kv_dim, sizeof(float));
    bb->value_cache = calloc((size_t)c->bb_layers * c->bb_max_seq * kv_dim, sizeof(float));
    bb->pos = 0;

    RQState *rqs = &m->rq;
    rqs->x = calloc(rq_dim, sizeof(float));
    rqs->xb = calloc(rq_dim, sizeof(float));
    rqs->xb2 = calloc(rq_dim, sizeof(float));
    rqs->hb = calloc(rq_hidden, sizeof(float));
    rqs->hb2 = calloc(rq_hidden, sizeof(float));
    rqs->q = calloc(rq_dim, sizeof(float));
    rqs->att = calloc(c->rq_heads * RQ_MAX_SEQ, sizeof(float));
    rqs->key_cache = calloc((size_t)c->rq_layers * RQ_MAX_SEQ * rq_dim, sizeof(float));
    rqs->value_cache = calloc((size_t)c->rq_layers * RQ_MAX_SEQ * rq_dim, sizeof(float));
}

/* ================================================================== */
/* High-level operations                                               */
/* ================================================================== */

static void prefill_token_emb(VuiModel *m, int token_id, int add_cond_bias) {
    int dim = m->cfg.bb_dim;
    float *emb = malloc(dim * sizeof(float));
    memcpy(emb, m->w.token_emb + (size_t)token_id * dim, dim * sizeof(float));
    if (add_cond_bias) add_vec(emb, m->w.cond_bias, dim);
    backbone_forward(m, emb);
    free(emb);
}

/* Predict one frame (code0 + RQ codes 1..Q-1) from an already-computed backbone
 * hidden state. Used for frame-0 (post-text-prefill hidden, no audio input) and
 * as the tail of decode_step for subsequent frames. Matches the reference:
 * inference.py first_codes = rq_sample(codec_head(hidden), hidden). */
static float predict_frame_from_hidden(VuiModel *m, float *hidden, int Q,
                                       float temperature, int *codes_out) {
    int dim = m->cfg.bb_dim, cs = m->cfg.rq_cs;
    float *logits = m->bb.logits;
    matmul(logits, hidden, m->w.codec_head, dim, cs);

    float eos_logit = 0.0f;
    for (int i = 0; i < dim; i++) eos_logit += hidden[i] * m->w.eos_head[i];
    eos_logit += m->cfg.eos_bias;

    for (int i = 0; i < cs; i++) logits[i] /= temperature;
    softmax(logits, cs);

    float coin = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    int code0 = cs - 1;
    for (int i = 0; i < cs; i++) { cdf += logits[i]; if (coin < cdf) { code0 = i; break; } }

    /* Clear RQ KV cache */
    size_t rq_kv_size = (size_t)m->cfg.rq_layers * RQ_MAX_SEQ * m->cfg.rq_dim;
    memset(m->rq.key_cache, 0, rq_kv_size * sizeof(float));
    memset(m->rq.value_cache, 0, rq_kv_size * sizeof(float));

    rq_generate(m, hidden, code0, temperature, Q, codes_out);
    return eos_logit;
}

static float decode_step(VuiModel *m, const int *codes_in, int Q,
                          float temperature, int *codes_out) {
    int dim = m->cfg.bb_dim, cs = m->cfg.rq_cs;

    float *emb = calloc(dim, sizeof(float));
    for (int q = 0; q < Q; q++) {
        int idx = codes_in[q] + q * cs;
        add_vec(emb, m->w.audio_emb + (size_t)idx * dim, dim);
    }
    backbone_forward(m, emb);
    free(emb);

    return predict_frame_from_hidden(m, m->bb.x, Q, temperature, codes_out);
}

/* ================================================================== */
/* KV cache save/load                                                  */
/* ================================================================== */

static void save_kv_cache(VuiModel *m, const char *path) {
    int dim = m->cfg.bb_dim;
    int kv_dim = (m->cfg.bb_kv_heads * dim) / m->cfg.bb_heads;
    size_t kv_size = (size_t)m->cfg.bb_layers * m->cfg.bb_max_seq * kv_dim;
    FILE *fp = fopen(path, "wb");
    int32_t pos = m->bb.pos;
    fwrite(&pos, sizeof(int32_t), 1, fp);
    fwrite(m->bb.key_cache, sizeof(float), kv_size, fp);
    fwrite(m->bb.value_cache, sizeof(float), kv_size, fp);
    fclose(fp);
    fprintf(stderr, "Saved KV cache: pos=%d\n", pos);
}

static void load_kv_cache(VuiModel *m, const char *path) {
    int dim = m->cfg.bb_dim;
    int kv_dim = (m->cfg.bb_kv_heads * dim) / m->cfg.bb_heads;
    size_t kv_size = (size_t)m->cfg.bb_layers * m->cfg.bb_max_seq * kv_dim;
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int32_t pos;
    fread(&pos, sizeof(int32_t), 1, fp);
    fread(m->bb.key_cache, sizeof(float), kv_size, fp);
    fread(m->bb.value_cache, sizeof(float), kv_size, fp);
    fclose(fp);
    m->bb.pos = pos;
    fprintf(stderr, "Loaded KV cache: pos=%d\n", pos);
}

/* ================================================================== */
/* Overlapped codec worker: decode completed chunks (batched codec_decode
 * with `lookback` frames of warmup context) on a separate thread while the
 * main thread keeps generating. codec_decode is purely functional (reads
 * weights, mallocs its own buffers, never writes model state) so this is
 * thread-safe vs concurrent backbone forward.                          */
/* ================================================================== */
#define SAMPLES_PER_FRAME 1920  /* 24kHz / 12.5Hz */
typedef struct {
    VuiModel *m;
    const int *all_codes;
    int Q, lookback, chunk;
    pthread_mutex_t mtx;
    pthread_cond_t cv;
    int frames_ready;   /* frames generated so far (protected) */
    int gen_done;       /* generation finished (protected) */
    float *audio;       /* output buffer */
    int n_samples, audio_cap;
} CodecWorker;

static void *codec_worker_fn(void *arg) {
    CodecWorker *cw = (CodecWorker *)arg;
    int next = 0;  /* first not-yet-emitted frame */
    for (;;) {
        pthread_mutex_lock(&cw->mtx);
        while (cw->frames_ready - next < cw->chunk && !cw->gen_done)
            pthread_cond_wait(&cw->cv, &cw->mtx);
        int ready = cw->frames_ready, done = cw->gen_done;
        pthread_mutex_unlock(&cw->mtx);

        if (next >= ready) { if (done) break; else continue; }

        int chunk_end = done ? ready : next + cw->chunk;
        if (chunk_end > ready) chunk_end = ready;
        int dec_start = next - cw->lookback; if (dec_start < 0) dec_start = 0;
        int discard = next - dec_start;
        int n_dec = chunk_end - dec_start;
        int ns = 0;
        float *ca = codec_decode(cw->m, cw->all_codes + (size_t)dec_start * 16, n_dec, cw->Q, &ns);
        int keep_off = discard * SAMPLES_PER_FRAME;
        int keep = ns - keep_off;
        if (keep > 0 && cw->n_samples + keep <= cw->audio_cap) {
            memcpy(cw->audio + cw->n_samples, ca + keep_off, (size_t)keep * sizeof(float));
            cw->n_samples += keep;
        }
        free(ca);
        next = chunk_end;
        if (next >= ready && done) break;
    }
    return NULL;
}

/* ================================================================== */
/* Main                                                                */
/* ================================================================== */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "VUI TTS - Full C inference\n");
        fprintf(stderr, "Usage: %s model.bin [options]\n", argv[0]);
        fprintf(stderr, "  --text TEXT          Text to synthesize\n");
        fprintf(stderr, "  --kv-cache FILE      Load backbone KV cache\n");
        fprintf(stderr, "  --save-cache FILE    Save KV cache after prefill\n");
        fprintf(stderr, "  --quantizers N       RQ quantizers (default 12)\n");
        fprintf(stderr, "  --temperature F      Sampling temperature (default 0.9)\n");
        fprintf(stderr, "  --max-frames N       Max frames (default 375)\n");
        fprintf(stderr, "  --output FILE        Output WAV (default output.wav)\n");
        fprintf(stderr, "  --tokens T1,T2,...   Pre-tokenized IDs (skip C tokenizer)\n");
        fprintf(stderr, "  --stream             Stream audio to speaker via paplay\n");
        fprintf(stderr, "  --benchmark          Benchmark decode loop\n");
        return 1;
    }

    const char *model_path = argv[1];
    const char *kv_cache_path = NULL, *save_cache_path = NULL;
    const char *text = NULL, *output_path = "output.wav";
    const char *tokens_str = NULL;
    int n_quantizers = 12, max_frames = 375, do_benchmark = 0, do_stream = 0;
    float temperature = 0.9f;
    float eos_threshold = 0.35f;  /* matches reference engine.py n_threshold */
    int min_frames = 6;
    int codec_stream_mode = 0;
    int overlap_mode = 0, overlap_chunk = 24, overlap_lookback = 8;
    int emit_codes = 0;  /* stream generated frame codes to stdout for external (ONNX) codec */
    int chunk_frames = 0;  /* >0: pause for consumer ack every N frames (flow-controlled streaming) */
    int first_chunk = 0;   /* size of the first chunk (low time-to-first-audio); 0 = use chunk_frames */
    int server_mode = 0;   /* stay resident: load once, loop reading text lines from stdin */

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) text = argv[++i];
        else if (strcmp(argv[i], "--eos-threshold") == 0 && i + 1 < argc) eos_threshold = atof(argv[++i]);
        else if (strcmp(argv[i], "--min-frames") == 0 && i + 1 < argc) min_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--codec-stream") == 0) codec_stream_mode = 1;
        else if (strcmp(argv[i], "--overlap") == 0) overlap_mode = 1;
        else if (strcmp(argv[i], "--emit-codes") == 0) emit_codes = 1;
        else if (strcmp(argv[i], "--chunk-frames") == 0 && i + 1 < argc) chunk_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--first-chunk") == 0 && i + 1 < argc) first_chunk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--server") == 0) { server_mode = 1; emit_codes = 1; }
        else if (strcmp(argv[i], "--overlap-chunk") == 0 && i + 1 < argc) overlap_chunk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--overlap-lookback") == 0 && i + 1 < argc) overlap_lookback = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv-cache") == 0 && i + 1 < argc) kv_cache_path = argv[++i];
        else if (strcmp(argv[i], "--save-cache") == 0 && i + 1 < argc) save_cache_path = argv[++i];
        else if (strcmp(argv[i], "--quantizers") == 0 && i + 1 < argc) n_quantizers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) max_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens_str = argv[++i];
        else if (strcmp(argv[i], "--stream") == 0) do_stream = 1;
        else if (strcmp(argv[i], "--benchmark") == 0) do_benchmark = 1;
    }

    srand((unsigned)time(NULL));

    VuiModel model;
    load_model(&model, model_path);

    if (kv_cache_path) load_kv_cache(&model, kv_cache_path);
    if (save_cache_path && !kv_cache_path) save_kv_cache(&model, save_cache_path);

    if (do_benchmark) {
        fprintf(stderr, "\n--- Benchmark (Q=%d) ---\n", n_quantizers);
        int Q = n_quantizers, dim = model.cfg.bb_dim;
        for (int i = 0; i < 12; i++)
            prefill_token_emb(&model, i % model.cfg.vocab_size, 1);

        int *codes_in = calloc(Q, sizeof(int));
        int *codes_out = calloc(Q, sizeof(int));
        int n_steps = 50;
        double t0 = time_ms();
        for (int step = 0; step < n_steps; step++) {
            decode_step(&model, codes_in, Q, temperature, codes_out);
            memcpy(codes_in, codes_out, Q * sizeof(int));
        }
        double elapsed = time_ms() - t0;
        fprintf(stderr, "%d steps in %.0fms = %.1fms/step = %.2fx RTF\n",
                n_steps, elapsed, elapsed / n_steps, 80.0 / (elapsed / n_steps));
        free(codes_in); free(codes_out);
    }

    int prompt_pos = model.bb.pos;   /* KV position right after the voice prompt cache */
    char server_text[8192];
    do {
    if (server_mode) {
        model.bb.pos = prompt_pos;   /* reset KV to post-prompt; new prefill+gen overwrites the rest */
        printf("READY\n"); fflush(stdout);
        if (!fgets(server_text, sizeof(server_text), stdin)) break;
        server_text[strcspn(server_text, "\n")] = 0;
        if (!server_text[0] || strcmp(server_text, "QUIT") == 0) break;
        text = server_text;
    }
    if (text || tokens_str) {
        int *token_ids = NULL;
        int n_tokens = 0;

        if (tokens_str) {
            /* Parse comma-separated token IDs */
            token_ids = malloc(1024 * sizeof(int));
            const char *p = tokens_str;
            while (*p) {
                token_ids[n_tokens++] = atoi(p);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
            fprintf(stderr, "Using %d pre-tokenized IDs\n", n_tokens);
        } else {
            /* C BPE tokenizer */
            token_ids = tokenize(&model.tok, text, &n_tokens);
            fprintf(stderr, "Tokenized '%s' -> %d tokens:", text, n_tokens);
            for (int i = 0; i < n_tokens; i++) fprintf(stderr, " %d", token_ids[i]);
            fprintf(stderr, "\n");
        }

        /* Prefill generation text with cond_bias */
        for (int i = 0; i < n_tokens; i++)
            prefill_token_emb(&model, token_ids[i], 1);
        free(token_ids);

        /* Decode loop */
        int Q = n_quantizers;
        int *codes_in = calloc(Q, sizeof(int));
        int *all_codes = calloc((size_t)max_frames * 16, sizeof(int));
        int n_frames = 0;
        int done = 0;

        /* Streaming: open paplay pipe and decode in chunks */
        FILE *audio_pipe = NULL;
        int chunk_size = 8;  /* decode every N frames */
        int prev_samples = 0;
        double t0 = time_ms();
        double first_audio_ms = 0;

        if (do_stream) {
            audio_pipe = popen("paplay --raw --format=float32le --rate=24000 --channels=1", "w");
            if (!audio_pipe) { fprintf(stderr, "Failed to open paplay\n"); do_stream = 0; }
        }

        /* Overlap: spawn codec worker that decodes completed chunks while we generate. */
        CodecWorker cw; pthread_t cw_thread; int cw_started = 0;
        if (overlap_mode && !do_stream) {
            g_codec_quiet = 1;
            cw.m = &model; cw.all_codes = all_codes; cw.Q = Q;
            cw.lookback = overlap_lookback; cw.chunk = overlap_chunk;
            cw.frames_ready = 0; cw.gen_done = 0; cw.n_samples = 0;
            cw.audio_cap = (max_frames + 16) * SAMPLES_PER_FRAME;
            cw.audio = malloc((size_t)cw.audio_cap * sizeof(float));
            pthread_mutex_init(&cw.mtx, NULL); pthread_cond_init(&cw.cv, NULL);
            pthread_create(&cw_thread, NULL, codec_worker_fn, &cw);
            cw_started = 1;
        }

        fprintf(stderr, "Generating (Q=%d, temp=%.2f)...\n", Q, temperature);
        while (!done && n_frames < max_frames) {
            /* Generate a chunk of frames */
            int chunk_start = n_frames;
            for (int i = 0; i < chunk_size && n_frames < max_frames; i++) {
                int frame_codes[16];
                memset(frame_codes, 0, sizeof(frame_codes));
                /* Frame 0: predict directly from the post-text-prefill hidden state
                 * (no phantom audio frame), matching reference inference.py:1070.
                 * Subsequent frames: embed previous codes + forward. */
                float eos_logit;
                if (n_frames == 0)
                    eos_logit = predict_frame_from_hidden(&model, model.bb.x, Q, temperature, frame_codes);
                else
                    eos_logit = decode_step(&model, codes_in, Q, temperature, frame_codes);
                memcpy(all_codes + n_frames * 16, frame_codes, 16 * sizeof(int));
                n_frames++;

                if (emit_codes) {  /* stream codes to stdout for external ONNX codec */
                    char line[256]; int off = 0;
                    for (int q = 0; q < 16; q++) off += snprintf(line + off, sizeof(line) - off, "%d ", frame_codes[q]);
                    line[off ? off - 1 : 0] = '\n';
                    fwrite(line, 1, off, stdout); fflush(stdout);
                    /* Flow control: after every chunk_frames, emit CHUNK and block until the
                     * consumer (Python) finishes decoding -> gen and codec never run concurrently
                     * (avoids memory-bandwidth contention), pipeline overlaps only with playback. */
                    if (chunk_frames > 0) {
                        int fc = first_chunk > 0 ? first_chunk : chunk_frames;
                        int boundary = (n_frames == fc) ||
                                       (n_frames > fc && ((n_frames - fc) % chunk_frames) == 0);
                        if (boundary) {
                            printf("CHUNK\n"); fflush(stdout);
                            int ch; while ((ch = getchar()) != '\n' && ch != EOF) {}
                        }
                    }
                }

                if (cw_started) {  /* publish new frame to codec worker */
                    pthread_mutex_lock(&cw.mtx);
                    cw.frames_ready = n_frames;
                    pthread_cond_signal(&cw.cv);
                    pthread_mutex_unlock(&cw.mtx);
                }

                float eos_prob = 1.0f / (1.0f + expf(-eos_logit));
                if (eos_prob > eos_threshold && n_frames > min_frames) {
                    fprintf(stderr, "EOS at frame %d (p=%.3f)\n", n_frames - 1, eos_prob);
                    done = 1;
                    break;
                }
                memcpy(codes_in, frame_codes, Q * sizeof(int));
            }

            if (do_stream && n_frames > chunk_start) {
                /* Decode ALL frames so far, output only new samples */
                int total_samples = 0;
                float *audio = codec_decode(&model, all_codes, n_frames, Q, &total_samples);
                int new_samples = total_samples - prev_samples;
                if (new_samples > 0) {
                    if (first_audio_ms == 0) first_audio_ms = time_ms() - t0;
                    fwrite(audio + prev_samples, sizeof(float), new_samples, audio_pipe);
                    fflush(audio_pipe);
                }
                prev_samples = total_samples;
                free(audio);
            }
        }
        double total_ms = time_ms() - t0;
        double audio_s = n_frames / 12.5;

        if (do_stream) {
            if (audio_pipe) pclose(audio_pipe);
            fprintf(stderr, "Streamed %d frames (%.1fs) in %.0fms, first audio at %.0fms\n",
                    n_frames, audio_s, total_ms, first_audio_ms);
        } else if (emit_codes) {
            /* Codes already streamed to stdout per-frame; external ONNX codec decodes. */
            printf("END\n"); fflush(stdout);
            fprintf(stderr, "Generated %d frames (%.1fs) in %.0fms = %.2fx RTF (codes emitted)\n",
                    n_frames, audio_s, total_ms, audio_s / (total_ms / 1000.0));
        } else {
            fprintf(stderr, "Generated %d frames (%.1fs) in %.0fms = %.2fx RTF\n",
                    n_frames, audio_s, total_ms, audio_s / (total_ms / 1000.0));

            double t1 = time_ms();
            int n_samples = 0;
            float *audio;
            if (overlap_mode && cw_started) {
                /* Signal end-of-generation, wait for codec worker to drain. */
                pthread_mutex_lock(&cw.mtx);
                cw.gen_done = 1; pthread_cond_signal(&cw.cv);
                pthread_mutex_unlock(&cw.mtx);
                pthread_join(cw_thread, NULL);
                pthread_mutex_destroy(&cw.mtx); pthread_cond_destroy(&cw.cv);
                audio = cw.audio; n_samples = cw.n_samples;
                double wall = time_ms() - t0;
                fprintf(stderr, "Overlap codec drain: %.0fms after gen; wall %.0fms => %.2fx realtime\n",
                        time_ms() - t1, wall, audio_s / (wall / 1000.0));
            } else if (codec_stream_mode) {
                /* Incremental O(n) decode, frame-by-frame (enabler for gen/codec overlap). */
                codec_stream_init(&model.codec_stream, max_frames);
                int cap = (n_frames + 8) * 1920 * 2, used = 0;  /* 1920 samples/frame @12.5Hz,24kHz */
                audio = malloc((size_t)cap * sizeof(float));
                for (int fr = 0; fr < n_frames; fr++) {
                    int ns = 0;
                    float *chunk = codec_stream_frame(&model, all_codes + fr * 16, Q, &ns);
                    if (chunk && used + ns <= cap) { memcpy(audio + used, chunk, ns * sizeof(float)); used += ns; }
                    free(chunk);
                }
                n_samples = used;
                codec_stream_free(&model.codec_stream);
            } else {
                audio = codec_decode(&model, all_codes, n_frames, Q, &n_samples);
            }
            double codec_ms = time_ms() - t1;
            fprintf(stderr, "Codec decode: %d samples in %.0fms%s\n", n_samples, codec_ms,
                    codec_stream_mode ? " (incremental)" : "");

            write_wav(output_path, audio, n_samples, 24000);
            fprintf(stderr, "Saved %s (%.1fs, 24kHz)\n", output_path, n_samples / 24000.0);
            free(audio);
        }

        free(codes_in);
        free(all_codes);
    }
    } while (server_mode);

    return 0;
}
