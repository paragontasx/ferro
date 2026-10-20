// Derived from llama.cpp tools/mtmd (MIT, Copyright (c) The ggml authors). See NOTICE.
// mm_preproc.cpp — see mm_preproc.h. Ported from llama.cpp tools/mtmd/mtmd-image.cpp and mtmd-audio.cpp
// (MIT, (c) the ggml authors); arithmetic kept identical so the encoder inputs match bit for bit.
#include "mm_preproc.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ferro {

// ============================================================ images
// Pillow-compatible separable resampling (bilinear), 22-bit fixed point
static void resize_pillow(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, bool bicubic) {
    const int PRECISION_BITS = 32 - 8 - 2;
    const double filter_support = bicubic ? 2.0 : 1.0;
    auto filt = [bicubic](double x) {
        if (x < 0.0) x = -x;
        if (!bicubic) return x < 1.0 ? 1.0 - x : 0.0;
        constexpr double a = -0.5;                       // Pillow's bicubic (GGML/PyTorch use -0.75)
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    auto clip8 = [](int v) -> uint8_t { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; };
    auto precompute = [&](int inSize, int outSize, std::vector<int>& bounds, std::vector<int32_t>& weights) {
        double scale, filterscale;
        filterscale = scale = (double)inSize / outSize;
        if (filterscale < 1.0) filterscale = 1.0;
        const double support = filter_support * filterscale;
        const int ksize = (int)std::ceil(support) * 2 + 1;
        std::vector<double> pre((size_t)outSize * ksize);
        bounds.resize((size_t)outSize * 2);
        for (int xx = 0; xx < outSize; xx++) {
            const double center = (xx + 0.5) * scale;
            double ww = 0.0;
            const double ss = 1.0 / filterscale;
            int xmin = (int)(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = (int)(center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            xmax -= xmin;
            int x;
            for (x = 0; x < xmax; x++) {
                const double w = filt((x + xmin - center + 0.5) * ss);
                pre[(size_t)xx * ksize + x] = w;
                ww += w;
            }
            for (x = 0; x < xmax; x++) if (ww != 0.0) pre[(size_t)xx * ksize + x] /= ww;
            for (; x < ksize; x++) pre[(size_t)xx * ksize + x] = 0;
            bounds[xx * 2 + 0] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        weights.resize((size_t)outSize * ksize);
        const double fxp = std::ldexp(1.0, PRECISION_BITS);
        for (size_t i = 0; i < (size_t)outSize * ksize; i++)
            weights[i] = (int32_t)(pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5));
        return ksize;
    };
    auto horiz = [&](const uint8_t* s, int in_nx, int in_ny, int out_nx, int ksize, const std::vector<int>& b, const std::vector<int32_t>& w) {
        std::vector<uint8_t> out((size_t)out_nx * in_ny * 3);
        for (int yy = 0; yy < in_ny; yy++) {
            const uint8_t* sr = s + (size_t)yy * in_nx * 3;
            uint8_t* dr = out.data() + (size_t)yy * out_nx * 3;
            for (int xx = 0; xx < out_nx; xx++) {
                const int xmin = b[xx * 2], xcnt = b[xx * 2 + 1];
                const int32_t* k = &w[(size_t)xx * ksize];
                const uint8_t* p = sr + (size_t)xmin * 3;
                int32_t s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xcnt; x++) { s0 += p[0] * k[x]; s1 += p[1] * k[x]; s2 += p[2] * k[x]; p += 3; }
                dr[xx * 3 + 0] = clip8(s0 >> PRECISION_BITS);
                dr[xx * 3 + 1] = clip8(s1 >> PRECISION_BITS);
                dr[xx * 3 + 2] = clip8(s2 >> PRECISION_BITS);
            }
        }
        return out;
    };
    auto vert = [&](const uint8_t* s, int in_nx, int out_ny, int ksize, const std::vector<int>& b, const std::vector<int32_t>& w) {
        const size_t re = (size_t)in_nx * 3;
        std::vector<uint8_t> out(re * out_ny);
        std::vector<int32_t> acc(re);
        for (int yy = 0; yy < out_ny; yy++) {
            const int ymin = b[yy * 2], ycnt = b[yy * 2 + 1];
            const int32_t* k = &w[(size_t)yy * ksize];
            std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));
            for (int y = 0; y < ycnt; y++) {
                const uint8_t* sr = s + (size_t)(ymin + y) * re;
                const int32_t wy = k[y];
                for (size_t i = 0; i < re; i++) acc[i] += sr[i] * wy;
            }
            uint8_t* dr = out.data() + (size_t)yy * re;
            for (size_t i = 0; i < re; i++) dr[i] = clip8(acc[i] >> PRECISION_BITS);
        }
        return out;
    };
    if (dw <= 0 || dw > 65536 || dh <= 0 || dh > 65536) throw std::invalid_argument("resize target out of range");
    const bool nh = dw != sw, nv = dh != sh;
    std::vector<int> bh, bv;
    std::vector<int32_t> wh, wv;
    int kh = 0, kv = 0;
    if (nh) kh = precompute(sw, dw, bh, wh);
    if (nv) kv = precompute(sh, dh, bv, wv);
    std::vector<uint8_t> out;
    if (nh && nv) { auto t = horiz(src, sw, sh, dw, kh, bh, wh); out = vert(t.data(), dw, dh, kv, bv, wv); }
    else if (nh) out = horiz(src, sw, sh, dw, kh, bh, wh);
    else if (nv) out = vert(src, sw, dh, kv, bv, wv);
    else out.assign(src, src + (size_t)sw * sh * 3);
    std::copy(out.begin(), out.end(), dst);
}

void resize_pillow_bilinear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) { resize_pillow(src, sw, sh, dst, dw, dh, false); }
void resize_pillow_bicubic(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) { resize_pillow(src, sw, sh, dst, dw, dh, true); }

namespace {
struct Size { int w = 0, h = 0; };

// img_tool::calc_size_preserved_ratio (no longest_edge)
Size calc_size_preserved_ratio(Size in, int align, int min_pixels, int max_pixels) {
    const int width = in.w, height = in.h;
    if (width <= 0 || height <= 0) return {0, 0};
    auto round_by = [align](float x) { return (int)std::round(x / (float)align) * align; };
    auto ceil_by = [align](float x) { return (int)std::ceil(x / (float)align) * align; };
    auto floor_by = [align](float x) { return (int)std::floor(x / (float)align) * align; };
    int w_bar = std::max(align, round_by((float)width));
    int h_bar = std::max(align, round_by((float)height));
    if (max_pixels > 0 && h_bar * w_bar > max_pixels) {
        const float beta = std::sqrt((float)height * width / max_pixels);
        h_bar = std::max(align, floor_by(height / beta));
        w_bar = std::max(align, floor_by(width / beta));
    } else if (min_pixels > 0 && h_bar * w_bar < min_pixels) {
        const float beta = std::sqrt((float)min_pixels / ((float)height * width));
        h_bar = ceil_by(height * beta);
        w_bar = ceil_by(width * beta);
    }
    return {w_bar, h_bar};
}

constexpr int MIN_TILES = 2, MAX_TILES = 10, TILE = 512;
constexpr float MAX_PIXELS_TOLERANCE = 2.0f;

bool should_tile(const VisionPreParams& p, Size orig) {
    const int align = p.patch_size * p.n_merge;
    auto round_by = [align](float x) { return (int)std::nearbyint((double)x / align) * align; };
    const int h_bar = std::max(p.patch_size, round_by((float)orig.h));
    const int w_bar = std::max(p.patch_size, round_by((float)orig.w));
    return (double)h_bar * (double)w_bar > (double)p.max_pixels * MAX_PIXELS_TOLERANCE;
}

std::vector<Size> target_ratios() {
    std::vector<Size> r;
    for (int n = MIN_TILES; n <= MAX_TILES; n++)
        for (int w = 1; w <= n; w++)
            for (int h = 1; h <= n; h++)
                if (w * h >= MIN_TILES && w * h <= MAX_TILES) {
                    bool found = false;
                    for (auto& x : r) if (x.w == w && x.h == h) { found = true; break; }
                    if (!found) r.push_back({w, h});
                }
    std::sort(r.begin(), r.end(), [](const Size& a, const Size& b) { return a.w * a.h < b.w * b.h; });
    return r;
}

Size grid_layout(int height, int width) {
    const float aspect = (float)width / height;
    const auto ratios = target_ratios();
    float best_diff = std::numeric_limits<float>::max();
    Size best{1, 1};
    const float area = (float)(width * height);
    for (const auto& r : ratios) {
        const float target = (float)r.w / r.h;
        const float diff = std::abs(aspect - target);
        if (diff < best_diff) { best_diff = diff; best = r; }
        else if (diff == best_diff) {
            const float target_area = (float)(TILE * TILE * r.w * r.h);
            if (area > 0.5f * target_area) best = r;
        }
    }
    return best;
}

VisionInput to_input(const std::vector<uint8_t>& rgb, int nx, int ny, const VisionPreParams& p) {
    VisionInput v;
    v.nx = nx; v.ny = ny;
    const size_t n = (size_t)nx * ny;
    v.chw.resize(n * 3);
    for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) {
            float x = (float)rgb[i * 3 + c] / 255.0f;              // clip_image_f32::from_u8
            v.chw[c * n + i] = (x - p.mean[c]) / p.stdv[c];       // normalize
        }
    return v;
}

std::vector<uint8_t> resize(const uint8_t* rgb, int nx, int ny, int w, int h) {
    std::vector<uint8_t> out((size_t)w * h * 3);
    resize_pillow_bilinear(rgb, nx, ny, out.data(), w, h);
    return out;
}
}  // namespace

std::vector<VisionInput> d1omni_preprocess_image(const uint8_t* rgb, int nx, int ny, const VisionPreParams& p) {
    if (nx <= 0 || ny <= 0) throw std::invalid_argument("empty image");
    const int align = p.patch_size * p.n_merge;
    const Size orig{nx, ny};
    const Size ov = calc_size_preserved_ratio(orig, align, p.min_pixels, p.max_pixels);
    std::vector<VisionInput> out;
    if (!should_tile(p, orig)) {
        out.push_back(to_input(resize(rgb, nx, ny, ov.w, ov.h), ov.w, ov.h, p));
        return out;
    }
    const Size grid = grid_layout(ny, nx);
    const int rw = TILE * grid.w, rh = TILE * grid.h;
    std::vector<uint8_t> refined = resize(rgb, nx, ny, rw, rh);   // d1omni_v: stretched, no padding
    std::vector<uint8_t> tile((size_t)TILE * TILE * 3);
    for (int row = 0; row < grid.h; row++)
        for (int col = 0; col < grid.w; col++) {
            for (int y = 0; y < TILE; ++y) {
                const uint8_t* s = refined.data() + ((size_t)(row * TILE + y) * rw + (size_t)col * TILE) * 3;
                std::copy(s, s + TILE * 3, tile.data() + (size_t)y * TILE * 3);
            }
            out.push_back(to_input(tile, TILE, TILE, p));
        }
    out.push_back(to_input(resize(rgb, nx, ny, ov.w, ov.h), ov.w, ov.h, p));   // thumbnail last (ov_img_first = false)
    return out;
}

int vision_tokens(const VisionInput& v, const VisionPreParams& p) {
    const int op = p.patch_size * p.n_merge;
    return ((v.nx + op - 1) / op) * ((v.ny + op - 1) / op);
}

// ============================================================ audio
namespace {
struct MelCache {
    std::vector<float> sin_vals, cos_vals, hann, filters;   // filters [n_mel][n_fft/2+1]
    int n_mel = 0;
};

void fill_sin_cos(MelCache& c, int n) {
    c.sin_vals.resize(n); c.cos_vals.resize(n);
    for (int i = 0; i < n; i++) {
        double theta = (2 * M_PI * i) / n;
        c.sin_vals[i] = sinf((float)theta);
        c.cos_vals[i] = cosf((float)theta);
    }
}
void fill_hann(MelCache& c, int length, bool periodic) {
    c.hann.resize(length);
    int offset = periodic ? 0 : -1;
    for (int i = 0; i < length; i++) c.hann[i] = (float)(0.5 * (1.0 - cosf((float)((2.0 * M_PI * i) / (length + offset)))));
}
void fill_mel_filterbank(MelCache& c, int n_mel, int n_fft, int sample_rate) {
    const double fmin = 0.0, fmax = 0.5 * sample_rate;
    const double min_log_hz = 1000.0, lin_slope = 3 / 200., min_log_mel = min_log_hz * lin_slope, log_step = log(6.4) / 27.0;
    auto hz_to_mel = [&](double f) { return f < min_log_hz ? f * lin_slope : min_log_mel + log(f / min_log_hz) / log_step; };
    auto mel_to_hz = [&](double m) { return m < min_log_mel ? m / lin_slope : min_log_hz * exp((m - min_log_mel) * log_step); };
    const double bin_hz = (double)sample_rate / n_fft;
    const double m_lo = hz_to_mel(fmin), m_hi = hz_to_mel(fmax);
    std::vector<double> hz(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) hz[i] = mel_to_hz(m_lo + (m_hi - m_lo) * ((double)i / (n_mel + 1)));
    const int nb = n_fft / 2 + 1;
    c.filters.assign((size_t)n_mel * nb, 0.f);
    for (int m = 0; m < n_mel; ++m) {
        const double fl = hz[m], fc = hz[m + 1], fr = hz[m + 2];
        const double dl = std::max(1e-30, fc - fl), dr = std::max(1e-30, fr - fc);
        const double en = 2.0 / std::max(1e-30, fr - fl);   // slaney area norm
        for (int k = 0; k < nb; ++k) {
            const double f = k * bin_hz;
            double w = 0.0;
            if (f >= fl && f <= fc) w = (f - fl) / dl;
            else if (f > fc && f <= fr) w = (fr - f) / dr;
            c.filters[(size_t)m * nb + k] = (float)(w * en * 1.0f);
        }
    }
    c.n_mel = n_mel;
}

// llama.cpp fft_impl<false, true> (recursive radix-2, real input)
void fft_real(const MelCache& c, float* in, int N, float* out) {
    const int n_sc = (int)c.sin_vals.size();
    if (N == 1) { out[0] = in[0]; out[1] = 0.0f; return; }
    const int half = N / 2;
    if (N - half * 2 == 1) {   // odd N: DFT
        const int step = n_sc / N;
        for (int k = 0; k < N; k++) {
            float re = 0, im = 0;
            for (int n = 0; n < N; n++) {
                int idx = (k * n * step) % n_sc;
                re += in[n] * c.cos_vals[idx];
                im += -1.0f * in[n] * c.sin_vals[idx];
            }
            out[k * 2] = re; out[k * 2 + 1] = im;
        }
        return;
    }
    float* even = in + N;
    for (int i = 0; i < half; ++i) even[i] = in[2 * i];
    float* even_fft = out + 2 * N;
    fft_real(c, even, half, even_fft);
    float* odd = even;
    for (int i = 0; i < half; ++i) odd[i] = in[2 * i + 1];
    float* odd_fft = even_fft + N;
    fft_real(c, odd, half, odd_fft);
    const int step = n_sc / N;
    for (int k = 0; k < half; k++) {
        const int idx = k * step;
        const float re = c.cos_vals[idx], im = -1.0f * c.sin_vals[idx];
        const float ro = odd_fft[2 * k], io = odd_fft[2 * k + 1];
        out[2 * k + 0] = 1.0f * (even_fft[2 * k + 0] + re * ro - im * io);
        out[2 * k + 1] = 1.0f * (even_fft[2 * k + 1] + re * io + im * ro);
        out[2 * (k + half) + 0] = 1.0f * (even_fft[2 * k + 0] - re * ro + im * io);
        out[2 * (k + half) + 1] = 1.0f * (even_fft[2 * k + 1] - re * io - im * ro);
    }
}

const MelCache& mel_cache(int n_mel) {
    static MelCache c;
    if (c.n_mel != n_mel) {
        fill_sin_cos(c, 512);
        fill_hann(c, 400, false);   // NeMo: torch.hann_window(periodic=False)
        fill_mel_filterbank(c, n_mel, 512, 16000);
    }
    return c;
}
}  // namespace

bool d1omni_audio_mel(const float* samples_in, size_t n_in, int n_mel, Mel& out) {
    if (n_in == 0) return false;
    const int SR = 16000, N_FFT = 512, WIN = 400, HOP = 160;
    // d1omni: cut to 30 s, pad to 0.5 s
    const size_t n_max = 30 * SR, n_min = SR / 2;
    std::vector<float> buf(samples_in, samples_in + std::min(n_in, n_max));
    buf.resize(std::max(buf.size(), n_min), 0.0f);
    const MelCache& c = mel_cache(n_mel);

    // ---- log_mel_spectrogram (center padding, preemph, natural log, mel_floor_add, per-feature norm)
    const int n_samples_in = (int)buf.size();
    const int frame_size = N_FFT, frame_step = HOP;
    const int pad = frame_size / 2;
    std::vector<float> sp((size_t)n_samples_in + 2 * pad, 0.f);
    std::copy(buf.begin(), buf.end(), sp.begin() + pad);
    const int n_samples = (int)sp.size();
    {   // preemphasis (over the padded buffer, as in llama.cpp)
        const float pre = 0.97f;
        float prev = sp[pad];
        for (int i = pad + 1; i + pad < n_samples; ++i) { float cur = sp[i]; sp[i] = cur - pre * prev; prev = cur; }
    }
    std::vector<float> hann(frame_size, 0.f);
    std::copy(c.hann.begin(), c.hann.end(), hann.begin() + (frame_size - WIN) / 2);
    const int n_fft_bins = N_FFT / 2 + 1;
    int n_len = (n_samples - frame_size) / frame_step + 1;
    std::vector<float> data((size_t)n_mel * n_len);
    const float mel_floor = 5.960464477539063e-08f;
    std::vector<float> fft_in(frame_size * 2, 0.f), fft_out(frame_size * 2 * 2 * 2);
    int64_t i = 0;
    for (; i < std::min((int64_t)(n_samples / frame_step + 1), (int64_t)n_len); ++i) {
        const int64_t off = i * frame_step;
        const int valid = std::min(frame_size, std::max(0, n_samples - (int)off));
        for (int j = 0; j < valid; j++) fft_in[j] = hann[j] * sp[off + j];
        if (valid < frame_size) std::fill(fft_in.begin() + valid, fft_in.end(), 0.0f);
        fft_real(c, fft_in.data(), frame_size, fft_out.data());
        for (int j = 0; j < n_fft_bins; j++) fft_out[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        for (int j = 0; j < n_mel; j++) {
            double sum = 0.0;
            int k = 0;
            const float* f = c.filters.data() + (size_t)j * n_fft_bins;
            for (k = 0; k < n_fft_bins - 3; k += 4)
                sum += fft_out[k] * f[k] + fft_out[k + 1] * f[k + 1] + fft_out[k + 2] * f[k + 2] + fft_out[k + 3] * f[k + 3];
            for (; k < n_fft_bins; k++) sum += fft_out[k] * f[k];
            sum = sum + (double)mel_floor;
            data[(size_t)j * n_len + i] = (float)log(sum);
        }
    }
    for (; i < n_len; ++i) for (int j = 0; j < n_mel; j++) data[(size_t)j * n_len + i] = (float)log(1e-10);
    const int64_t eff = n_samples_in / frame_step;
    if (eff <= 1) return false;
    for (int m = 0; m < n_mel; m++) {
        float* row = data.data() + (size_t)m * n_len;
        double mean = 0;
        for (int64_t j = 0; j < eff; ++j) mean += row[j];
        mean /= eff;
        double var = 0;
        for (int64_t j = 0; j < eff; ++j) { double v = row[j] - mean; var += v * v; }
        var /= eff - 1;
        const double mstd = std::sqrt(var) + 1e-5;   // std_eps_after_sqrt
        for (int64_t j = 0; j < eff; ++j) row[j] = (float)((row[j] - mean) / mstd);
        for (int64_t j = eff; j < n_len; ++j) row[j] = 0.0f;
    }
    // d1omni: the encoder reads one frame per hop (drop the extra centre-padding frame)
    const int n_frames = (int)(buf.size() / HOP);
    if (n_len > n_frames) {
        std::vector<float> d2((size_t)n_mel * n_frames);
        for (int m = 0; m < n_mel; ++m)
            std::copy_n(data.begin() + (size_t)m * n_len, n_frames, d2.begin() + (size_t)m * n_frames);
        data.swap(d2);
        n_len = n_frames;
    }
    out.n_mel = n_mel; out.n_len = n_len; out.data = std::move(data);
    return true;
}

// ---------------------------------------------------------------- Gemma 4 audio frontend
namespace {
struct G4MelCache {
    MelCache c;
    int n_mel = 0;
};
void fill_mel_filterbank_htk(MelCache& c, int n_mel, int n_fft, int sample_rate) {
    const double fmin = 0.0, fmax = 0.5 * sample_rate;
    auto hz_to_mel = [](double f) { return 2595.0 * log10(1.0 + f / 700.0); };
    auto mel_to_hz = [](double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); };
    const double bin_hz = (double)sample_rate / n_fft;
    const double m_lo = hz_to_mel(fmin), m_hi = hz_to_mel(fmax);
    std::vector<double> hz(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) hz[i] = mel_to_hz(m_lo + (m_hi - m_lo) * ((double)i / (n_mel + 1)));
    const int nb = n_fft / 2 + 1;
    c.filters.assign((size_t)n_mel * nb, 0.f);
    for (int m = 0; m < n_mel; ++m) {
        const double fl = hz[m], fc = hz[m + 1], fr = hz[m + 2];
        const double dl = std::max(1e-30, fc - fl), dr = std::max(1e-30, fr - fc);
        for (int k = 0; k < nb; ++k) {
            const double f = k * bin_hz;
            double w = 0.0;
            if (f >= fl && f <= fc) w = (f - fl) / dl;
            else if (f > fc && f <= fr) w = (fr - f) / dr;
            c.filters[(size_t)m * nb + k] = (float)w;                      // no Slaney area normalisation
        }
    }
    c.n_mel = n_mel;
}
const MelCache& g4_mel_cache(int n_mel) {
    static MelCache c;
    if (c.n_mel != n_mel) {
        fill_sin_cos(c, 512);
        c.hann.assign(512, 0.f);                                          // periodic Hann over the 20 ms window, zero-padded to the FFT size
        for (int i = 0; i < 320; ++i) c.hann[i] = 0.5f - 0.5f * cosf((2.0f * (float)M_PI * i) / 320.0f);
        fill_mel_filterbank_htk(c, n_mel, 512, 16000);
    }
    return c;
}
}  // namespace

bool gemma4a_audio_mel(const float* samples_in, size_t n_in, int n_mel, std::vector<Mel>& out) {
    if (n_in == 0) return false;
    const int SR = 16000, N_FFT = 512, WIN = 320, HOP = 160;
    const MelCache& c = g4_mel_cache(n_mel);
    const size_t chunk_samples = (size_t)30 * SR;
    for (size_t off = 0; off < n_in; off += chunk_samples) {
        const float* chunk = samples_in + off;
        const size_t chunk_len = std::min(chunk_samples, n_in - off);
        // semicausal left padding + right padding to match the PyTorch frame count
        const int pad_left = WIN / 2;
        const int n_with_left = (int)chunk_len + pad_left;
        const int64_t pt_frames = (n_with_left - (WIN + 1)) / HOP + 1;
        const int64_t n_padded_needed = (pt_frames - 1) * HOP + N_FFT;
        const int total_pad = std::max((int)(n_padded_needed - (int)chunk_len), pad_left);
        std::vector<float> sp(total_pad + chunk_len, 0.0f);
        std::copy(chunk, chunk + chunk_len, sp.begin() + pad_left);
        const int n_samples = (int)sp.size();
        const int n_fft_bins = N_FFT / 2 + 1;
        int n_len = (n_samples - N_FFT) / HOP + 1;
        if (n_len <= 0 || n_samples < N_FFT) return false;
        std::vector<float> data((size_t)n_mel * n_len);
        std::vector<float> fft_in(N_FFT * 2, 0.f), fft_out(N_FFT * 2 * 2 * 2);
        int64_t i = 0;
        for (; i < std::min((int64_t)(n_samples / HOP + 1), (int64_t)n_len); ++i) {
            const int64_t o = i * HOP;
            const int valid = std::min(N_FFT, std::max(0, n_samples - (int)o));
            for (int j = 0; j < valid; j++) fft_in[j] = c.hann[j] * sp[o + j];
            if (valid < N_FFT) std::fill(fft_in.begin() + valid, fft_in.end(), 0.0f);
            fft_real(c, fft_in.data(), N_FFT, fft_out.data());
            for (int j = 0; j < n_fft_bins; j++) {
                const float power = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
                fft_out[j] = sqrtf(power);                                // magnitude
            }
            for (int j = 0; j < n_mel; j++) {
                double sum = 0.0;
                int k = 0;
                const float* f = c.filters.data() + (size_t)j * n_fft_bins;
                for (k = 0; k < n_fft_bins - 3; k += 4)
                    sum += fft_out[k] * f[k] + fft_out[k + 1] * f[k + 1] + fft_out[k + 2] * f[k + 2] + fft_out[k + 3] * f[k + 3];
                for (; k < n_fft_bins; k++) sum += fft_out[k] * f[k];
                sum = std::max(sum, 0.001);                               // mel floor
                data[(size_t)j * n_len + i] = (float)log(sum);
            }
        }
        for (; i < n_len; ++i) for (int j = 0; j < n_mel; j++) data[(size_t)j * n_len + i] = (float)log(1e-10);
        Mel m; m.n_mel = n_mel; m.n_len = (int)std::min<int64_t>(n_len, pt_frames); m.data.resize((size_t)n_mel * m.n_len);
        for (int j = 0; j < n_mel; ++j)                                   // trim to the PyTorch frame count, keep [mel][frame]
            std::copy_n(data.begin() + (size_t)j * n_len, m.n_len, m.data.begin() + (size_t)j * m.n_len);
        out.push_back(std::move(m));
    }
    return true;
}

int audio_tokens(const Mel& m) { return ((((m.n_len + 1) / 2) + 1) / 2 + 1) / 2; }

// ---- Gemma 4 / EmbeddingGemma 2 vision: mtmd_image_preprocessor_dyn_size (smart_resize, align patch*merge, bicubic)
static void gemma4_target_size(int width, int height, int align, long min_pixels, long max_pixels, int& w_bar, int& h_bar) {
    auto round_by = [align](float x) { return (int)std::round(x / (float)align) * align; };
    auto ceil_by = [align](float x) { return (int)std::ceil(x / (float)align) * align; };
    auto floor_by = [align](float x) { return (int)std::floor(x / (float)align) * align; };
    w_bar = std::max(align, round_by((float)width));
    h_bar = std::max(align, round_by((float)height));
    if (max_pixels > 0 && (long)h_bar * w_bar > max_pixels) {
        const float beta = std::sqrt((float)height * width / (float)max_pixels);
        h_bar = std::max(align, floor_by(height / beta));
        w_bar = std::max(align, floor_by(width / beta));
    } else if (min_pixels > 0 && (long)h_bar * w_bar < min_pixels) {
        const float beta = std::sqrt((float)min_pixels / ((float)height * width));
        h_bar = ceil_by(height * beta);
        w_bar = ceil_by(width * beta);
    }
}

VisionInput gemma4_preprocess_image(const uint8_t* rgb, int nx, int ny, const Gemma4VisionPre& p) {
    const int patch_area = p.patch_size * p.patch_size * p.n_merge * p.n_merge;
    int w, h;
    gemma4_target_size(nx, ny, p.patch_size * p.n_merge, (long)p.min_tokens * patch_area, (long)p.max_tokens * patch_area, w, h);
    std::vector<uint8_t> rs((size_t)w * h * 3);
    if (w == nx && h == ny) {
        std::copy(rgb, rgb + (size_t)nx * ny * 3, rs.begin());
    } else {
        // mtmd img_tool::resize with PAD_CEIL: fit inside the target keeping aspect ratio (ceil), black letterbox, centered
        const float scale = std::min((float)w / nx, (float)h / ny);
        const int nw = std::min((int)std::ceil(nx * scale), w), nh = std::min((int)std::ceil(ny * scale), h);
        std::vector<uint8_t> fit((size_t)nw * nh * 3);
        resize_pillow_bicubic(rgb, nx, ny, fit.data(), nw, nh);
        std::fill(rs.begin(), rs.end(), 0);
        const int ox = (w - nw) / 2, oy = (h - nh) / 2;
        for (int y = 0; y < nh; ++y) memcpy(rs.data() + ((size_t)(y + oy) * w + ox) * 3, fit.data() + (size_t)y * nw * 3, (size_t)nw * 3);
    }
    VisionInput out;
    out.nx = w; out.ny = h; out.chw.resize((size_t)3 * w * h);
    for (int c = 0; c < 3; ++c)                          // mean 0 / std 1: x / 255 (the encoder applies 2x - 1)
        for (size_t i = 0; i < (size_t)w * h; ++i) out.chw[(size_t)c * w * h + i] = rs[i * 3 + c] / 255.0f;
    return out;
}

}  // namespace ferro
