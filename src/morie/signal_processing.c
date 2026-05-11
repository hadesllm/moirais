/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * signal_processing.c — MORIE Signal Processing Kernels
 *
 * References:
 *   - Cooley & Tukey (1965). An Algorithm for the Machine Calculation of
 *     Complex Fourier Series. Math. Comp. 19(90):297-301.
 *   - Welch (1967). The Use of FFT for Estimation of Power Spectra. IEEE TAU.
 *   - Oppenheim & Schafer (2010). Discrete-Time Signal Processing, 3rd ed.
 *   - Rangayyan (2015). Biomedical Signal Analysis, 3rd ed.
 */

#include "signal_processing.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define MAX_SIG (1 << 24)

#define CHECK_NULL(ptr)  do { if (!(ptr)) return DSP_ERR_NULL; } while (0)
#define CHECK_SIZE(n)    do { if ((n) <= 0 || (n) > MAX_SIG) return DSP_ERR_SIZE; } while (0)

static int is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static int next_pow2(int n) {
    int v = 1;
    while (v < n) v <<= 1;
    return v;
}

/* ---- Bit-reversal permutation ------------------------------------------- */

static void bit_reverse(float *re, float *im, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
}

/* ---- Cooley-Tukey Radix-2 FFT (in-place) -------------------------------- */

int morie_fft(float *re, float *im, int n) {
    CHECK_NULL(re);
    CHECK_NULL(im);
    CHECK_SIZE(n);
    if (!is_power_of_2(n)) return DSP_ERR_SIZE;

    bit_reverse(re, im, n);

    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(angle);
        float wim = sinf(angle);

        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f;
            float cur_im = 0.0f;

            int half = len >> 1;
            for (int j = 0; j < half; j++) {
                int u = i + j;
                int v = u + half;

                float tre = cur_re * re[v] - cur_im * im[v];
                float tim = cur_re * im[v] + cur_im * re[v];

                re[v] = re[u] - tre;
                im[v] = im[u] - tim;
                re[u] += tre;
                im[u] += tim;

                float next_re = cur_re * wre - cur_im * wim;
                float next_im = cur_re * wim + cur_im * wre;
                cur_re = next_re;
                cur_im = next_im;
            }
        }
    }

    return DSP_OK;
}

/* ---- Inverse FFT -------------------------------------------------------- */

int morie_ifft(float *re, float *im, int n) {
    CHECK_NULL(re);
    CHECK_NULL(im);
    CHECK_SIZE(n);
    if (!is_power_of_2(n)) return DSP_ERR_SIZE;

    for (int i = 0; i < n; i++) im[i] = -im[i];

    int rc = morie_fft(re, im, n);
    if (rc != DSP_OK) return rc;

    float inv_n = 1.0f / (float)n;
    for (int i = 0; i < n; i++) {
        re[i] *= inv_n;
        im[i] = -im[i] * inv_n;
    }

    return DSP_OK;
}

/* ---- Window Functions --------------------------------------------------- */

int morie_window_hamming(float *w, int n) {
    CHECK_NULL(w);
    CHECK_SIZE(n);
    for (int i = 0; i < n; i++) {
        w[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1));
    }
    return DSP_OK;
}

int morie_window_hanning(float *w, int n) {
    CHECK_NULL(w);
    CHECK_SIZE(n);
    for (int i = 0; i < n; i++) {
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1)));
    }
    return DSP_OK;
}

int morie_window_blackman(float *w, int n) {
    CHECK_NULL(w);
    CHECK_SIZE(n);
    for (int i = 0; i < n; i++) {
        float t = 2.0f * (float)M_PI * (float)i / (float)(n - 1);
        w[i] = 0.42f - 0.5f * cosf(t) + 0.08f * cosf(2.0f * t);
    }
    return DSP_OK;
}

/* ---- Periodogram PSD ---------------------------------------------------- */

int morie_psd_periodogram(
    const float *x,
    int          n,
    float       *psd,
    int          n_psd,
    float        fs
) {
    CHECK_NULL(x);
    CHECK_NULL(psd);
    CHECK_SIZE(n);
    if (fs <= 0.0f) return DSP_ERR_PARAM;

    int nfft = next_pow2(n);
    int expected = nfft / 2 + 1;
    if (n_psd < expected) return DSP_ERR_SIZE;

    float *re = (float *)calloc(nfft, sizeof(float));
    float *im = (float *)calloc(nfft, sizeof(float));
    if (!re || !im) {
        free(re);
        free(im);
        return DSP_ERR_NULL;
    }

    memcpy(re, x, n * sizeof(float));

    morie_fft(re, im, nfft);

    float scale = 1.0f / (fs * (float)nfft);
    for (int i = 0; i < expected; i++) {
        psd[i] = (re[i] * re[i] + im[i] * im[i]) * scale;
        if (i > 0 && i < expected - 1) psd[i] *= 2.0f;
    }

    free(re);
    free(im);
    return DSP_OK;
}

/* ---- Welch PSD ---------------------------------------------------------- */

int morie_psd_welch(
    const float *x,
    int          n,
    float       *psd,
    int          n_psd,
    int          seg_len,
    int          overlap,
    float        fs
) {
    CHECK_NULL(x);
    CHECK_NULL(psd);
    CHECK_SIZE(n);
    if (fs <= 0.0f || seg_len <= 0 || overlap < 0 || overlap >= seg_len)
        return DSP_ERR_PARAM;
    if (!is_power_of_2(seg_len)) return DSP_ERR_SIZE;

    int n_bins = seg_len / 2 + 1;
    if (n_psd < n_bins) return DSP_ERR_SIZE;

    int step = seg_len - overlap;
    int n_seg = 0;

    float *win = (float *)malloc(seg_len * sizeof(float));
    float *re  = (float *)malloc(seg_len * sizeof(float));
    float *im  = (float *)calloc(seg_len, sizeof(float));
    if (!win || !re || !im) {
        free(win); free(re); free(im);
        return DSP_ERR_NULL;
    }

    morie_window_hanning(win, seg_len);

    float win_power = 0.0f;
    for (int i = 0; i < seg_len; i++) win_power += win[i] * win[i];
    win_power /= (float)seg_len;

    for (int i = 0; i < n_bins; i++) psd[i] = 0.0f;

    for (int start = 0; start + seg_len <= n; start += step) {
        for (int i = 0; i < seg_len; i++) {
            re[i] = x[start + i] * win[i];
        }
        memset(im, 0, seg_len * sizeof(float));

        morie_fft(re, im, seg_len);

        float scale = 1.0f / (fs * win_power * (float)seg_len);
        for (int i = 0; i < n_bins; i++) {
            float p = (re[i] * re[i] + im[i] * im[i]) * scale;
            if (i > 0 && i < n_bins - 1) p *= 2.0f;
            psd[i] += p;
        }
        n_seg++;
    }

    if (n_seg > 0) {
        float inv = 1.0f / (float)n_seg;
        for (int i = 0; i < n_bins; i++) psd[i] *= inv;
    }

    free(win);
    free(re);
    free(im);
    return DSP_OK;
}

/* ---- Autocorrelation ---------------------------------------------------- */

int morie_autocorrelation(
    const float *x,
    int          n,
    float       *r,
    int          max_lag
) {
    CHECK_NULL(x);
    CHECK_NULL(r);
    CHECK_SIZE(n);
    if (max_lag <= 0 || max_lag > n) return DSP_ERR_PARAM;

    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }

    if (var < 1e-30f) {
        for (int k = 0; k < max_lag; k++) r[k] = 0.0f;
        return DSP_OK;
    }

    for (int k = 0; k < max_lag; k++) {
        float sum = 0.0f;
        for (int i = 0; i < n - k; i++) {
            sum += (x[i] - mean) * (x[i + k] - mean);
        }
        r[k] = sum / var;
    }

    return DSP_OK;
}

/* ---- Cross-correlation -------------------------------------------------- */

int morie_crosscorrelation(
    const float *x,
    const float *y,
    int          n,
    float       *r,
    int          max_lag
) {
    CHECK_NULL(x);
    CHECK_NULL(y);
    CHECK_NULL(r);
    CHECK_SIZE(n);
    if (max_lag <= 0 || max_lag > n) return DSP_ERR_PARAM;

    float mx = 0.0f, my = 0.0f;
    for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= (float)n;
    my /= (float)n;

    float sx = 0.0f, sy = 0.0f;
    for (int i = 0; i < n; i++) {
        sx += (x[i] - mx) * (x[i] - mx);
        sy += (y[i] - my) * (y[i] - my);
    }
    float denom = sqrtf(sx * sy);
    if (denom < 1e-30f) {
        for (int k = 0; k < max_lag; k++) r[k] = 0.0f;
        return DSP_OK;
    }

    for (int k = 0; k < max_lag; k++) {
        float sum = 0.0f;
        for (int i = 0; i < n - k; i++) {
            sum += (x[i] - mx) * (y[i + k] - my);
        }
        r[k] = sum / denom;
    }

    return DSP_OK;
}

/* ---- Convolution (time-domain) ------------------------------------------ */

int morie_convolve(
    const float *x,
    int          nx,
    const float *h,
    int          nh,
    float       *out,
    int          n_out
) {
    CHECK_NULL(x);
    CHECK_NULL(h);
    CHECK_NULL(out);
    CHECK_SIZE(nx);
    CHECK_SIZE(nh);

    int full_len = nx + nh - 1;
    if (n_out < full_len) return DSP_ERR_SIZE;

    for (int i = 0; i < full_len; i++) {
        float sum = 0.0f;
        int j_min = (i >= nh - 1) ? i - nh + 1 : 0;
        int j_max = (i < nx - 1) ? i : nx - 1;
        for (int j = j_min; j <= j_max; j++) {
            sum += x[j] * h[i - j];
        }
        out[i] = sum;
    }

    return DSP_OK;
}

/* ---- Convolution (FFT-based) -------------------------------------------- */

int morie_convolve_fft(
    const float *x,
    int          nx,
    const float *h,
    int          nh,
    float       *out,
    int          n_out
) {
    CHECK_NULL(x);
    CHECK_NULL(h);
    CHECK_NULL(out);
    CHECK_SIZE(nx);
    CHECK_SIZE(nh);

    int full_len = nx + nh - 1;
    if (n_out < full_len) return DSP_ERR_SIZE;

    int nfft = next_pow2(full_len);

    float *xr = (float *)calloc(nfft, sizeof(float));
    float *xi = (float *)calloc(nfft, sizeof(float));
    float *hr = (float *)calloc(nfft, sizeof(float));
    float *hi = (float *)calloc(nfft, sizeof(float));
    if (!xr || !xi || !hr || !hi) {
        free(xr); free(xi); free(hr); free(hi);
        return DSP_ERR_NULL;
    }

    memcpy(xr, x, nx * sizeof(float));
    memcpy(hr, h, nh * sizeof(float));

    morie_fft(xr, xi, nfft);
    morie_fft(hr, hi, nfft);

    for (int i = 0; i < nfft; i++) {
        float a = xr[i] * hr[i] - xi[i] * hi[i];
        float b = xr[i] * hi[i] + xi[i] * hr[i];
        xr[i] = a;
        xi[i] = b;
    }

    morie_ifft(xr, xi, nfft);

    for (int i = 0; i < full_len; i++) out[i] = xr[i];

    free(xr); free(xi); free(hr); free(hi);
    return DSP_OK;
}

/* ---- FIR Filter --------------------------------------------------------- */

int morie_fir_filter(
    const float *x,
    int          n,
    const float *b,
    int          nb,
    float       *out
) {
    CHECK_NULL(x);
    CHECK_NULL(b);
    CHECK_NULL(out);
    CHECK_SIZE(n);
    CHECK_SIZE(nb);

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < nb; j++) {
            int idx = i - j;
            if (idx >= 0) sum += b[j] * x[idx];
        }
        out[i] = sum;
    }

    return DSP_OK;
}

/* ---- IIR Filter (Direct Form II Transposed) ----------------------------- */

int morie_iir_filter(
    const float *x,
    int          n,
    const float *b,
    int          nb,
    const float *a,
    int          na,
    float       *out
) {
    CHECK_NULL(x);
    CHECK_NULL(b);
    CHECK_NULL(a);
    CHECK_NULL(out);
    CHECK_SIZE(n);
    CHECK_SIZE(nb);
    CHECK_SIZE(na);
    if (a[0] == 0.0f) return DSP_ERR_PARAM;

    int max_order = (nb > na) ? nb : na;
    float *state = (float *)calloc(max_order, sizeof(float));
    if (!state) return DSP_ERR_NULL;

    float inv_a0 = 1.0f / a[0];

    for (int i = 0; i < n; i++) {
        float y = state[0] + b[0] * inv_a0 * x[i];

        for (int j = 0; j < max_order - 1; j++) {
            state[j] = state[j + 1];
            if (j + 1 < nb) state[j] += b[j + 1] * inv_a0 * x[i];
            if (j + 1 < na) state[j] -= a[j + 1] * inv_a0 * y;
        }

        out[i] = y;
    }

    free(state);
    return DSP_OK;
}

/* ---- Zero-Crossing Rate ------------------------------------------------- */

float morie_zero_crossing_rate(const float *x, int n) {
    if (!x || n < 2) return 0.0f;

    int count = 0;
    for (int i = 1; i < n; i++) {
        if ((x[i] >= 0.0f && x[i - 1] < 0.0f) ||
            (x[i] < 0.0f && x[i - 1] >= 0.0f)) {
            count++;
        }
    }

    return (float)count / (float)(n - 1);
}

/* ---- RMS Energy --------------------------------------------------------- */

float morie_rms_energy(const float *x, int n) {
    if (!x || n <= 0) return 0.0f;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }

    return sqrtf(sum / (float)n);
}
