/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * signal_processing.h — MORIE Signal Processing Kernels
 *
 * Pure C99 implementations of DSP primitives for biosignal analysis.
 * All functions operate on float* arrays with explicit length parameters.
 *
 * References:
 *   - Cooley & Tukey (1965). An Algorithm for the Machine Calculation of
 *     Complex Fourier Series. Math. Comp. 19(90):297-301.
 *   - Welch (1967). The Use of FFT for Estimation of Power Spectra. IEEE TAU.
 *   - Oppenheim & Schafer (2010). Discrete-Time Signal Processing, 3rd ed.
 *   - Rangayyan (2015). Biomedical Signal Analysis, 3rd ed.
 *
 * Compile:
 *   macOS:  cc -O2 -march=native -shared -o signal_processing.dylib signal_processing.c -lm
 *   Linux:  cc -O2 -march=native -shared -fPIC -o signal_processing.so signal_processing.c -lm
 */

#ifndef MORIE_SIGNAL_PROCESSING_H
#define MORIE_SIGNAL_PROCESSING_H

#ifdef __cplusplus
extern "C" {
#endif

#define DSP_OK          0
#define DSP_ERR_NULL   -1
#define DSP_ERR_SIZE   -2
#define DSP_ERR_PARAM  -3

int morie_fft(float *re, float *im, int n);

int morie_ifft(float *re, float *im, int n);

int morie_psd_periodogram(
    const float *x,
    int          n,
    float       *psd,
    int          n_psd,
    float        fs
);

int morie_psd_welch(
    const float *x,
    int          n,
    float       *psd,
    int          n_psd,
    int          seg_len,
    int          overlap,
    float        fs
);

int morie_autocorrelation(
    const float *x,
    int          n,
    float       *r,
    int          max_lag
);

int morie_crosscorrelation(
    const float *x,
    const float *y,
    int          n,
    float       *r,
    int          max_lag
);

int morie_convolve(
    const float *x,
    int          nx,
    const float *h,
    int          nh,
    float       *out,
    int          n_out
);

int morie_convolve_fft(
    const float *x,
    int          nx,
    const float *h,
    int          nh,
    float       *out,
    int          n_out
);

int morie_fir_filter(
    const float *x,
    int          n,
    const float *b,
    int          nb,
    float       *out
);

int morie_iir_filter(
    const float *x,
    int          n,
    const float *b,
    int          nb,
    const float *a,
    int          na,
    float       *out
);

int morie_window_hamming(float *w, int n);
int morie_window_hanning(float *w, int n);
int morie_window_blackman(float *w, int n);

float morie_zero_crossing_rate(const float *x, int n);

float morie_rms_energy(const float *x, int n);

#ifdef __cplusplus
}
#endif

#endif /* MORIE_SIGNAL_PROCESSING_H */
