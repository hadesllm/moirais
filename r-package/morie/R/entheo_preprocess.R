# SPDX-License-Identifier: MIT OR Apache-2.0
#' Preprocess EEG (Butterworth bandpass + notch + ASR-style trimming)
#'
#' R parity of \code{morie.entheo.preprocess_eeg}.  Uses
#' \pkg{signal} when available for Butterworth filtering; falls back
#' to an FFT-mask when not.
#'
#' @param record list.  Subject record from \code{load_dmt_imaging()}.
#' @param bandpass numeric vector length 2.  Passband (Hz).  Default
#'   \code{c(1, 40)}.
#' @param notch numeric.  Line-noise frequency (Hz).  Default 60.
#' @param asr_threshold numeric.  Z-score threshold for ASR-style
#'   sample reconstruction.  Default 20.
#' @return Named list with cleaned \code{record}, \code{n_bad},
#'   \code{sfreq}, and parameters.
#' @keywords internal
preprocess_eeg <- function(record,
                           bandpass = c(1, 40),
                           notch = 60,
                           asr_threshold = 20) {
  eeg <- record$eeg
  sfreq <- if (is.null(eeg$sfreq) || is.na(eeg$sfreq)) 250 else eeg$sfreq
  cleaned <- record
  warnings_vec <- character(0)
  n_bad_total <- 0L
  n_chan <- 0L

  for (key in c("data_dmt", "data_pcb")) {
    arr <- eeg[[key]]
    if (is.null(arr) || !is.matrix(arr)) {
      warnings_vec <- c(warnings_vec, sprintf("eeg.%s absent -- skipping", key))
      next
    }
    n_chan <- max(n_chan, nrow(arr))
    arr <- .entheo_bandpass(arr, sfreq, bandpass[1], bandpass[2])
    arr <- .entheo_notch(arr, sfreq, notch)
    tr <- .entheo_asr_trim(arr, asr_threshold)
    arr <- tr$arr
    n_bad_total <- n_bad_total + tr$n_bad
    cleaned$eeg[[key]] <- arr
  }

  list(record = cleaned,
       n_bad  = n_bad_total,
       sfreq  = sfreq,
       bandpass = bandpass,
       notch    = notch,
       asr_threshold = asr_threshold,
       n_channels = n_chan,
       warnings = warnings_vec,
       interpretation = sprintf(
         "EEG bandpass-filtered (%g-%g Hz) and notch-filtered at %g Hz; %d sample(s) reconstructed by toy ASR.",
         bandpass[1], bandpass[2], notch, n_bad_total))
}


#' Preprocess fMRI (motion scrubbing + ICA-AROMA-style noise removal)
#'
#' R parity of \code{morie.entheo.preprocess_fmri}.
#'
#' @param record list.  Subject record.
#' @param motion_threshold_mm numeric.  FD threshold (mm).  Default 0.5.
#' @param n_noise_components integer.  Top-k SVD components to project
#'   out as a toy AROMA stand-in.  Default 5.
#' @return Named list with cleaned \code{record}, \code{n_scrubbed}.
#' @keywords internal
preprocess_fmri <- function(record,
                            motion_threshold_mm = 0.5,
                            n_noise_components = 5L) {
  fmri <- record$fmri
  cleaned <- record
  warnings_vec <- character(0)
  n_scrubbed <- 0L
  n_parcels <- 0L

  for (key in c("data_dmt", "data_pcb")) {
    arr <- fmri[[key]]
    if (is.null(arr) || !is.matrix(arr)) {
      warnings_vec <- c(warnings_vec, sprintf("fmri.%s absent -- skipping", key))
      next
    }
    n_parcels <- max(n_parcels, nrow(arr))
    fd <- fmri$motion_fd_mm
    if (!is.null(fd)) {
      t <- ncol(arr)
      fd_t <- if (length(fd) >= t) fd[seq_len(t)] else c(fd, rep(0, t - length(fd)))
      bad <- fd_t > motion_threshold_mm
      n_scrubbed <- n_scrubbed + sum(bad)
      if (any(bad)) arr[, bad] <- 0
    } else {
      warnings_vec <- c(warnings_vec,
                        sprintf("fmri.motion_fd_mm absent -- skipping scrubbing on %s", key))
    }
    sv <- tryCatch(svd(arr), error = function(e) NULL)
    if (!is.null(sv)) {
      k <- min(n_noise_components, length(sv$d))
      d <- sv$d
      d[seq_len(k)] <- 0
      arr <- sv$u %*% diag(d, nrow = length(d)) %*% t(sv$v)
    } else {
      warnings_vec <- c(warnings_vec, sprintf("SVD failed on fmri.%s; skipping AROMA", key))
    }
    cleaned$fmri[[key]] <- arr
  }

  list(record = cleaned,
       n_scrubbed = n_scrubbed,
       motion_threshold_mm = motion_threshold_mm,
       n_noise_components = n_noise_components,
       n_parcels = n_parcels,
       warnings = warnings_vec,
       interpretation = sprintf(
         "Motion-scrubbed %d volume(s) above %g mm FD; top-%d singular components projected out as toy ICA-AROMA stand-in.",
         n_scrubbed, motion_threshold_mm, n_noise_components))
}


# ---------------------------------------------------------------------------
# Internal filter helpers
# ---------------------------------------------------------------------------

.entheo_bandpass <- function(x, sfreq, low, high, order = 4L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    ny <- sfreq / 2
    bf <- signal::butter(order, c(low / ny, high / ny), type = "pass")
    out <- t(apply(x, 1, function(row) signal::filtfilt(bf, row)))
    return(out)
  }
  # FFT-mask fallback.
  n <- ncol(x)
  freqs <- seq(0, sfreq / 2, length.out = n %/% 2 + 1)
  mask <- (freqs >= low) & (freqs <= high)
  out <- x
  for (i in seq_len(nrow(x))) {
    spec <- stats::fft(x[i, ])
    spec[!c(mask, rev(mask[-c(1, length(mask))]))] <- 0
    out[i, ] <- Re(stats::fft(spec, inverse = TRUE) / n)
  }
  out
}

.entheo_notch <- function(x, sfreq, freq, q = 30) {
  if (requireNamespace("signal", quietly = TRUE)) {
    bw <- freq / q
    bf <- signal::butter(2, c((freq - bw / 2) / (sfreq / 2),
                              (freq + bw / 2) / (sfreq / 2)),
                         type = "stop")
    out <- t(apply(x, 1, function(row) signal::filtfilt(bf, row)))
    return(out)
  }
  n <- ncol(x)
  freqs <- seq(0, sfreq / 2, length.out = n %/% 2 + 1)
  bw <- freq / q
  mask <- !((freqs >= freq - bw / 2) & (freqs <= freq + bw / 2))
  out <- x
  for (i in seq_len(nrow(x))) {
    spec <- stats::fft(x[i, ])
    spec[!c(mask, rev(mask[-c(1, length(mask))]))] <- 0
    out[i, ] <- Re(stats::fft(spec, inverse = TRUE) / n)
  }
  out
}

.entheo_asr_trim <- function(x, threshold) {
  mu <- rowMeans(x)
  sd <- apply(x, 1, stats::sd) + 1e-9
  z <- (x - mu) / sd
  bad <- abs(z) > threshold
  n_bad <- sum(bad)
  if (n_bad > 0) x[bad] <- mu[row(x)][bad]
  list(arr = x, n_bad = n_bad)
}
