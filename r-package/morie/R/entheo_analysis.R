# SPDX-License-Identifier: MIT OR Apache-2.0
#' Beautiful Loop phenomenal-binding metric
#'
#' R parity of \code{morie.entheo.beautiful_loop_metric}.  Scores
#' the cross-modal coupling between EEG-power envelope and fMRI
#' gradient dispersion (Bayne, Carter, Laukkonen, Slagter).
#'
#' @param eeg matrix or list.  Either an EEG channels x timepoints
#'   matrix, or a subject record from \code{load_dmt_imaging()}.
#' @param fmri matrix or NULL.  Required if \code{eeg} is a matrix
#'   (parcels x timepoints).
#' @return Named list with \code{score_dmt}, \code{score_pcb},
#'   \code{contrast}, \code{per_frame_dmt}, \code{per_frame_pcb}.
#' @references
#' Bayne, T. & Carter, O. (2018). Dimensions of consciousness and the
#'   psychedelic state. Neuroscience of Consciousness.
#' Laukkonen, R. E. & Slagter, H. A. (2021). From many to (n)one:
#'   meditation and the plasticity of the predictive mind.
#'   Neurosci Biobehav Rev.
#' @keywords internal
beautiful_loop_metric <- function(eeg, fmri = NULL) {
  pair <- .entheo_extract_pair(eeg, fmri)
  warnings_vec <- character(0)
  if (is.null(pair$e_dmt) || is.null(pair$f_dmt)) {
    return(list(score = NA_real_,
                score_dmt = NA_real_, score_pcb = NA_real_,
                contrast = NA_real_,
                per_frame_dmt = NULL, per_frame_pcb = NULL,
                warnings = "EEG or fMRI missing for primary condition"))
  }
  pf_dmt <- .entheo_binding_per_frame(pair$e_dmt, pair$f_dmt)
  score_dmt <- mean(abs(pf_dmt))
  pf_pcb <- NULL; score_pcb <- NA_real_; contrast <- NA_real_
  if (!is.null(pair$e_pcb) && !is.null(pair$f_pcb)) {
    pf_pcb <- .entheo_binding_per_frame(pair$e_pcb, pair$f_pcb)
    score_pcb <- mean(abs(pf_pcb))
    contrast <- score_dmt - score_pcb
  }
  list(score = score_dmt,
       score_dmt = score_dmt, score_pcb = score_pcb,
       contrast = contrast,
       per_frame_dmt = pf_dmt, per_frame_pcb = pf_pcb,
       method = "Bayne-Laukkonen Beautiful Loop (v0.4.0-alpha toy)",
       warnings = warnings_vec)
}


#' Self-Aware Networks recurrence score
#'
#' R parity of \code{morie.entheo.san_score}.
#'
#' @param eeg matrix or list.
#' @param fmri matrix or NULL.
#' @return Named list with \code{score_dmt}, \code{score_pcb},
#'   \code{contrast}, per-frame vectors.
#' @keywords internal
san_score <- function(eeg, fmri = NULL) {
  pair <- .entheo_extract_pair(eeg, fmri)
  if (is.null(pair$e_dmt) || is.null(pair$f_dmt)) {
    return(list(score = NA_real_,
                score_dmt = NA_real_, score_pcb = NA_real_,
                contrast = NA_real_,
                per_frame_dmt = NULL, per_frame_pcb = NULL,
                warnings = "EEG or fMRI missing for primary condition"))
  }
  pf_dmt <- .entheo_san_per_frame(pair$e_dmt, pair$f_dmt)
  score_dmt <- mean(pf_dmt)
  pf_pcb <- NULL; score_pcb <- NA_real_; contrast <- NA_real_
  if (!is.null(pair$e_pcb) && !is.null(pair$f_pcb)) {
    pf_pcb <- .entheo_san_per_frame(pair$e_pcb, pair$f_pcb)
    score_pcb <- mean(pf_pcb)
    contrast <- score_dmt - score_pcb
  }
  list(score = score_dmt,
       score_dmt = score_dmt, score_pcb = score_pcb,
       contrast = contrast,
       per_frame_dmt = pf_dmt, per_frame_pcb = pf_pcb,
       method = "Pirez Self-Aware Networks (v0.4.0-alpha toy)",
       warnings = character(0))
}


# ---------------------------------------------------------------------------
# Internal: theoretical-framework helpers (parity with _theory.py)
# ---------------------------------------------------------------------------

.entheo_extract_pair <- function(record_or_eeg, fmri) {
  if (is.list(record_or_eeg) && !is.null(record_or_eeg$fmri)) {
    rec <- record_or_eeg
    return(list(e_dmt = rec$eeg$data_dmt, f_dmt = rec$fmri$data_dmt,
                e_pcb = rec$eeg$data_pcb, f_pcb = rec$fmri$data_pcb))
  }
  list(e_dmt = record_or_eeg, f_dmt = fmri, e_pcb = NULL, f_pcb = NULL)
}

.entheo_envelope <- function(x) {
  kern <- rep(1 / 5, 5)
  abs_x <- abs(x)
  if (is.matrix(abs_x)) {
    t(apply(abs_x, 1, function(row) stats::filter(row, kern, sides = 2)))
  } else {
    as.numeric(stats::filter(abs_x, kern, sides = 2))
  }
}

.entheo_align <- function(e, f) {
  e_tc <- if (is.matrix(e)) colMeans(e) else e
  f_tc <- if (is.matrix(f)) colMeans(f) else f
  n <- min(length(e_tc), length(f_tc))
  if (n == 0) return(list(e = e_tc, f = f_tc))
  .bin <- function(x) {
    if (length(x) == n) return(x)
    step <- length(x) %/% n
    if (step <= 1L) return(x[seq_len(n)])
    trimmed <- x[seq_len(step * n)]
    colMeans(matrix(trimmed, nrow = step))
  }
  list(e = .bin(e_tc), f = .bin(f_tc))
}

.entheo_binding_per_frame <- function(eeg, fmri) {
  env <- .entheo_envelope(eeg)
  al <- .entheo_align(env, fmri)
  e_tc <- al$e; f_tc <- al$f
  n <- min(length(e_tc), length(f_tc))
  if (n < 4) return(rep(0, n))
  # Replace NA introduced by stats::filter padding with 0.
  e_tc[is.na(e_tc)] <- 0
  f_grad <- c(diff(f_tc), 0)
  win <- 7L
  out <- numeric(n)
  for (i in seq_len(n)) {
    a <- max(1L, i - win %/% 2L)
    b <- min(n, i + win %/% 2L)
    if (b - a < 2) next
    x <- e_tc[a:b] - mean(e_tc[a:b])
    y <- f_grad[a:b] - mean(f_grad[a:b])
    denom <- sqrt(sum(x^2) * sum(y^2)) + 1e-9
    out[i] <- sum(x * y) / denom
  }
  out <- out - mean(out)
  sd_o <- stats::sd(out) + 1e-9
  out / sd_o
}

.entheo_san_per_frame <- function(eeg, fmri) {
  al <- .entheo_align(eeg, fmri)
  e_tc <- al$e; f_tc <- al$f
  n <- min(length(e_tc), length(f_tc))
  if (n < 4) return(rep(0, n))
  joint <- rbind(e_tc[seq_len(n)], f_tc[seq_len(n)])
  joint <- (joint - rowMeans(joint)) /
    (apply(joint, 1, stats::sd) + 1e-9)
  win <- 9L
  out <- numeric(n)
  for (i in seq_len(n)) {
    a <- max(1L, i - win %/% 2L)
    b <- min(n, i + win %/% 2L)
    if (b - a < 2) next
    seg <- joint[, a:b, drop = FALSE]
    s0 <- seg[, -ncol(seg), drop = FALSE]
    s1 <- seg[, -1L, drop = FALSE]
    num <- sum(s0 * s1)
    denom <- sqrt(sum(s0^2) * sum(s1^2)) + 1e-9
    out[i] <- num / denom
  }
  out
}
