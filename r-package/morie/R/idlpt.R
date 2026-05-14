# SPDX-License-Identifier: MIT OR Apache-2.0

#' Ideal point recovery from unfolding configuration (Armstrong Ch 2)
#'
#' Recovers ideal points from row/respondent coordinates of an
#' unfolding solution. By the unfolding definition the respondent
#' position is the ideal point (Eq 4.36 in Armstrong et al. 2014).
#'
#' @param X_r Row (respondent) coordinates (n by k matrix or vector).
#' @param X_s Column (stimulus) coordinates (m by k); used for the
#'   diagnostic mean stimulus distance (`mean_stim_dist`).
#' @return Named list with `ideal_points`, `n_respondents`, `k`,
#'   `mean_stim_dist`, `method`.
#' @export
idlpt <- function(X_r, X_s = NULL) {
  Xr <- if (is.matrix(X_r)) X_r else matrix(as.numeric(X_r), ncol = 1L)
  out <- Xr
  msd <- NA_real_
  if (!is.null(X_s)) {
    Xs <- if (is.matrix(X_s)) X_s else matrix(as.numeric(X_s), ncol = 1L)
    dvec <- numeric(nrow(Xr) * nrow(Xs))
    idx <- 1L
    for (i in seq_len(nrow(Xr))) for (j in seq_len(nrow(Xs))) {
      dvec[idx] <- sqrt(sum((Xr[i, ] - Xs[j, ])^2)); idx <- idx + 1L
    }
    msd <- mean(dvec)
  }
  list(ideal_points = out, n_respondents = nrow(Xr),
       k = ncol(Xr), mean_stim_dist = msd,
       method = "ideal_point_recovery")
}

#' @keywords internal
#' @rdname idlpt
#' @export
ideal_point_recovery <- idlpt

#' @rdname idlpt
#' @keywords internal
#' @export
ideal_point_model <- idlpt
