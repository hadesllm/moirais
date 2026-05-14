# SPDX-License-Identifier: MIT OR Apache-2.0

#' Metric unfolding for preference data (Schoenemann; Armstrong Ch 7)
#'
#' Closed-form Schoenemann (1970) unfolding via SVD of the doubly
#' centred squared-distance matrix, followed by SMACOF-lite refinement.
#'
#' @param x Preference dissimilarity matrix Delta (n_resp by n_stim).
#' @param k Output dimensionality (default 2).
#' @param n_iter Refinement iterations.
#' @param tol Convergence tolerance.
#' @return Named list with `X`, `Y`, `stress`, `k`, `n_resp`, `n_stim`,
#'   `method`.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
unfdl <- function(x, k = 2L, n_iter = 100L, tol = 1e-6) {
  P <- if (is.matrix(x)) x else stop("x must be a matrix")
  if (nrow(P) < 2L || ncol(P) < 2L)
    return(list(X = matrix(0, 0L, k), Y = matrix(0, 0L, k),
                stress = NA_real_, k = k,
                n_resp = 0L, n_stim = 0L, method = "unfolding"))
  n <- nrow(P); m <- ncol(P)
  P2 <- P^2
  rmeans <- rowMeans(P2); cmeans <- colMeans(P2); gmean <- mean(P2)
  B <- -0.5 * (P2 - matrix(rmeans, n, m) -
               matrix(cmeans, n, m, byrow = TRUE) + gmean)
  sv <- svd(B)
  k_eff <- min(k, length(sv$d))
  Xm <- sv$u[, seq_len(k_eff), drop = FALSE] *
        matrix(sqrt(sv$d[seq_len(k_eff)]), n, k_eff, byrow = TRUE)
  Ym <- sv$v[, seq_len(k_eff), drop = FALSE] *
        matrix(sqrt(sv$d[seq_len(k_eff)]), m, k_eff, byrow = TRUE)
  pairwise <- function(A, B) {
    out <- matrix(0, nrow(A), nrow(B))
    for (i in seq_len(nrow(A))) for (j in seq_len(nrow(B))) {
      out[i, j] <- sqrt(sum((A[i, ] - B[j, ])^2))
    }
    out
  }
  for (it in seq_len(n_iter)) {
    Dh <- pairwise(Xm, Ym) + 1e-12
    ratio <- P / Dh
    Xm_new <- matrix(0, n, k_eff)
    for (i in seq_len(n)) for (d in seq_len(k_eff)) {
      Xm_new[i, d] <- sum(ratio[i, ] * (Xm[i, d] - Ym[, d])) / m +
                      mean(Ym[, d])
    }
    Ym_new <- matrix(0, m, k_eff)
    for (j in seq_len(m)) for (d in seq_len(k_eff)) {
      Ym_new[j, d] <- sum(ratio[, j] * (Ym[j, d] - Xm[, d])) / n +
                      mean(Xm_new[, d])
    }
    delta <- max(abs(Xm_new - Xm), abs(Ym_new - Ym))
    Xm <- Xm_new; Ym <- Ym_new
    if (delta < tol) break
  }
  Dh <- pairwise(Xm, Ym)
  denom <- sum(P^2)
  stress <- if (denom > 0) sqrt(sum((P - Dh)^2) / denom) else NA_real_
  list(X = Xm, Y = Ym, stress = stress, k = k_eff,
       n_resp = n, n_stim = m, method = "unfolding")
}

#' @keywords internal
#' @rdname unfdl
#' @export
unfolding_analysis <- unfdl
