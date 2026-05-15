# SPDX-License-Identifier: AGPL-3.0-or-later

#' Support-vector regression for genomic prediction
#'
#' @param x Optional fixed-effect features.
#' @param y Numeric response.
#' @param markers (n x m) genotype matrix.
#' @param C Cost (default 1).
#' @param epsilon SVR tube width (default 0.1).
#' @param gamma RBF kernel scale ("scale" = 1/(m * var(M)) or numeric).
#' @return list(estimate, y_hat, alpha, support_indices, se, n, method).
#' @references Vapnik (1995); Montesinos Lopez Ch 7.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
svm_genomic <- function(x, y, markers, C = 1, epsilon = 0.1,
                         gamma = "scale") {
  y <- as.numeric(y); n <- length(y)
  M <- as.matrix(markers)
  feats <- if (is.null(x) || (is.numeric(x) && length(x) == 0)) M
           else cbind(as.matrix(x), M)
  use_e <- requireNamespace("e1071", quietly = TRUE)
  method_used <- "Kernel-ridge RBF fallback (no e1071)"
  if (use_e) {
    method_used <- "e1071 eps-SVR (RBF)"
    g_val <- if (identical(gamma, "scale")) {
      v <- stats::var(as.numeric(M)); if (!is.finite(v) || v <= 0) v <- 1
      1 / (ncol(M) * v)
    } else as.numeric(gamma)
    fit <- e1071::svm(feats, y, type = "eps-regression", kernel = "radial",
                       cost = C, epsilon = epsilon, gamma = g_val)
    y_hat <- as.numeric(stats::predict(fit, feats))
    alpha <- as.numeric(fit$coefs)
    sv_idx <- as.integer(fit$index)
    intercept <- as.numeric(-fit$rho)
  } else {
    g_val <- if (identical(gamma, "scale")) {
      v <- stats::var(as.numeric(M)); if (!is.finite(v) || v <= 0) v <- 1
      1 / (ncol(M) * v)
    } else as.numeric(gamma)
    sq <- rowSums(feats^2)
    D2 <- pmax(outer(sq, sq, "+") - 2 * tcrossprod(feats), 0)
    K <- exp(-g_val * D2)
    intercept <- mean(y); yc <- y - intercept
    alpha <- as.numeric(solve(K + (1 / max(C, 1e-8)) * diag(n), yc))
    y_hat <- as.numeric(K %*% alpha) + intercept
    sv_idx <- which(abs(alpha) > 1e-6)
  }
  resid <- y - y_hat
  list(estimate = mean(y_hat), y_hat = y_hat, alpha = alpha,
       support_indices = sv_idx, intercept = intercept,
       se = sqrt(mean(resid^2)), n = n, method = method_used)
}

# CANONICAL TEST
# set.seed(12); M <- matrix(rnorm(100), 25, 4); y <- sin(M[,1])+0.2*rnorm(25)
# svm_genomic(rep(0, 25), y, M)
