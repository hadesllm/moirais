# SPDX-License-Identifier: MIT OR Apache-2.0

#' Heckman-Ichimura-Todd kernel-matching ATE
#'
#' @param x Numeric covariate vector or matrix.
#' @param y Numeric outcome vector.
#' @param treatment Integer/logical treatment indicator (1 = treated).
#' @param bandwidth Optional kernel bandwidth (Silverman default).
#' @param .bootstrap Logical; bootstrap SEs (default TRUE).
#' @return Named list with estimate, se, att, atu, bandwidth, n, method.
#' @keywords internal
hrzt1 <- function(x, y, treatment, bandwidth = NULL, .bootstrap = TRUE) {
  y <- as.numeric(y); D <- as.numeric(treatment)
  X <- if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x)
  n <- length(y)
  if (n < 30 || length(D) != n || nrow(X) != n)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "kernel-matching ATE (insufficient data)"))
  Xp <- if (!isTRUE(all(X[, 1] == 1))) cbind(1, X) else X
  e <- .hrz_logit_newton(D, Xp)
  e <- pmin(pmax(e, 1e-6), 1 - 1e-6)
  h <- if (is.null(bandwidth)) max(.hrz_silverman(e), 1e-3) else as.numeric(bandwidth)
  t_idx <- which(D > 0.5); c_idx <- which(D < 0.5)
  if (length(t_idx) < 2 || length(c_idx) < 2)
    return(list(estimate = NA_real_, se = NA_real_, n = n,
                method = "kernel-matching ATE (one arm empty)"))
  e_t <- e[t_idx]; e_c <- e[c_idx]
  u <- outer(e_t, e_c, `-`) / h; K <- exp(-0.5 * u^2)
  w <- K / pmax(rowSums(K), 1e-12)
  cf_t <- as.numeric(w %*% y[c_idx])
  u2 <- outer(e_c, e_t, `-`) / h; K2 <- exp(-0.5 * u2^2)
  w2 <- K2 / pmax(rowSums(K2), 1e-12)
  cf_c <- as.numeric(w2 %*% y[t_idx])
  att <- mean(y[t_idx] - cf_t); atu <- mean(cf_c - y[c_idx])
  ate <- (length(t_idx) * att + length(c_idx) * atu) / n
  # Bootstrap SE (guarded against recursive blow-up)
  se <- NA_real_
  if (.bootstrap) {
    set.seed(0); B <- 50; boot <- numeric(B)
    for (b in 1:B) {
      idx <- sample.int(n, n, replace = TRUE)
      sub <- tryCatch(hrzt1(X[idx, , drop = FALSE], y[idx], D[idx],
                            bandwidth = h, .bootstrap = FALSE),
                      error = function(e) list(estimate = ate))
      boot[b] <- if (is.numeric(sub$estimate) && !is.na(sub$estimate)) sub$estimate else ate
    }
    se <- as.numeric(stats::sd(boot))
  }
  list(estimate = as.numeric(ate), se = se,
       att = att, atu = atu, bandwidth = h, n = n,
       n_treated = as.integer(length(t_idx)),
       n_control = as.integer(length(c_idx)),
       method = "Kernel-matching ATE (Heckman-Ichimura-Todd)")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzt1
#' @keywords internal
#' @export
horowitz_treatment_effect <- hrzt1
