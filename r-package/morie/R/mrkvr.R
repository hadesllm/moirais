# SPDX-License-Identifier: GPL-2.0-only

#' Marker variance-component estimation
#'
#' Reports sigma_m^2 via the VanRaden split sigma_g^2 / (2 sum p_j q_j)
#' alongside the naive sigma_g^2 / p form. sigma_g^2 is obtained from a
#' quick GBLUP fit.
#'
#' @param x Fixed-effect design (optional).
#' @param y Numeric response.
#' @param markers (n × m) genotype matrix coded 0/1/2.
#' @return list(estimate, sigma_g2, sigma_e2, h2, sigma_m2_vanraden,
#'   sigma_m2_naive, sum_2pq, p_freq, n, p, method).
#' @references VanRaden (2008); Montesinos Lopez Ch 3.
#' @export
marker_variance <- function(x, y, markers) {
  y <- as.numeric(y); n <- length(y)
  M <- as.matrix(markers); m <- ncol(M)
  cand <- if (is.null(x) || (is.numeric(x) && length(x) == 0)) matrix(1, n, 1)
          else cbind(1, as.matrix(x))
  qrx <- qr(cand); X <- cand[, qrx$pivot[seq_len(qrx$rank)], drop = FALSE]
  p_freq <- colMeans(M) / 2
  sum_2pq <- 2 * sum(p_freq * (1 - p_freq))
  if (sum_2pq <= 0) sum_2pq <- 1
  G <- grm_vanraden(M, method = 1)$estimate + 1e-6 * diag(n)
  Ginv <- solve(G)
  lam <- 1
  px <- ncol(X)
  C <- rbind(cbind(crossprod(X), t(X)),
             cbind(X,            diag(n) + lam * Ginv))
  rhs <- c(crossprod(X, y), y)
  sol <- solve(C, rhs)
  beta <- sol[seq_len(px)]; g_hat <- sol[(px + 1):(px + n)]
  sigma_g2 <- if (n > 1) stats::var(g_hat) else 0
  resid <- y - X %*% beta - g_hat
  sigma_e2 <- if (n > 1) stats::var(as.numeric(resid)) else 0
  sigma_m2_v <- sigma_g2 / sum_2pq
  sigma_m2_n <- sigma_g2 / max(m, 1)
  h2 <- if (sigma_g2 + sigma_e2 > 0) sigma_g2 / (sigma_g2 + sigma_e2)
        else NA_real_
  list(estimate = sigma_m2_v, sigma_g2 = sigma_g2, sigma_e2 = sigma_e2,
       h2 = h2, sigma_m2_vanraden = sigma_m2_v, sigma_m2_naive = sigma_m2_n,
       sum_2pq = sum_2pq, p_freq = p_freq, n = n, p = m,
       method = "VanRaden + naive marker-variance split")
}

# CANONICAL TEST
# set.seed(16); M <- matrix(sample(0:2, 160, TRUE), 20, 8)
# y <- M %*% rnorm(8) + 0.5*rnorm(20)
# marker_variance(rep(0, 20), y, M)$h2
