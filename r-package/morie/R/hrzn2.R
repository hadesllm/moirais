# SPDX-License-Identifier: GPL-2.0-only

#' Stefanski-Carroll Fourier-deconvolution density estimator
#'
#' @param y Numeric vector of noisy observations.
#' @param sigma_u Noise standard deviation (default 0.5).
#' @param bandwidth Optional kernel bandwidth.
#' @param grid Optional evaluation grid.
#' @param noise Character; noise distribution ("laplace" or "normal").
#' @return Named list with estimate, grid, bandwidth, sigma_u, noise, n, method.
#' @keywords internal
hrzn2 <- function(y, sigma_u = 0.5, bandwidth = NULL, grid = NULL,
                  noise = "laplace") {
  y <- as.numeric(y); n <- length(y)
  if (n < 30) return(list(estimate = NA_real_, n = n,
                          method = "deconvolution (insufficient data)"))
  h <- if (is.null(bandwidth)) max(1.5 * stats::sd(y) * n^(-1/7), 1e-3)
       else as.numeric(bandwidth)
  if (is.null(grid)) grid <- seq(min(y), max(y), length.out = 51)
  grid <- as.numeric(grid)
  T <- seq(-15, 15, length.out = 2049) / max(h, 1e-3)
  dt <- T[2] - T[1]
  phi_Y <- colMeans(exp(1i * outer(y, T)))
  phi_U <- if (noise == "normal") exp(-0.5 * (sigma_u * T)^2)
          else 1 / (1 + (sigma_u * T)^2)
  th <- T * h
  phi_K <- ifelse(abs(th) <= 1, (1 - th^2)^3, 0)
  integrand <- phi_K * phi_Y / ifelse(abs(phi_U) > 1e-10, phi_U, complex(real = Inf))
  f_hat <- numeric(length(grid))
  for (i in seq_along(grid)) {
    f_hat[i] <- Re(sum(exp(-1i * T * grid[i]) * integrand)) * dt / (2 * pi)
  }
  f_hat <- pmax(f_hat, 0)
  list(estimate = f_hat, grid = grid, bandwidth = h,
       sigma_u = as.numeric(sigma_u), noise = noise, n = n,
       method = "Fourier deconvolution density (sinc kernel)")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzn2
#' @keywords internal
#' @export
horowitz_deconvolution <- hrzn2
