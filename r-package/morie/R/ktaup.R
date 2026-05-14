# SPDX-License-Identifier: MIT OR Apache-2.0

#' Kendall partial-tau correlation (Gibbons Ch 12.6)
#'
#' tau_xy.z = (tau_xy - tau_xz tau_yz) /
#'             sqrt((1 - tau_xz^2)(1 - tau_yz^2))
#'
#' @param x,y,z Numeric vectors of equal length.
#' @return Named list: statistic (partial tau), p_value, tau_xy,
#'   tau_xz, tau_yz, z, n.
#' @importFrom stats cor pnorm
#' @export
kendall_tau_partial <- function(x, y, z) {
  x <- as.numeric(x); y <- as.numeric(y); z <- as.numeric(z)
  n <- min(length(x), length(y), length(z))
  if (n < 4) {
    return(list(statistic = NA_real_, p_value = NA_real_,
                tau_xy = NA_real_, tau_xz = NA_real_, tau_yz = NA_real_,
                n = n, method = "Kendall partial tau"))
  }
  x <- x[1:n]; y <- y[1:n]; z <- z[1:n]
  tau_xy <- stats::cor(x, y, method = "kendall")
  tau_xz <- stats::cor(x, z, method = "kendall")
  tau_yz <- stats::cor(y, z, method = "kendall")
  denom <- sqrt((1 - tau_xz^2) * (1 - tau_yz^2))
  if (denom == 0 || !is.finite(denom)) {
    return(list(statistic = NA_real_, p_value = NA_real_,
                tau_xy = tau_xy, tau_xz = tau_xz, tau_yz = tau_yz,
                n = n, method = "Kendall partial tau"))
  }
  tau_p <- (tau_xy - tau_xz * tau_yz) / denom
  z_stat <- tau_p * sqrt(9 * n * (n - 1) / (2 * (2 * n + 5)))
  p <- 2 * (1 - stats::pnorm(abs(z_stat)))
  list(
    statistic = tau_p,
    p_value = p,
    tau_xy = tau_xy,
    tau_xz = tau_xz,
    tau_yz = tau_yz,
    z = z_stat,
    n = n,
    method = "Kendall partial tau (xy controlling z)"
  )
}
