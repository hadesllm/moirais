# SPDX-License-Identifier: GPL-2.0-only

#' Temperature-scaled softmax (Hinton 2015)
#'
#' @param x Numeric vector of logits.
#' @param T Numeric softmax temperature > 0 (default 1).
#' @return Named list with tensor, entropy, T, method.
#' @keywords internal
temperature_scaling <- function(x, T = 1) {
  if (T <= 0) stop("Temperature must be > 0")
  z <- as.numeric(x) / T
  z <- z - max(z)
  p <- exp(z); p <- p / sum(p)
  H <- -sum(ifelse(p > 0, p * log(p), 0))
  list(tensor = p, entropy = H, T = T, method = "temperature-softmax")
}
