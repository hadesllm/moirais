#' AR(p) Burg estimator -- Rangayyan Ch 4
#'
#' Burg's recursion for autoregressive coefficients and innovation
#' variance. Always yields a stable (minimum-phase) all-pole model.
#'
#' Sign convention: \eqn{x[n] = -\sum_{k=1}^p a_k x[n-k] + e[n]}.
#'
#' @param x Numeric vector.
#' @param order AR order (default 10).
#' @return Named list `ar_coeffs` (length p), `variance`, `order`,
#'   `reflection` (PARCOR coefficients).
#' @references Burg (1975); Marple (1987); Rangayyan Ch 4.
#' @export
#' @examples
#' set.seed(0); r <- rgarb(rnorm(500), order = 4); length(r$ar_coeffs)
rgarb <- function(x, order = 10L) {
  N <- length(x); p <- as.integer(order)
  if (p < 1 || p >= N) stop("order must be 1 <= p < length(x).")
  f <- as.numeric(x); b <- as.numeric(x)
  a <- c(1, rep(0, p))
  var_ <- mean(x^2)
  k <- numeric(p)
  for (m in 0:(p - 1)) {
    num <- -2 * sum(f[(m + 2):N] * b[(m + 1):(N - 1)])
    den <- sum(f[(m + 2):N]^2) + sum(b[(m + 1):(N - 1)]^2)
    km <- if (den > 0) num / den else 0
    k[m + 1] <- km
    new_a <- a
    for (i in 1:(m + 1)) {
      new_a[i + 1] <- a[i + 1] + km * a[m + 2 - i]
    }
    a <- new_a
    f_new <- f[(m + 2):N] + km * b[(m + 1):(N - 1)]
    b_new <- b[(m + 1):(N - 1)] + km * f[(m + 2):N]
    f[(m + 2):N] <- f_new
    b[(m + 2):N] <- b_new
    var_ <- var_ * (1 - km^2)
  }
  list(ar_coeffs = a[-1], variance = var_, order = p, reflection = k)
}

#' @rdname rgarb
#' @keywords internal
#' @export
rangayyan_ar_burg <- rgarb
