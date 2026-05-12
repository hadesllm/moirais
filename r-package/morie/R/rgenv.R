#' Hilbert envelope — Rangayyan Ch 5
#'
#' Analytic-signal envelope via the discrete Hilbert transform.
#'
#' \deqn{\mathrm{env}(t) = |x(t) + j\,H\{x(t)\}|}
#'
#' Uses [seewave::hilbert()] if available, otherwise computes the analytic
#' signal directly from the FFT.
#'
#' @param x Numeric vector.
#' @return Named list `envelope`, `analytic`, `instantaneous_phase`,
#'   `instantaneous_freq`.
#' @references Rangayyan Ch 5.
#' @export
#' @examples
#' t <- seq(0, 1, length.out = 200)
#' x <- cos(2 * pi * 5 * t) * (1 + 0.3 * cos(2 * pi * 0.5 * t))
#' r <- rgenv(x); length(r$envelope)
rgenv <- function(x) {
  N <- length(x)
  X <- stats::fft(x)
  h <- numeric(N)
  if (N %% 2 == 0) {
    h[1] <- 1; h[N / 2 + 1] <- 1
    h[2:(N / 2)] <- 2
  } else {
    h[1] <- 1
    h[2:((N + 1) / 2)] <- 2
  }
  z <- stats::fft(X * h, inverse = TRUE) / N
  env <- Mod(z)
  phase <- Arg(z)
  # unwrap
  dphi <- diff(phase)
  dphi <- ((dphi + pi) %% (2 * pi)) - pi
  phase_unwrapped <- cumsum(c(phase[1], dphi))
  inst_freq <- diff(phase_unwrapped) / (2 * pi)
  list(envelope = env, analytic = z,
       instantaneous_phase = phase_unwrapped,
       instantaneous_freq = inst_freq)
}

#' @rdname rgenv
#' @keywords internal
#' @export
rangayyan_envelope <- rgenv
