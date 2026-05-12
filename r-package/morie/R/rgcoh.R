#' Magnitude-squared coherence — Rangayyan Ch 4
#'
#' Welch-averaged magnitude-squared coherence::
#'
#' \deqn{C_{xy}(f) = |S_{xy}(f)|^2 / (S_{xx}(f) S_{yy}(f))}
#'
#' Bounded in \[0, 1\].
#'
#' @param x,y Numeric vectors of equal length.
#' @param fs Sampling rate (Hz).
#' @param nperseg Welch segment length (default `min(N, 256)`).
#' @return Named list `freqs`, `coherence`, `mean_coherence`,
#'   `peak_freq`, `peak_coherence`.
#' @references Rangayyan Ch 4.
#' @export
#' @examples
#' \donttest{
#'   set.seed(0); fs <- 100; t <- seq(0, 10, length.out = 1024)
#'   a <- sin(2 * pi * 10 * t); b <- a + 0.1 * rnorm(length(t))
#'   rgcoh(a, b, fs = fs)$peak_coherence > 0.5
#' }
rgcoh <- function(x, y, fs = 1.0, nperseg = NULL) {
  N <- length(x)
  if (length(y) != N) stop("x and y must have equal length.")
  if (is.null(nperseg)) nperseg <- min(N, 256L)
  nperseg <- as.integer(nperseg)
  step <- nperseg %/% 2L
  starts <- seq(1, N - nperseg + 1, by = step)
  if (length(starts) < 1) starts <- 1
  w <- 0.5 - 0.5 * cos(2 * pi * (seq_len(nperseg) - 1) / (nperseg - 1))
  W <- sum(w^2)
  nf <- nperseg %/% 2L + 1L
  freqs <- seq(0, fs / 2, length.out = nf)
  Sxx <- Syy <- numeric(nf); Sxy <- complex(nf)
  for (s in starts) {
    sx <- (x[s:(s + nperseg - 1)] - mean(x[s:(s + nperseg - 1)])) * w
    sy <- (y[s:(s + nperseg - 1)] - mean(y[s:(s + nperseg - 1)])) * w
    X <- stats::fft(sx)[1:nf]
    Y <- stats::fft(sy)[1:nf]
    Sxx <- Sxx + Mod(X)^2
    Syy <- Syy + Mod(Y)^2
    Sxy <- Sxy + X * Conj(Y)
  }
  Cxy <- Mod(Sxy)^2 / (Sxx * Syy)
  peak <- which.max(Cxy)
  list(freqs = freqs, coherence = Cxy,
       mean_coherence = mean(Cxy),
       peak_freq = freqs[peak],
       peak_coherence = Cxy[peak])
}

#' @rdname rgcoh
#' @keywords internal
#' @export
rangayyan_coherence <- rgcoh
