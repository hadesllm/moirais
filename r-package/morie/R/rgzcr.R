#' Zero-crossing rate -- Rangayyan Ch 5
#'
#' \deqn{\mathrm{ZCR} = (N-1)^{-1} \sum_{n=1}^{N-1}
#'        \tfrac{1}{2} |\mathrm{sgn}(x[n]) - \mathrm{sgn}(x[n-1])|}
#'
#' @param x Numeric vector.
#' @param fs Sampling frequency (Hz). When `fs > 1`, also returns
#'   crossings per second.
#' @return Named list `zcr`, `zcr_per_second`, `crossings`, `n`.
#' @references Rangayyan Ch 5.
#' @export
#' @examples
#' rgzcr(sin(2 * pi * seq_len(100) / 10), fs = 100)$crossings
rgzcr <- function(x, fs = 1.0) {
  n <- length(x)
  if (n < 2) {
    return(list(zcr = NA_real_, zcr_per_second = NA_real_,
                crossings = 0L, n = n))
  }
  s <- sign(x); s[s == 0] <- 1
  crossings <- sum(abs(diff(s)) > 0)
  zcr <- crossings / (n - 1)
  list(zcr = zcr, zcr_per_second = zcr * fs,
       crossings = as.integer(crossings), n = n)
}

#' @rdname rgzcr
#' @keywords internal
#' @export
rangayyan_zero_crossing <- rgzcr
