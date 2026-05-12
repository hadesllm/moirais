#' Heart rate variability (time-domain) -- Rangayyan Ch 6
#'
#' Task-Force-of-the-ESC time-domain HRV indices computed from a
#' sequence of consecutive NN (RR) intervals.
#'
#' \itemize{
#'   \item `meanNN`           mean NN (ms)
#'   \item `SDNN`             SD of NN (ms)
#'   \item `RMSSD`            sqrt mean of (DeltaNN)^2 (ms)
#'   \item `pNN50`            % of |DeltaNN| > 50 ms
#'   \item `heart_rate_bpm`   60000 / meanNN
#' }
#'
#' @param rr_ms Numeric vector of NN intervals in milliseconds.
#' @return Named list with the indices above + `n`.
#' @references Task Force (1996), Circulation 93:1043. Rangayyan Ch 6.
#' @export
#' @examples
#' set.seed(0); rgh <- rghrv(800 + rnorm(200, sd = 40)); rgh$heart_rate_bpm
rghrv <- function(rr_ms) {
  rr <- as.numeric(rr_ms)
  n <- length(rr)
  if (n < 2) stop("Need at least 2 RR intervals.")
  mean_nn <- mean(rr)
  sdnn <- stats::sd(rr)
  d <- diff(rr)
  rmssd <- sqrt(mean(d^2))
  pnn50 <- 100 * mean(abs(d) > 50)
  hr <- if (mean_nn > 0) 60000 / mean_nn else NA_real_
  list(meanNN = mean_nn, SDNN = sdnn, RMSSD = rmssd, pNN50 = pnn50,
       heart_rate_bpm = hr, n = n)
}

#' @rdname rghrv
#' @keywords internal
#' @export
rangayyan_hrv <- rghrv
