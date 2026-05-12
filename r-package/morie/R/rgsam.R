#' Sample entropy -- Rangayyan Ch 7
#'
#' Richman & Moorman (2000) sample entropy.
#'
#' \deqn{\mathrm{SampEn}(m, r) = -\ln(A / B)}
#'
#' where `B` is the number of unordered template-vector pairs of length
#' `m` within Chebyshev distance `r` and `A` is the count at length
#' `m+1` (self-matches excluded).
#'
#' @param x Numeric vector.
#' @param m Template length (default 2).
#' @param r Tolerance (default `0.2 * sd(x)`).
#' @return Named list `SampEn`, `A`, `B`, `m`, `r`, `n`.
#' @references Richman & Moorman (2000), AJP Heart 278:H2039.
#' @export
#' @examples
#' set.seed(0); rgsam(rnorm(100), m = 2)$SampEn
rgsam <- function(x, m = 2L, r = NULL) {
  N <- length(x)
  if (is.null(r)) r <- 0.2 * stats::sd(x)
  m <- as.integer(m)
  if (N <= m + 1) stop("Need length(x) > m + 1.")
  matches <- function(mm) {
    nT <- N - mm + 1
    M <- matrix(0, nrow = nT, ncol = mm)
    for (i in seq_len(nT)) M[i, ] <- x[i:(i + mm - 1)]
    cnt <- 0L
    for (i in seq_len(nT - 1)) {
      d <- apply(abs(sweep(M[(i + 1):nT, , drop = FALSE], 2, M[i, ])), 1, max)
      cnt <- cnt + sum(d <= r)
    }
    cnt
  }
  B <- matches(m); A <- matches(m + 1L)
  sampen <- if (A == 0 || B == 0) Inf else -log(A / B)
  list(SampEn = sampen, A = A, B = B, m = m, r = r, n = N)
}

#' @rdname rgsam
#' @keywords internal
#' @export
rangayyan_sample_entropy <- rgsam
