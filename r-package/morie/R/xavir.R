# SPDX-License-Identifier: AGPL-3.0-or-later

#' Xavier / Glorot weight initialization
#'
#' R parity for \code{morie.fn.xavir.xavier_init}.
#'
#' \deqn{W \sim \mathcal{N}\!\left(0, \tfrac{2}{n_{in} + n_{out}}\right)}
#' (normal) or
#' \eqn{W \sim U[-\sqrt{6/(n_{in}+n_{out})}, +\sqrt{6/(n_{in}+n_{out})}]}
#' (uniform).
#'
#' @param fan_in Number of input units.
#' @param fan_out Number of output units.
#' @param seed RNG seed.
#' @param uniform Use uniform (TRUE, default) or normal (FALSE).
#' @return Named list \code{(weights, value, fan_in, fan_out, mean, std,
#'   shape, method)}.
#' @references Glorot & Bengio (2010), AISTATS.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
xavir_xavier_init <- function(fan_in, fan_out, seed = 42L, uniform = TRUE) {
  if (fan_in <= 0 || fan_out <= 0)
    stop(sprintf("fan_in and fan_out must be > 0, got %d, %d", fan_in, fan_out))
  old <- .Random.seed_safe()
  on.exit(.Random.seed_restore(old))
  set.seed(seed)
  if (uniform) {
    limit <- sqrt(6 / (fan_in + fan_out))
    W <- matrix(stats::runif(fan_in * fan_out, -limit, limit),
                nrow = fan_in, ncol = fan_out)
  } else {
    sd <- sqrt(2 / (fan_in + fan_out))
    W <- matrix(stats::rnorm(fan_in * fan_out, 0, sd),
                nrow = fan_in, ncol = fan_out)
  }
  list(
    weights = W, value = stats::sd(W),
    fan_in = fan_in, fan_out = fan_out,
    mean = mean(W), std = stats::sd(W),
    shape = c(fan_in, fan_out),
    method = if (uniform) "uniform" else "normal"
  )
}

.Random.seed_safe <- function() {
  if (exists(".Random.seed", envir = globalenv()))
    get(".Random.seed", envir = globalenv())
  else NULL
}

.Random.seed_restore <- function(old) {
  if (is.null(old)) {
    if (exists(".Random.seed", envir = globalenv()))
      rm(".Random.seed", envir = globalenv())
  } else {
    assign(".Random.seed", old, envir = globalenv())
  }
}

#' @rdname xavir_xavier_init
#' @keywords internal
#' @export
xavier_initialization <- xavir_xavier_init
