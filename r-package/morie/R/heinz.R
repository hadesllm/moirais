# SPDX-License-Identifier: MIT OR Apache-2.0

#' He / Kaiming weight initialization
#'
#' R parity for \code{morie.fn.heinz.he_initialization}.
#'
#' \deqn{W \sim \mathcal{N}\!\left(0, \tfrac{2}{n_{in}}\right)}
#'
#' @param fan_in Number of input units.
#' @param fan_out Number of output units (NULL returns a length-\code{fan_in}
#'   vector).
#' @param seed RNG seed.
#' @param mode One of \code{"normal"} (default) or \code{"uniform"}.
#' @param deterministic_seed Optional integer; if non-NULL, a SHA-keyed
#'   seed from \code{\link{morie_det_rng}("heinz", deterministic_seed)} is
#'   installed before sampling so Py<->R streams agree.  Overrides
#'   \code{seed} when set.
#' @return Named list \code{(W, estimate, mean, std, shape, method)}.
#' @references He, Zhang, Ren & Sun (2015), ICCV.
#' @export
heinz_he_initialization <- function(fan_in, fan_out = NULL, seed = 42L,
                                    mode = "normal",
                                    deterministic_seed = NULL) {
  fan_in <- as.integer(fan_in)
  if (fan_in <= 0) stop(sprintf("fan_in must be > 0, got %d", fan_in))
  if (!is.null(deterministic_seed)) {
    morie_det_rng("heinz", deterministic_seed)
  } else {
    set.seed(seed)
  }
  if (is.null(fan_out)) {
    n <- fan_in
    shape <- fan_in
  } else {
    n <- fan_in * fan_out
    shape <- c(as.integer(fan_out), fan_in)
  }
  if (mode == "normal") {
    sd <- sqrt(2 / fan_in)
    W <- stats::rnorm(n, 0, sd)
  } else if (mode == "uniform") {
    limit <- sqrt(6 / fan_in)
    W <- stats::runif(n, -limit, limit)
  } else {
    stop(sprintf("mode must be 'normal' or 'uniform', got %s", mode))
  }
  if (!is.null(fan_out)) W <- matrix(W, nrow = fan_out, ncol = fan_in)
  list(W = W, estimate = W, mean = mean(W), std = stats::sd(W),
       shape = shape,
       method = sprintf("He initialization (%s)", mode))
}

#' @rdname heinz_he_initialization
#' @keywords internal
#' @export
he_initialization <- heinz_he_initialization
