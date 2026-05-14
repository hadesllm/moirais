# SPDX-License-Identifier: MIT OR Apache-2.0

#' Dropout forward pass (inverted)
#'
#' R parity for \code{morie.fn.drpfw.dropout_forward}.
#'
#' \deqn{y = x \odot m / (1-p), \quad m \sim \mathrm{Bernoulli}(1-p)}
#'
#' @param x Numeric array.
#' @param p Drop probability in \code{[0, 1)}.
#' @param seed RNG seed.
#' @param training If FALSE, returns input unchanged.
#' @param deterministic_seed Optional integer; if non-NULL, a SHA-keyed
#'   seed from \code{\link{morie_det_rng}("drpfw", deterministic_seed)} is
#'   installed before sampling so Py<->R streams agree.  Overrides
#'   \code{seed} when set.
#' @return Named list \code{(y, estimate, mask, p, kept_fraction, method)}.
#' @references Srivastava et al. (2014), JMLR 15:1929-1958.
#' @export
drpfw_dropout_forward <- function(x, p = 0.5, seed = 0L, training = TRUE,
                                  deterministic_seed = NULL) {
  if (p < 0 || p >= 1) stop(sprintf("p must be in [0, 1), got %g", p))
  x <- as.array(x)
  if (!training || p == 0) {
    return(list(y = x, estimate = x, mask = array(1, dim(x)), p = p,
                kept_fraction = 1.0,
                method = "Dropout (pass-through)"))
  }
  if (!is.null(deterministic_seed)) {
    morie_det_rng("drpfw", deterministic_seed)
  } else {
    set.seed(seed)
  }
  mask <- array((stats::runif(length(x)) >= p) * 1.0, dim = dim(x))
  y <- x * mask / (1 - p)
  list(y = y, estimate = y, mask = mask, p = p,
       kept_fraction = mean(mask),
       method = "Dropout forward (inverted)")
}

#' @rdname drpfw_dropout_forward
#' @keywords internal
#' @export
dropout_forward <- drpfw_dropout_forward
