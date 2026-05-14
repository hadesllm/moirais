# SPDX-License-Identifier: MIT OR Apache-2.0

#' Multinomial probit (spatial choice; Armstrong Ch 9)
#'
#' GHK-style Monte-Carlo choice probabilities for independent Gaussian
#' errors over a deterministic utility matrix; binary case uses the
#' closed-form Phi((U1-U0)/sqrt(2)).
#'
#' @param x Utility matrix (n_obs by n_alt).
#' @param n_draws Number of MC draws (default 2000).
#' @param seed RNG seed.
#' @return Named list with `probs`, `max_alt`, `n_obs`, `n_alt`,
#'   `method`.
#' @export
mnpbt <- function(x, n_draws = 2000L, seed = 0L) {
  U <- if (is.matrix(x)) x else matrix(as.numeric(x), nrow = 1L)
  n <- nrow(U); J <- ncol(U)
  if (J < 2L)
    return(list(probs = matrix(1, n, J),
                max_alt = rep(1L, n), n_obs = n, n_alt = J,
                method = "multinomial_probit"))
  set.seed(seed)
  draws <- array(stats::rnorm(n_draws * n * J), dim = c(n_draws, n, J))
  Y <- sweep(draws, c(2L, 3L), U, FUN = "+")
  picks <- apply(Y, c(1L, 2L), which.max)  # n_draws by n
  probs <- matrix(0, n, J)
  for (j in seq_len(J)) probs[, j] <- colMeans(picks == j)
  if (J == 2L) {
    probs[, 2] <- stats::pnorm((U[, 2] - U[, 1]) / sqrt(2))
    probs[, 1] <- 1 - probs[, 2]
  }
  list(probs = probs, max_alt = max.col(probs),
       n_obs = n, n_alt = J, method = "multinomial_probit")
}

#' @keywords internal
#' @rdname mnpbt
#' @export
multinomial_probit_spatial <- mnpbt
