# SPDX-License-Identifier: MIT OR Apache-2.0

#' Mini-batch stochastic gradient descent for OLS (R parity)
#'
#' @param x Numeric matrix / vector predictors.
#' @param y Numeric response.
#' @param lr Learning rate.
#' @param n_epochs Number of passes over the data.
#' @param batch_size Mini-batch size.
#' @param seed RNG seed for shuffling.
#' @return Named list: estimate, reference_ols, n_epochs, batch_size,
#'   loss, n, method.
#' @export
mini_batch_gradient <- function(x, y, lr = 0.01, n_epochs = 200,
                                batch_size = 32L, seed = 0L) {
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); y <- as.numeric(y)
  n <- nrow(x); p <- ncol(x)
  X1 <- cbind(1, x)
  theta <- rep(0.0, p + 1)
  set.seed(seed)
  for (e in seq_len(n_epochs)) {
    idx <- sample.int(n)
    starts <- seq.int(1L, n, by = batch_size)
    for (s in starts) {
      j <- idx[s:min(s + batch_size - 1L, n)]
      xb <- X1[j, , drop = FALSE]; yb <- y[j]
      grad <- (2 / length(j)) * crossprod(xb, xb %*% theta - yb)
      theta <- theta - lr * as.numeric(grad)
    }
  }
  loss <- mean((X1 %*% theta - y)^2)
  ref <- stats::lm.fit(X1, y)$coefficients
  list(
    estimate      = as.numeric(theta),
    reference_ols = unname(ref),
    n_epochs      = as.integer(n_epochs),
    batch_size    = as.integer(batch_size),
    loss          = as.numeric(loss),
    n             = n,
    method        = "Mini-batch SGD (linear regression)"
  )
}
