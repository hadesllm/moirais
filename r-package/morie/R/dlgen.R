# SPDX-License-Identifier: GPL-2.0-only

#' Single-hidden-layer MLP genomic predictor (base R)
#'
#' @param x Fixed-effect design (optional).
#' @param y Numeric response.
#' @param markers Genotype matrix (n x m).
#' @param hidden Hidden units (default 16).
#' @param n_epochs Training epochs.
#' @param lr Learning rate.
#' @param l2 L2 weight decay.
#' @param seed Seed.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("dlgen", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return list(estimate, y_hat, beta, W1, b1, w2, b2, se, n, method).
#' @references Montesinos Lopez Ch 12.
#' @export
deep_learning_genomic <- function(x, y, markers, hidden = 16,
                                   n_epochs = 200, lr = 1e-2,
                                   l2 = 1e-3, seed = 0,
                                   deterministic_seed = NULL) {
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("dlgen", deterministic_seed)
  } else {
    set.seed(seed)
  }
  y <- as.numeric(y); n <- length(y)
  M <- as.matrix(markers); m <- ncol(M)
  M_mu <- colMeans(M); M_sd <- apply(M, 2, stats::sd); M_sd[M_sd == 0] <- 1
  Ms <- sweep(sweep(M, 2, M_mu), 2, M_sd, "/")
  W1 <- matrix(stats::rnorm(m * hidden, 0, 1 / sqrt(m)), m, hidden)
  b1 <- rep(0, hidden)
  w2 <- stats::rnorm(hidden, 0, 1 / sqrt(hidden))
  b2 <- mean(y)
  losses <- numeric(n_epochs)
  for (ep in seq_len(n_epochs)) {
    z1 <- sweep(Ms %*% W1, 2, b1, "+")
    h <- tanh(z1)
    y_hat <- as.numeric(h %*% w2) + b2
    resid <- y_hat - y
    dy <- resid / n
    dw2 <- as.numeric(crossprod(h, dy)) + l2 * w2
    db2 <- sum(dy)
    dh <- outer(dy, w2)
    dz1 <- dh * (1 - h^2)
    dW1 <- crossprod(Ms, dz1) + l2 * W1
    db1 <- colSums(dz1)
    W1 <- W1 - lr * dW1; b1 <- b1 - lr * db1
    w2 <- w2 - lr * dw2; b2 <- b2 - lr * db2
    losses[ep] <- mean(resid^2)
  }
  z1 <- sweep(Ms %*% W1, 2, b1, "+"); h <- tanh(z1)
  y_hat <- as.numeric(h %*% w2) + b2
  resid <- y - y_hat
  list(estimate = mean(y_hat), y_hat = y_hat, beta = numeric(0),
       W1 = W1, b1 = b1, w2 = w2, b2 = b2,
       loss_curve = losses, se = sqrt(mean(resid^2)),
       n = n, method = "MLP-1H base-R")
}

# CANONICAL TEST
# set.seed(6); M <- matrix(rnorm(100), 20, 5)
# y <- M[,1] + 0.3*rnorm(20); deep_learning_genomic(rep(0,20), y, M, seed=6)
