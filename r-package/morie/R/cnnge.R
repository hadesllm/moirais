# SPDX-License-Identifier: MIT OR Apache-2.0

#' CNN genomic predictor (Conv1D + GAP + dense, base R)
#'
#' @param x Optional fixed-effect design.
#' @param y Numeric response.
#' @param markers (n x m) genotype matrix.
#' @param n_filters,kernel,hidden,n_epochs,lr,l2,seed Hyperparameters.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("cnnge", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return list(estimate, y_hat, W_conv, b_conv, W1, b1, w2, b2, se, n, method).
#' @references Montesinos Lopez Ch 13.
#' @export
cnn_genomic <- function(x, y, markers, n_filters = 8, kernel = 3,
                         hidden = 8, n_epochs = 150, lr = 1e-2,
                         l2 = 1e-3, seed = 0,
                         deterministic_seed = NULL) {
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("cnnge", deterministic_seed)
  } else {
    set.seed(seed)
  }
  y <- as.numeric(y); n <- length(y)
  M <- as.matrix(markers); m <- ncol(M)
  if (kernel > m) kernel <- max(1, m)
  M_mu <- colMeans(M); M_sd <- apply(M, 2, stats::sd); M_sd[M_sd == 0] <- 1
  Ms <- sweep(sweep(M, 2, M_mu), 2, M_sd, "/")
  Wc <- matrix(stats::rnorm(kernel * n_filters, 0, 1 / sqrt(kernel)),
               kernel, n_filters)
  bc <- rep(0, n_filters)
  W1 <- matrix(stats::rnorm(n_filters * hidden, 0, 1 / sqrt(n_filters)),
               n_filters, hidden)
  b1 <- rep(0, hidden)
  w2 <- stats::rnorm(hidden, 0, 1 / sqrt(hidden))
  b2 <- mean(y)
  conv <- function(M_in) {
    L <- m - kernel + 1
    out <- array(0, dim = c(n, L, n_filters))
    for (s in seq_len(L)) {
      seg <- M_in[, s:(s + kernel - 1), drop = FALSE]
      out[, s, ] <- sweep(seg %*% Wc, 2, bc, "+")
    }
    out
  }
  L <- m - kernel + 1
  losses <- numeric(n_epochs)
  for (ep in seq_len(n_epochs)) {
    z <- conv(Ms)
    a <- pmax(z, 0)
    p_mat <- apply(a, c(1, 3), mean)
    h_pre <- sweep(p_mat %*% W1, 2, b1, "+")
    h <- tanh(h_pre)
    y_hat <- as.numeric(h %*% w2) + b2
    resid <- y_hat - y
    dy <- resid / n
    dw2 <- as.numeric(crossprod(h, dy)) + l2 * w2; db2 <- sum(dy)
    dh <- outer(dy, w2); dh_pre <- dh * (1 - h^2)
    dW1 <- crossprod(p_mat, dh_pre) + l2 * W1; db1 <- colSums(dh_pre)
    dp <- dh_pre %*% t(W1)
    da <- array(0, dim = c(n, L, n_filters))
    for (s in seq_len(L)) da[, s, ] <- dp / L
    dz <- da * (z > 0)
    dWc <- matrix(0, kernel, n_filters)
    dbc <- apply(dz, 3, sum)
    for (s in seq_len(L)) {
      seg <- Ms[, s:(s + kernel - 1), drop = FALSE]
      dWc <- dWc + crossprod(seg, dz[, s, ])
    }
    dWc <- dWc + l2 * Wc
    Wc <- Wc - lr * dWc; bc <- bc - lr * dbc
    W1 <- W1 - lr * dW1; b1 <- b1 - lr * db1
    w2 <- w2 - lr * dw2; b2 <- b2 - lr * db2
    losses[ep] <- mean(resid^2)
  }
  z <- conv(Ms); a <- pmax(z, 0); p_mat <- apply(a, c(1, 3), mean)
  h <- tanh(sweep(p_mat %*% W1, 2, b1, "+"))
  y_hat <- as.numeric(h %*% w2) + b2
  resid <- y - y_hat
  list(estimate = mean(y_hat), y_hat = y_hat,
       W_conv = Wc, b_conv = bc, W1 = W1, b1 = b1, w2 = w2, b2 = b2,
       loss_curve = losses, se = sqrt(mean(resid^2)),
       n = n, method = "Conv1D + GAP + dense (base R)")
}

# CANONICAL TEST
# set.seed(7); M <- matrix(rnorm(160), 20, 8); y <- M[,2]+M[,4]+0.2*rnorm(20)
# cnn_genomic(rep(0,20), y, M, n_epochs=20, seed=7)
