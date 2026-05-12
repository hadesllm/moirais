# SPDX-License-Identifier: GPL-2.0-only

#' Vanilla RNN genomic predictor (BPTT, base R)
#'
#' @param x Optional fixed-effect design.
#' @param y Numeric response.
#' @param markers (n x L) marker sequence.
#' @param hidden,n_epochs,lr,l2,seed Hyperparameters.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("rnnge", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return list(estimate, y_hat, W_h, W_x, b_h, w_o, b_o, se, n, method).
#' @references Montesinos Lopez Ch 14.
#' @export
rnn_genomic <- function(x, y, markers, hidden = 8, n_epochs = 150,
                         lr = 1e-2, l2 = 1e-3, seed = 0,
                         deterministic_seed = NULL) {
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("rnnge", deterministic_seed)
  } else {
    set.seed(seed)
  }
  y <- as.numeric(y); n <- length(y)
  M <- as.matrix(markers); L <- ncol(M)
  M_mu <- colMeans(M); M_sd <- apply(M, 2, stats::sd); M_sd[M_sd == 0] <- 1
  Ms <- sweep(sweep(M, 2, M_mu), 2, M_sd, "/")
  H <- hidden
  Wh <- matrix(stats::rnorm(H * H, 0, 1 / sqrt(H)), H, H)
  Wx <- stats::rnorm(H, 0, 1)
  bh <- rep(0, H)
  wo <- stats::rnorm(H, 0, 1 / sqrt(H))
  bo <- mean(y)
  losses <- numeric(n_epochs)
  for (ep in seq_len(n_epochs)) {
    h_state <- matrix(0, n, H)
    hs <- list(h_state)
    for (t in seq_len(L)) {
      xt <- Ms[, t]
      h_state <- tanh(h_state %*% Wh + outer(xt, Wx) + matrix(bh, n, H, byrow = TRUE))
      hs[[t + 1]] <- h_state
    }
    y_hat <- as.numeric(h_state %*% wo) + bo
    resid <- y_hat - y
    dy <- resid / n
    dwo <- as.numeric(crossprod(h_state, dy)) + l2 * wo; dbo <- sum(dy)
    dh <- outer(dy, wo)
    dWh <- matrix(0, H, H); dWx <- 0; dbh <- rep(0, H)
    for (t in rev(seq_len(L))) {
      h_t <- hs[[t + 1]]; h_prev <- hs[[t]]
      dh_raw <- dh * (1 - h_t^2)
      dWh <- dWh + crossprod(h_prev, dh_raw)
      dWx <- dWx + sum(Ms[, t] * dh_raw)
      dbh <- dbh + colSums(dh_raw)
      dh <- dh_raw %*% t(Wh)
    }
    dWh <- dWh + l2 * Wh
    Wh <- Wh - lr * dWh
    Wx <- Wx - lr * (dWx + l2 * Wx)
    bh <- bh - lr * dbh
    wo <- wo - lr * dwo; bo <- bo - lr * dbo
    losses[ep] <- mean(resid^2)
  }
  h_state <- matrix(0, n, H)
  for (t in seq_len(L)) {
    xt <- Ms[, t]
    h_state <- tanh(h_state %*% Wh + outer(xt, Wx) + matrix(bh, n, H, byrow = TRUE))
  }
  y_hat <- as.numeric(h_state %*% wo) + bo
  resid <- y - y_hat
  list(estimate = mean(y_hat), y_hat = y_hat,
       W_h = Wh, W_x = Wx, b_h = bh, w_o = wo, b_o = bo,
       loss_curve = losses, se = sqrt(mean(resid^2)),
       n = n, method = "Vanilla RNN BPTT (base R)")
}

# CANONICAL TEST
# set.seed(8); M <- matrix(rnorm(90), 15, 6); y <- rowSums(M)+0.2*rnorm(15)
# rnn_genomic(rep(0,15), y, M, n_epochs=20, seed=8)
