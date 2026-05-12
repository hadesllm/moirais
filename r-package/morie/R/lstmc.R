# SPDX-License-Identifier: GPL-2.0-only

#' LSTM cell forward pass
#'
#' R parity for \code{morie.fn.lstmc.lstm_cell}.  Gates stacked as
#' \code{[W_i; W_f; W_g; W_o]}.
#'
#' \deqn{f, i, o = \sigma(\ldots); \quad g = \tanh(\ldots); \quad
#'       c = f \odot c_\text{prev} + i \odot g; \quad
#'       h = o \odot \tanh(c)}
#'
#' @param x Numeric vector (input).
#' @param h_prev,c_prev Previous hidden / cell states (default zeros).
#' @param W,U,b Optional weight matrices and bias.
#' @param hidden_size Hidden size \code{H}.
#' @param seed RNG seed for default weights.
<<<<<<< HEAD
#' @param deterministic_seed Optional integer; if non-NULL, a SHA-keyed
#'   seed from \code{\link{morie_det_rng}("lstmc", deterministic_seed)} is
#'   installed before sampling so Py<->R streams agree.  Overrides
#'   \code{seed} when set.
=======
>>>>>>> origin/main
#' @return Named list \code{(h, c, estimate, i, f, g, o, method)}.
#' @references Hochreiter & Schmidhuber (1997), Neural Computation 9(8).
#' @export
lstmc_lstm_cell <- function(x, h_prev = NULL, c_prev = NULL,
                            W = NULL, U = NULL, b = NULL,
<<<<<<< HEAD
                            hidden_size = NULL, seed = 0L,
                            deterministic_seed = NULL) {
=======
                            hidden_size = NULL, seed = 0L) {
>>>>>>> origin/main
  x <- as.numeric(x); n_in <- length(x)
  if (is.null(hidden_size)) {
    hidden_size <- if (!is.null(h_prev)) length(h_prev)
                   else if (!is.null(W)) nrow(as.matrix(W)) %/% 4L
                   else n_in
  }
  H <- as.integer(hidden_size)
  if (is.null(h_prev)) h_prev <- rep(0, H)
  if (is.null(c_prev)) c_prev <- rep(0, H)
<<<<<<< HEAD
  if (!is.null(deterministic_seed)) {
    morie_det_rng("lstmc", deterministic_seed)
  } else {
    set.seed(seed)
  }
=======
  set.seed(seed)
>>>>>>> origin/main
  if (is.null(W)) W <- matrix(stats::rnorm(4 * H * n_in, 0, 0.1), 4 * H, n_in)
  if (is.null(U)) U <- matrix(stats::rnorm(4 * H * H, 0, 0.1), 4 * H, H)
  if (is.null(b)) b <- rep(0, 4 * H)
  gates <- as.numeric(W %*% x + U %*% h_prev + b)
  sig <- function(z) 1 / (1 + exp(-z))
  i <- sig(gates[1:H])
  f <- sig(gates[(H + 1L):(2 * H)])
  g <- tanh(gates[(2 * H + 1L):(3 * H)])
  o <- sig(gates[(3 * H + 1L):(4 * H)])
  c_new <- f * c_prev + i * g
  h_new <- o * tanh(c_new)
  list(h = h_new, c = c_new, estimate = h_new,
       i = i, f = f, g = g, o = o,
       method = "LSTM cell forward")
}

#' @rdname lstmc_lstm_cell
#' @keywords internal
#' @export
lstm_cell <- lstmc_lstm_cell
