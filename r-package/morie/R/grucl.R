# SPDX-License-Identifier: GPL-2.0-only

#' GRU cell forward pass
#'
#' R parity for \code{morie.fn.grucl.gru_cell}.  Gates stacked as
#' \code{[W_z; W_r; W_n]}.
#'
#' \deqn{z, r = \sigma(\ldots); \quad n = \tanh(W_n x + r \odot U_n h); \quad
#'       h = (1 - z) \odot n + z \odot h_\text{prev}}
#'
#' @param x Numeric vector.
#' @param h_prev Previous hidden state.
#' @param W,U,b Weight matrices / bias.
#' @param hidden_size Hidden size.
#' @param seed RNG seed.
#' @param deterministic_seed Optional integer; if non-NULL, a SHA-keyed
#'   seed from \code{\link{morie_det_rng}("grucl", deterministic_seed)} is
#'   installed before sampling so Py<->R streams agree.  Overrides
#'   \code{seed} when set.
#' @return Named list \code{(h, estimate, z, r, n, method)}.
#' @references Cho et al. (2014), EMNLP.
#' @export
grucl_gru_cell <- function(x, h_prev = NULL, W = NULL, U = NULL, b = NULL,
                           hidden_size = NULL, seed = 0L,
                           deterministic_seed = NULL) {
  x <- as.numeric(x); n_in <- length(x)
  if (is.null(hidden_size)) {
    hidden_size <- if (!is.null(h_prev)) length(h_prev)
                   else if (!is.null(W)) nrow(as.matrix(W)) %/% 3L
                   else n_in
  }
  H <- as.integer(hidden_size)
  if (is.null(h_prev)) h_prev <- rep(0, H)
  if (!is.null(deterministic_seed)) {
    morie_det_rng("grucl", deterministic_seed)
  } else {
    set.seed(seed)
  }
  if (is.null(W)) W <- matrix(stats::rnorm(3 * H * n_in, 0, 0.1), 3 * H, n_in)
  if (is.null(U)) U <- matrix(stats::rnorm(3 * H * H, 0, 0.1), 3 * H, H)
  if (is.null(b)) b <- rep(0, 3 * H)
  pre <- as.numeric(W %*% x + b)
  Wz <- pre[1:H]; Wr <- pre[(H + 1L):(2 * H)]; Wn <- pre[(2 * H + 1L):(3 * H)]
  Uh <- as.numeric(U %*% h_prev)
  Uz <- Uh[1:H]; Ur <- Uh[(H + 1L):(2 * H)]; Un <- Uh[(2 * H + 1L):(3 * H)]
  sig <- function(z) 1 / (1 + exp(-z))
  z <- sig(Wz + Uz)
  r <- sig(Wr + Ur)
  n <- tanh(Wn + r * Un)
  h <- (1 - z) * n + z * h_prev
  list(h = h, estimate = h, z = z, r = r, n = n,
       method = "GRU cell forward")
}

#' @rdname grucl_gru_cell
#' @keywords internal
#' @export
gru_cell <- grucl_gru_cell
