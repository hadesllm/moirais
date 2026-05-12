# SPDX-License-Identifier: GPL-2.0-only

#' Multi-head attention with output projection
#'
#' R parity for \code{morie.fn.mhatf.multi_head_attention_full}.
#'
#' \deqn{\mathrm{MultiHead}(x) =
#'       \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O}
#'
#' @param x Numeric matrix \code{(seq_len, d_model)}.
#' @param num_heads Number of heads (must divide \code{d_model}).
#' @param W_q,W_k,W_v Q/K/V projections (default small random normal).
#' @param W_o Output projection (default identity).
#' @param seed RNG seed for default weights.
#' @param deterministic_seed Optional integer; if non-NULL, a SHA-keyed
#'   seed from \code{\link{morie_det_rng}("mhatf", deterministic_seed)} is
#'   installed before sampling so Py<->R streams agree.  Overrides
#'   \code{seed} when set.
#' @return Named list \code{(output, estimate, heads, num_heads, d_k,
#'   d_model, method)}.
#' @references Vaswani et al. (2017), NeurIPS.
#' @export
mhatf_multi_head_attention_full <- function(x, num_heads = 2L,
                                            W_q = NULL, W_k = NULL,
                                            W_v = NULL, W_o = NULL,
                                            seed = 0L,
                                            deterministic_seed = NULL) {
  x <- as.matrix(x)
  seq_len <- nrow(x); d_model <- ncol(x)
  if (d_model %% num_heads != 0L)
    stop(sprintf("d_model=%d must be divisible by num_heads=%d",
                 d_model, num_heads))
  d_k <- d_model %/% num_heads
  if (!is.null(deterministic_seed)) {
    morie_det_rng("mhatf", deterministic_seed)
  } else {
    set.seed(seed)
  }
  rn <- function() matrix(stats::rnorm(d_model * d_model, 0, 1 / sqrt(d_model)),
                          d_model, d_model)
  if (is.null(W_q)) W_q <- rn()
  if (is.null(W_k)) W_k <- rn()
  if (is.null(W_v)) W_v <- rn()
  if (is.null(W_o)) W_o <- diag(d_model)

  Q <- x %*% W_q; K <- x %*% W_k; V <- x %*% W_v
  head_outputs <- vector("list", num_heads)
  head_attns <- vector("list", num_heads)
  for (h in seq_len(num_heads)) {
    cols <- ((h - 1L) * d_k + 1L):(h * d_k)
    res <- attnq_scaled_dot_product_attention(Q[, cols, drop = FALSE],
                                              K[, cols, drop = FALSE],
                                              V[, cols, drop = FALSE])
    head_outputs[[h]] <- res$output
    head_attns[[h]] <- res$attn
  }
  concat <- do.call(cbind, head_outputs)
  out <- concat %*% W_o
  list(output = out, estimate = out, heads = head_attns,
       num_heads = as.integer(num_heads),
       d_k = as.integer(d_k), d_model = as.integer(d_model),
       method = "Multi-head attention")
}

#' @rdname mhatf_multi_head_attention_full
#' @keywords internal
#' @export
multi_head_attention_full <- mhatf_multi_head_attention_full
