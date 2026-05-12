# SPDX-License-Identifier: GPL-2.0-only

#' Transformer encoder block (post-LN)
#'
#' R parity for \code{morie.fn.trfbl.transformer_block}.
#'
#' \deqn{h_1 = \mathrm{LN}(x + \mathrm{MHA}(x)), \quad
#'       h_2 = \mathrm{LN}(h_1 + \mathrm{FFN}(h_1))}
#'
#' @param x Numeric matrix \code{(seq_len, d_model)}.
#' @param num_heads Number of attention heads.
#' @param d_ff Width of feed-forward layer (default \code{4 * d_model}).
#' @param seed RNG seed.
#' @return Named list \code{(output, estimate, h1, num_heads, d_ff, method)}.
#' @references Vaswani et al. (2017), NeurIPS.
#' @export
trfbl_transformer_block <- function(x, num_heads = 2L, d_ff = NULL,
                                    seed = 0L) {
  x <- as.matrix(x)
  seq_len <- nrow(x); d_model <- ncol(x)
  if (is.null(d_ff)) d_ff <- 4L * d_model

  attn <- mhatf_multi_head_attention_full(x, num_heads = num_heads, seed = seed)
  h1 <- .trfbl_layer_norm(x + attn$output)

  set.seed(seed + 1L)
  W1 <- matrix(stats::rnorm(d_model * d_ff, 0, 1 / sqrt(d_model)), d_model, d_ff)
  W2 <- matrix(stats::rnorm(d_ff * d_model, 0, 1 / sqrt(d_ff)), d_ff, d_model)
  ffn <- .trfbl_gelu(h1 %*% W1) %*% W2
  h2 <- .trfbl_layer_norm(h1 + ffn)

  list(output = h2, estimate = h2, h1 = h1,
       num_heads = as.integer(num_heads),
       d_ff = as.integer(d_ff),
       method = "Transformer encoder block (post-LN)")
}

.trfbl_layer_norm <- function(x, eps = 1e-5) {
  mu <- rowMeans(x)
  var <- apply(x, 1L, function(v) mean((v - mean(v))^2))
  sweep(sweep(x, 1L, mu, "-"), 1L, sqrt(var + eps), "/")
}

.trfbl_gelu <- function(z) {
  0.5 * z * (1 + tanh(sqrt(2 / pi) * (z + 0.044715 * z^3)))
}

#' @rdname trfbl_transformer_block
#' @keywords internal
#' @export
transformer_block <- trfbl_transformer_block
