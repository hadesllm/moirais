# SPDX-License-Identifier: MIT OR Apache-2.0

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
#' @param deterministic_seed Optional integer; if non-NULL, a SHA-keyed
#'   seed from \code{\link{morie_det_rng}("trfbl", deterministic_seed)} is
#'   installed before sampling so Py<->R streams agree.  Overrides
#'   \code{seed} when set.
#' @return Named list \code{(output, estimate, h1, num_heads, d_ff, method)}.
#' @references Vaswani et al. (2017), NeurIPS.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
trfbl_transformer_block <- function(x, num_heads = 2L, d_ff = NULL,
                                    seed = 0L,
                                    deterministic_seed = NULL) {
  x <- as.matrix(x)
  seq_len <- nrow(x); d_model <- ncol(x)
  if (is.null(d_ff)) d_ff <- 4L * d_model

  attn <- mhatf_multi_head_attention_full(x, num_heads = num_heads, seed = seed,
                                          deterministic_seed = deterministic_seed)
  h1 <- .trfbl_layer_norm(x + attn$output)

  if (!is.null(deterministic_seed)) {
    morie_det_rng("trfbl", deterministic_seed)
  } else {
    set.seed(seed + 1L)
  }
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
