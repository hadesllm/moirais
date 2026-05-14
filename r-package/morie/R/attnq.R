# SPDX-License-Identifier: MIT OR Apache-2.0

#' Scaled dot-product attention
#'
#' R parity for \code{morie.fn.attnq.scaled_dot_product_attention}.
#'
#' \deqn{\mathrm{Attention}(Q, K, V) =
#'       \mathrm{softmax}\!\left(\tfrac{Q K^\top}{\sqrt{d_k}}\right) V}
#'
#' @param Q Numeric matrix \code{(n_q, d_k)}.
#' @param K Numeric matrix \code{(n_k, d_k)} (defaults to \code{Q}).
#' @param V Numeric matrix \code{(n_k, d_v)} (defaults to \code{Q}).
#' @param mask Optional additive mask \code{(n_q, n_k)}.
#' @return Named list \code{(output, estimate, attn, logits, d_k, method)}.
#' @references Vaswani et al. (2017), NeurIPS.
#' @export
attnq_scaled_dot_product_attention <- function(Q, K = NULL, V = NULL,
                                               mask = NULL) {
  Q <- as.matrix(Q)
  if (is.null(K)) K <- Q else K <- as.matrix(K)
  if (is.null(V)) V <- Q else V <- as.matrix(V)
  d_k <- ncol(K)
  logits <- Q %*% t(K) / sqrt(d_k)
  if (!is.null(mask)) logits <- logits + as.matrix(mask)
  # row-wise softmax
  m <- apply(logits, 1L, max)
  e <- exp(sweep(logits, 1L, m, "-"))
  attn <- sweep(e, 1L, rowSums(e), "/")
  out <- attn %*% V
  list(output = out, estimate = out, attn = attn, logits = logits,
       d_k = as.integer(d_k),
       method = "Scaled dot-product attention")
}

#' @rdname attnq_scaled_dot_product_attention
#' @keywords internal
#' @export
scaled_dot_product_attention <- attnq_scaled_dot_product_attention
