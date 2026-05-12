# SPDX-License-Identifier: GPL-2.0-only

#' t-SNE for non-linear dimension reduction (R parity)
#'
#' Wraps \code{Rtsne::Rtsne}.
#'
#' @param x Numeric matrix.
#' @param n_components Embedding dimension.
#' @param perplexity t-SNE perplexity.
#' @param learning_rate Unused by Rtsne (kept for API parity).
#' @param n_iter Max iterations.
#' @param seed RNG seed.
#' @return Named list: estimate (shape), embedding, kl_divergence,
#'   perplexity, n_components, n, method.
#' @export
tsne_reduction <- function(x, n_components = 2L, perplexity = 30,
                            learning_rate = "auto", n_iter = 1000L,
                            seed = 0L) {
  if (!requireNamespace("Rtsne", quietly = TRUE)) {
    stop("Function 'tsne_reduction' requires package 'Rtsne'. Install with install.packages('Rtsne').")
  }
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x)
  n <- nrow(x)
  set.seed(seed)
  ts <- Rtsne::Rtsne(x, dims = n_components, perplexity = perplexity,
                     max_iter = n_iter, check_duplicates = FALSE,
                     verbose = FALSE, pca = TRUE)
  emb <- ts$Y
  list(
    estimate      = dim(emb),
    embedding     = emb,
    kl_divergence = as.numeric(ts$itercosts[length(ts$itercosts)]),
    perplexity    = as.numeric(perplexity),
    n_components  = as.integer(n_components),
    n             = n,
    method        = "t-SNE (van der Maaten 2008)"
  )
}
