# SPDX-License-Identifier: MIT OR Apache-2.0

#' Embedding row lookup (Mikolov 2013)
#'
#' @param x Integer token id(s) (0-indexed).
#' @param E Optional embedding matrix; randomised if NULL.
#' @param vocab_size Integer vocabulary size (default 100).
#' @param d_model Integer embedding dimension (default 16).
#' @param seed Integer RNG seed.
#' @return Named list with tensor, E, ids, shape, method.
#' @keywords internal
word_embedding <- function(x, E = NULL, vocab_size = 100L,
                           d_model = 16L, seed = 0L) {
  ids <- as.integer(x)
  if (is.null(E)) {
    set.seed(seed)
    lim <- sqrt(6 / (vocab_size + d_model))
    E <- matrix(stats::runif(vocab_size * d_model, -lim, lim),
                nrow = vocab_size, ncol = d_model)
  }
  if (any(ids < 0L) || any(ids >= nrow(E)))
    stop("token id out of range for embedding matrix")
  e <- E[ids + 1L, , drop = FALSE]   # R is 1-indexed
  list(tensor = e, E = E, ids = ids, shape = dim(e),
       method = "embedding-lookup")
}
