# SPDX-License-Identifier: MIT OR Apache-2.0

#' Learning curve -- train / val MSE vs training-set size (R parity)
#'
#' Manual implementation of the sklearn learning_curve flow: shuffle,
#' split into k folds, for each train-fraction fit on a prefix of the
#' training fold and score on the held-out fold.
#'
#' @param x Numeric matrix predictors.
#' @param y Numeric response.
#' @param sizes Training-set fractions (default seq(0.1, 1.0, length=5)).
#' @param cv Number of CV folds.
#' @param seed RNG seed for shuffling.
#' @return Named list: estimate (final val MSE), train_sizes, train_scores,
#'   val_scores, n, method.
#' @export
learning_curve <- function(x, y, sizes = NULL, cv = 5L, seed = 0L) {
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); y <- as.numeric(y)
  n <- nrow(x)
  if (is.null(sizes)) sizes <- seq(0.1, 1.0, length.out = 5)
  set.seed(seed)
  idx <- sample.int(n)
  folds <- cut(seq_len(n), breaks = cv, labels = FALSE)
  fold_idx <- split(idx, folds)
  train_scores <- numeric(length(sizes))
  val_scores   <- numeric(length(sizes))
  train_sizes  <- integer(length(sizes))
  for (k in seq_along(sizes)) {
    tr_mse <- numeric(cv); va_mse <- numeric(cv)
    for (f in seq_len(cv)) {
      val_id <- fold_idx[[f]]
      tr_pool <- setdiff(idx, val_id)
      m <- max(1L, floor(sizes[k] * length(tr_pool)))
      tr_id <- tr_pool[seq_len(m)]
      train_sizes[k] <- m
      df_tr <- as.data.frame(x[tr_id, , drop = FALSE]); df_tr$.y <- y[tr_id]
      df_va <- as.data.frame(x[val_id, , drop = FALSE]); df_va$.y <- y[val_id]
      fit <- stats::lm(.y ~ ., data = df_tr)
      tr_mse[f] <- mean((stats::predict(fit, newdata = df_tr) - y[tr_id])^2)
      va_mse[f] <- mean((stats::predict(fit, newdata = df_va) - y[val_id])^2)
    }
    train_scores[k] <- mean(tr_mse)
    val_scores[k]   <- mean(va_mse)
  }
  list(
    estimate     = as.numeric(val_scores[length(val_scores)]),
    train_sizes  = train_sizes,
    train_scores = train_scores,
    val_scores   = val_scores,
    n            = n,
    method       = "Learning curve (cv MSE)"
  )
}
