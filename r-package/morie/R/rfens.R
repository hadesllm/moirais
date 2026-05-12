# SPDX-License-Identifier: GPL-2.0-only

#' Random Forest ensemble (R parity)
#'
#' Wraps \code{randomForest::randomForest}.  Auto-detects task from y
#' (factor / integer-like -> classification, otherwise regression).
#'
#' @param x Numeric predictor matrix.
#' @param y Response.
#' @param n_estimators Number of trees.
#' @param max_depth Max tree depth (NULL -> unrestricted).
#' @param task "auto", "classification", or "regression".
#' @param seed RNG seed.
#' @return Named list: estimate, train_score, oob_score, feature_importances,
#'   n_estimators, task, n, method.
#' @importFrom stats predict
#' @export
random_forest_ensemble <- function(x, y, n_estimators = 100L,
                                    max_depth = NULL, task = "auto",
                                    seed = 0L) {
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("Function 'random_forest_ensemble' requires package 'randomForest'. Install with install.packages('randomForest').")
  }
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x)
  if (identical(task, "auto")) {
    task <- if (is.factor(y) || all(y %in% c(0L, 1L)) || is.integer(y))
              "classification" else "regression"
  }
  y_use <- if (task == "classification") as.factor(y) else as.numeric(y)
  set.seed(seed)
  args <- list(x = x, y = y_use, ntree = n_estimators, importance = TRUE)
  if (!is.null(max_depth)) args$maxnodes <- 2L^as.integer(max_depth)
  fit <- do.call(randomForest::randomForest, args)
  preds <- predict(fit, x)
  if (task == "classification") {
    train_score <- as.numeric(mean(preds == y_use))
    oob <- 1 - as.numeric(fit$err.rate[n_estimators, "OOB"])
  } else {
    train_score <- 1 - sum((preds - y_use)^2) / sum((y_use - mean(y_use))^2)
    oob <- 1 - fit$mse[n_estimators] / stats::var(y_use)
  }
  fi <- randomForest::importance(fit)
  fi_vec <- if (ncol(fi) > 0) fi[, ncol(fi)] else rep(NA_real_, ncol(x))
  if (sum(fi_vec, na.rm = TRUE) > 0) fi_vec <- fi_vec / sum(fi_vec, na.rm = TRUE)
  list(
    estimate            = as.numeric(train_score),
    train_score         = as.numeric(train_score),
    oob_score           = as.numeric(oob),
    feature_importances = as.numeric(fi_vec),
    n_estimators        = as.integer(n_estimators),
    task                = task,
    n                   = nrow(x),
    method              = sprintf("Random Forest (%s)", task)
  )
}
