# SPDX-License-Identifier: GPL-2.0-only

#' Grid search with cross-validation (R parity)
#'
#' Wraps \code{caret::train} with method = "glm" (classification) or
#' "lm" (regression) by default; users can pass any caret \code{method}.
#'
#' @param x Numeric predictor matrix.
#' @param y Response.
#' @param method caret method id (default chosen by task).
#' @param tune_grid data.frame of hyperparameter combos to evaluate.
#' @param cv CV folds.
#' @param task "auto", "classification", or "regression".
#' @param seed RNG seed.
#' @return Named list: estimate (best CV score), best_params, best_score,
#'   cv_results_params, cv_results_mean_score, task, n, method.
#' @export
grid_search_cv <- function(x, y, method = NULL, tune_grid = NULL,
                            cv = 5L, task = "auto", seed = 0L) {
  if (!requireNamespace("caret", quietly = TRUE)) {
    stop("Function 'grid_search_cv' requires package 'caret'. Install with install.packages('caret').")
  }
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); colnames(x) <- colnames(x) %||% paste0("x", seq_len(ncol(x)) - 1L)
  if (identical(task, "auto")) {
    task <- if (is.factor(y) || all(y %in% c(0L, 1L)) || is.integer(y))
              "classification" else "regression"
  }
  set.seed(seed)
  ctrl <- caret::trainControl(method = "cv", number = cv, classProbs = FALSE)
  if (is.null(method)) {
    if (task == "classification") {
      method <- "glmnet"
      if (is.null(tune_grid))
        tune_grid <- expand.grid(alpha = 1, lambda = c(0.01, 0.1, 1.0, 10.0))
      y_use <- factor(make.names(as.character(y)))
    } else {
      method <- "ridge"
      if (is.null(tune_grid))
        tune_grid <- expand.grid(lambda = c(0.01, 0.1, 1.0, 10.0))
      y_use <- as.numeric(y)
    }
  } else {
    y_use <- if (task == "classification") factor(make.names(as.character(y))) else as.numeric(y)
  }
  fit <- caret::train(x = x, y = y_use, method = method,
                       tuneGrid = tune_grid, trControl = ctrl)
  best <- fit$bestTune
  results <- fit$results
  metric <- if (task == "classification") "Accuracy" else "RMSE"
  scores <- if (metric %in% names(results)) results[[metric]] else results[[1L]]
  list(
    estimate              = as.numeric(max(scores, na.rm = TRUE)),
    best_params           = as.list(best),
    best_score            = as.numeric(max(scores, na.rm = TRUE)),
    cv_results_params     = results[, setdiff(colnames(results), c("Accuracy", "Kappa", "RMSE", "Rsquared", "MAE")), drop = FALSE],
    cv_results_mean_score = scores,
    task                  = task,
    n                     = nrow(x),
    method                = sprintf("Grid search CV (%s)", method)
  )
}
