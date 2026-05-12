# SPDX-License-Identifier: GPL-2.0-only

#' Random search hyperparameter optimisation (R parity)
#'
#' Uses \code{caret::train} with \code{search = "random"}.
#'
#' @param x Numeric predictor matrix.
#' @param y Response.
#' @param method caret method id (default by task).
#' @param n_iter Number of random draws.
#' @param cv CV folds.
#' @param task "auto" / "classification" / "regression".
#' @param seed RNG seed.
#' @param deterministic_seed Integer or NULL.  If supplied, the RNG state
#'   is derived from the SHA-keyed [morie_det_rng()] so Py<->R streams
#'   agree on the canonical fixture.  When `NULL` (default), behaviour
#'   is unchanged: `seed` drives `set.seed()` directly.
#' @return Named list: estimate, best_params, best_score, sampled_params,
#'   sampled_scores, n_iter, task, n, method.
#' @export
random_search_cv <- function(x, y, method = NULL, n_iter = 20L, cv = 5L,
                              task = "auto", seed = 0L,
                              deterministic_seed = NULL) {
  if (!requireNamespace("caret", quietly = TRUE)) {
    stop("Function 'random_search_cv' requires package 'caret'. Install with install.packages('caret').")
  }
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); colnames(x) <- colnames(x) %||% paste0("x", seq_len(ncol(x)) - 1L)
  if (identical(task, "auto")) {
    task <- if (is.factor(y) || all(y %in% c(0L, 1L)) || is.integer(y))
              "classification" else "regression"
  }
  if (!is.null(deterministic_seed)) {
    morie_det_rng("rndsr", deterministic_seed)
  } else {
    set.seed(seed)
  }
  ctrl <- caret::trainControl(method = "cv", number = cv,
                               search = "random", classProbs = FALSE)
  if (is.null(method)) {
    method <- if (task == "classification") "glmnet" else "ridge"
  }
  y_use <- if (task == "classification") factor(make.names(as.character(y))) else as.numeric(y)
  fit <- caret::train(x = x, y = y_use, method = method,
                       tuneLength = n_iter, trControl = ctrl)
  best <- fit$bestTune
  results <- fit$results
  metric <- if (task == "classification") "Accuracy" else "RMSE"
  scores <- if (metric %in% names(results)) results[[metric]] else results[[1L]]
  list(
    estimate        = as.numeric(max(scores, na.rm = TRUE)),
    best_params     = as.list(best),
    best_score      = as.numeric(max(scores, na.rm = TRUE)),
    sampled_params  = results[, setdiff(colnames(results), c("Accuracy", "Kappa", "RMSE", "Rsquared", "MAE")), drop = FALSE],
    sampled_scores  = scores,
    n_iter          = as.integer(n_iter),
    task            = task,
    n               = nrow(x),
    method          = sprintf("Random search CV (%s)", method)
  )
}
