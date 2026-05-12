# SPDX-License-Identifier: GPL-2.0-only

#' XGBoost regularized objective (R parity)
#'
#' Wraps the \code{xgboost} package.  If xgboost isn't installed, falls
#' back to \code{gbm} (gradient boosting) so users still get a usable
#' boosted-trees result; the backend is flagged in the output.
#'
#' @param x Numeric predictor matrix.
#' @param y Response.
#' @param n_estimators Number of boosting rounds.
#' @param learning_rate eta / shrinkage.
#' @param max_depth Tree depth.
#' @param reg_lambda L2 leaf penalty.
#' @param reg_alpha L1 leaf penalty.
#' @param task "auto", "classification", or "regression".
#' @param seed RNG seed.
#' @param deterministic_seed Integer or NULL.  If supplied, the RNG state
#'   is derived from the SHA-keyed [morie_det_rng()] so Py<->R streams
#'   agree on the canonical fixture.  When `NULL` (default), behaviour
#'   is unchanged: `seed` drives `set.seed()` directly.
#' @return Named list: estimate, train_score, feature_importances, backend,
#'   n_estimators, learning_rate, max_depth, reg_lambda, reg_alpha, task,
#'   n, method.
#' @importFrom stats predict
#' @export
xgboost_objective <- function(x, y, n_estimators = 100L, learning_rate = 0.1,
                               max_depth = 3L, reg_lambda = 1.0,
                               reg_alpha = 0.0, task = "auto", seed = 0L,
                               deterministic_seed = NULL) {
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x)
  if (identical(task, "auto")) {
    task <- if (is.factor(y) || all(y %in% c(0L, 1L)) || is.integer(y))
              "classification" else "regression"
  }
  if (!is.null(deterministic_seed)) {
    morie_det_rng("xgbst", deterministic_seed)
  } else {
    set.seed(seed)
  }
  if (requireNamespace("xgboost", quietly = TRUE)) {
    yv <- if (task == "classification") as.integer(as.factor(y)) - 1L else as.numeric(y)
    obj <- if (task == "classification") "binary:logistic" else "reg:squarederror"
    fit <- xgboost::xgboost(data = x, label = yv, nrounds = n_estimators,
                             eta = learning_rate, max_depth = max_depth,
                             lambda = reg_lambda, alpha = reg_alpha,
                             objective = obj, verbose = 0L)
    p <- predict(fit, x)
    if (task == "classification") {
      preds <- as.integer(p > 0.5)
      train_score <- mean(preds == yv)
    } else {
      train_score <- 1 - sum((p - yv)^2) / sum((yv - mean(yv))^2)
    }
    imp <- xgboost::xgb.importance(model = fit)
    fi <- rep(0, ncol(x))
    nms <- colnames(x) %||% paste0("V", seq_len(ncol(x)))
    names(fi) <- nms
    if (nrow(imp) > 0) fi[imp$Feature] <- imp$Gain
    backend <- "xgboost"
  } else {
    # gbm fallback (same objective family, no L1)
    if (!requireNamespace("gbm", quietly = TRUE)) {
      stop("install 'xgboost' (preferred) or 'gbm' for xgboost_objective")
    }
    yv <- if (task == "classification") as.integer(as.factor(y)) - 1L else as.numeric(y)
    df <- as.data.frame(x); df$.y <- yv
    distribution <- if (task == "classification") "bernoulli" else "gaussian"
    fit <- gbm::gbm(.y ~ ., data = df, distribution = distribution,
                    n.trees = n_estimators, interaction.depth = max_depth,
                    shrinkage = learning_rate, bag.fraction = 1.0,
                    verbose = FALSE)
    p <- gbm::predict.gbm(fit, df, n.trees = n_estimators, type = "response")
    if (task == "classification") {
      preds <- as.integer(p > 0.5)
      train_score <- mean(preds == yv)
    } else {
      train_score <- 1 - sum((p - yv)^2) / sum((yv - mean(yv))^2)
    }
    fi <- rep(NA_real_, ncol(x))
    backend <- "gbm"
  }
  list(
    estimate            = as.numeric(train_score),
    train_score         = as.numeric(train_score),
    feature_importances = as.numeric(fi),
    backend             = backend,
    n_estimators        = as.integer(n_estimators),
    learning_rate       = as.numeric(learning_rate),
    max_depth           = as.integer(max_depth),
    reg_lambda          = as.numeric(reg_lambda),
    reg_alpha           = as.numeric(reg_alpha),
    task                = task,
    n                   = nrow(x),
    method              = sprintf("XGBoost-style boosting (%s, %s)", backend, task)
  )
}
