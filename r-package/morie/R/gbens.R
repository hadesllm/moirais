# SPDX-License-Identifier: GPL-2.0-only

#' Gradient boosting ensemble (R parity)
#'
#' Wraps \code{gbm::gbm} when available, otherwise falls back to
#' \code{xgboost} as a portable boosted-trees backend.
#'
#' @param x Numeric predictor matrix.
#' @param y Response.
#' @param n_estimators Number of boosting iterations.
#' @param learning_rate Shrinkage nu.
#' @param max_depth Depth of each tree.
#' @param task "auto", "classification", or "regression".
#' @param seed RNG seed.
#' @return Named list: estimate, train_score, feature_importances,
#'   n_estimators, learning_rate, max_depth, task, n, method.
#' @importFrom stats predict
#' @export
gradient_boosting_ensemble <- function(x, y, n_estimators = 100L,
                                        learning_rate = 0.1,
                                        max_depth = 3L,
                                        task = "auto", seed = 0L) {
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x)
  if (identical(task, "auto")) {
    task <- if (is.factor(y) || all(y %in% c(0L, 1L)) || is.integer(y))
              "classification" else "regression"
  }
  n <- nrow(x)
  set.seed(seed)
  if (requireNamespace("gbm", quietly = TRUE)) {
    df <- as.data.frame(x); df$.y <- if (task == "classification") as.numeric(as.factor(y)) - 1 else as.numeric(y)
    distribution <- if (task == "classification") "bernoulli" else "gaussian"
    fit <- gbm::gbm(.y ~ ., data = df, distribution = distribution,
                    n.trees = n_estimators, interaction.depth = max_depth,
                    shrinkage = learning_rate, bag.fraction = 1.0,
                    verbose = FALSE)
    p <- gbm::predict.gbm(fit, df, n.trees = n_estimators, type = "response")
    if (task == "classification") {
      preds <- as.integer(p > 0.5)
      train_score <- mean(preds == df$.y)
    } else {
      train_score <- 1 - sum((p - df$.y)^2) / sum((df$.y - mean(df$.y))^2)
    }
    rel <- summary(fit, n.trees = n_estimators, plotit = FALSE)
    fi <- rep(0, ncol(x)); names(fi) <- colnames(df)[seq_len(ncol(x))]
    fi[as.character(rel$var)] <- rel$rel.inf / 100
    backend <- "gbm"
  } else {
    # xgboost fallback
    if (!requireNamespace("xgboost", quietly = TRUE)) {
      stop("install either 'gbm' or 'xgboost' for gradient_boosting_ensemble")
    }
    yv <- if (task == "classification") as.numeric(as.factor(y)) - 1 else as.numeric(y)
    obj <- if (task == "classification") "binary:logistic" else "reg:squarederror"
    fit <- xgboost::xgboost(data = x, label = yv, nrounds = n_estimators,
                             eta = learning_rate, max_depth = max_depth,
                             objective = obj, verbose = 0L)
    p <- predict(fit, x)
    if (task == "classification") {
      preds <- as.integer(p > 0.5)
      train_score <- mean(preds == yv)
    } else {
      train_score <- 1 - sum((p - yv)^2) / sum((yv - mean(yv))^2)
    }
    imp <- xgboost::xgb.importance(model = fit)
    fi <- rep(0, ncol(x)); names(fi) <- colnames(x) %||% paste0("V", seq_len(ncol(x)))
    if (nrow(imp) > 0) fi[imp$Feature] <- imp$Gain
    backend <- "xgboost"
  }
  list(
    estimate            = as.numeric(train_score),
    train_score         = as.numeric(train_score),
    feature_importances = as.numeric(fi),
    n_estimators        = as.integer(n_estimators),
    learning_rate       = as.numeric(learning_rate),
    max_depth           = as.integer(max_depth),
    task                = task,
    backend             = backend,
    n                   = n,
    method              = sprintf("Gradient Boosting (%s, %s)", task, backend)
  )
}
