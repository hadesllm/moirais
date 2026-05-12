# SPDX-License-Identifier: GPL-2.0-only

#' Decision tree split (R parity)
#'
#' CART tree via \code{rpart::rpart}, returning the root split structure
#' and feature importances.
#'
#' @param x Numeric predictor matrix.
#' @param y Response (factor for classification).
#' @param criterion "gini" or "entropy" -- only "gini" is supported by
#'   rpart for classification; "entropy" maps to information.
#' @param max_depth Max tree depth.
#' @param seed RNG seed.
#' @return Named list: estimate, train_accuracy, root_feature, root_threshold,
#'   root_impurity, n_leaves, feature_importances, criterion, n, method.
#' @importFrom stats predict
#' @export
decision_tree_split <- function(x, y, criterion = "gini", max_depth = 30L,
                                 seed = 0L) {
  if (!requireNamespace("rpart", quietly = TRUE)) {
    stop("Function 'decision_tree_split' requires package 'rpart'. Install with install.packages('rpart').")
  }
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); yf <- as.factor(y)
  colnames(x) <- colnames(x) %||% paste0("x", seq_len(ncol(x)) - 1L)
  parms_split <- if (criterion == "entropy") "information" else "gini"
  set.seed(seed)
  df <- as.data.frame(x); df$.y <- yf
  ctrl <- rpart::rpart.control(maxdepth = max_depth, cp = 0,
                               minsplit = 2L, minbucket = 1L,
                               xval = 0L)
  fit <- rpart::rpart(.y ~ ., data = df, method = "class",
                      parms = list(split = parms_split), control = ctrl)
  fr <- fit$frame
  root <- fr[1, , drop = FALSE]
  # rpart variable names are factor codes via fit$splits / fit$frame$var
  root_feat_name <- as.character(root$var)
  root_feat <- match(root_feat_name, colnames(x)) - 1L  # 0-indexed parity
  splits <- fit$splits
  if (!is.null(splits) && nrow(splits) > 0) {
    root_thr <- as.numeric(splits[1, "index"])
  } else {
    root_thr <- NA_real_
  }
  # rpart impurity = root yval2 deviance / wt; surrogate: Gini at root
  tab <- table(yf)
  pk <- tab / sum(tab)
  root_imp <- if (criterion == "entropy") -sum(ifelse(pk > 0, pk * log(pk), 0))
              else 1 - sum(pk^2)
  preds <- predict(fit, df, type = "class")
  acc <- mean(preds == yf)
  fi <- fit$variable.importance
  fi_full <- rep(0, ncol(x))
  names(fi_full) <- colnames(x)
  if (!is.null(fi)) {
    common <- intersect(names(fi), names(fi_full))
    fi_full[common] <- fi[common]
    if (sum(fi_full) > 0) fi_full <- fi_full / sum(fi_full)
  }
  n_leaves <- sum(fr$var == "<leaf>")
  list(
    estimate            = as.numeric(acc),
    train_accuracy      = as.numeric(acc),
    root_feature        = if (is.na(root_feat)) NA_integer_ else as.integer(root_feat),
    root_threshold      = root_thr,
    root_impurity       = as.numeric(root_imp),
    n_leaves            = as.integer(n_leaves),
    feature_importances = as.numeric(fi_full),
    criterion           = criterion,
    n                   = nrow(x),
    method              = sprintf("Decision tree (CART, %s)", criterion)
  )
}
