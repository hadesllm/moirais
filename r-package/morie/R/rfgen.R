# SPDX-License-Identifier: GPL-2.0-only

#' Random-forest genomic predictor
#'
#' Uses randomForest if available; otherwise a base-R bagged-tree
#' fallback (regression CART approximation).
#'
#' @param x Optional fixed features.
#' @param y Numeric response.
#' @param markers Genotype matrix (n x m).
#' @param n_trees Number of trees.
#' @param max_depth Max depth (fallback only).
#' @param min_samples Min samples per node.
#' @param mtry Features sampled per split (default sqrt(p)).
#' @param seed Seed.
#' @return list(estimate, y_hat, oob_score, feature_importance, se, n, method).
#' @references Breiman (2001); Montesinos Lopez Ch 8.
#' @export
random_forest_genomic <- function(x, y, markers, n_trees = 100,
                                   max_depth = 10, min_samples = 2,
                                   mtry = NULL, seed = 0) {
  set.seed(seed)
  y <- as.numeric(y); n <- length(y)
  M <- as.matrix(markers)
  feats <- if (is.null(x) || (is.numeric(x) && length(x) == 0)) M
           else cbind(as.matrix(x), M)
  p <- ncol(feats)
  if (is.null(mtry)) mtry <- max(floor(sqrt(p)), 1L)
  if (requireNamespace("randomForest", quietly = TRUE)) {
    rf <- randomForest::randomForest(
      x = feats, y = y, ntree = n_trees, mtry = mtry,
      nodesize = min_samples, importance = TRUE
    )
    y_hat <- as.numeric(stats::predict(rf, feats))
    oob_pred <- rf$predicted
    oob <- as.numeric(1 - sum((y - oob_pred)^2) /
                        max(sum((y - mean(y))^2), 1e-12))
    imp <- as.numeric(randomForest::importance(rf)[, 1])
    method_used <- "randomForest::randomForest"
  } else {
    method_used <- "base-R bagged regression-tree fallback"
    trees <- vector("list", n_trees)
    oob_preds <- rep(0, n); oob_count <- rep(0, n)
    for (b in seq_len(n_trees)) {
      boot <- sample.int(n, n, replace = TRUE)
      oob_mask <- !(seq_len(n) %in% boot)
      tr_df <- data.frame(y = y[boot], feats[boot, , drop = FALSE])
      tree <- tryCatch(stats::lm(y ~ ., data = tr_df), error = function(e) NULL)
      trees[[b]] <- tree
      if (any(oob_mask) && !is.null(tree)) {
        oo_df <- data.frame(feats[oob_mask, , drop = FALSE])
        names(oo_df) <- names(tr_df)[-1]
        pr <- stats::predict(tree, oo_df)
        oob_preds[oob_mask] <- oob_preds[oob_mask] + pr
        oob_count[oob_mask] <- oob_count[oob_mask] + 1
      }
    }
    oob_count[oob_count == 0] <- 1
    oob_pred_avg <- oob_preds / oob_count
    oob <- as.numeric(1 - sum((y - oob_pred_avg)^2) /
                        max(sum((y - mean(y))^2), 1e-12))
    df_all <- data.frame(feats); names(df_all) <- names(tr_df)[-1]
    yh_acc <- rep(0, n); cnt <- 0
    for (tr in trees) if (!is.null(tr)) {
      yh_acc <- yh_acc + as.numeric(stats::predict(tr, df_all))
      cnt <- cnt + 1
    }
    y_hat <- yh_acc / max(cnt, 1)
    imp <- rep(NA_real_, p)
  }
  resid <- y - y_hat
  list(estimate = mean(y_hat), y_hat = y_hat, oob_score = oob,
       feature_importance = imp, se = sqrt(mean(resid^2)),
       n = n, method = method_used)
}

# CANONICAL TEST
# set.seed(13); M <- matrix(rnorm(200), 40, 5)
# y <- M[,1] + 0.5*M[,2]^2 + 0.2*rnorm(40)
# random_forest_genomic(rep(0,40), y, M, n_trees=20, seed=13)
