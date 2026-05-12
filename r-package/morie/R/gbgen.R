# SPDX-License-Identifier: GPL-2.0-only

#' Gradient-boosting genomic predictor (Friedman 2001)
#'
#' Uses gbm if available; otherwise base-R boosted stumps.
#'
#' @param x Optional fixed features.
#' @param y Numeric response.
#' @param markers (n × m) genotype matrix.
#' @param n_estimators Boosting rounds.
#' @param learning_rate Shrinkage.
#' @param max_depth Tree depth (gbm only).
#' @param seed Seed.
#' @return list(estimate, y_hat, train_loss, se, n, method).
#' @references Friedman (2001); Montesinos Lopez Ch 9.
#' @export
gradient_boosting_genomic <- function(x, y, markers, n_estimators = 100,
                                       learning_rate = 0.1, max_depth = 3,
                                       seed = 0) {
  set.seed(seed)
  y <- as.numeric(y); n <- length(y)
  M <- as.matrix(markers)
  feats <- if (is.null(x) || (is.numeric(x) && length(x) == 0)) M
           else cbind(as.matrix(x), M)
  if (requireNamespace("gbm", quietly = TRUE)) {
    df <- data.frame(y = y, feats); colnames(df)[-1] <- paste0("V", seq_len(ncol(feats)))
    gb <- gbm::gbm(y ~ ., data = df, distribution = "gaussian",
                    n.trees = n_estimators, shrinkage = learning_rate,
                    interaction.depth = max_depth, n.minobsinnode = 2,
                    bag.fraction = 1, verbose = FALSE)
    y_hat <- as.numeric(stats::predict(gb, df, n.trees = n_estimators))
    train_loss <- as.numeric(gb$train.error)
    method_used <- "gbm::gbm"
  } else {
    method_used <- "base-R boosted stumps fallback"
    F_pred <- rep(mean(y), n)
    train_loss <- numeric(n_estimators)
    stumps <- vector("list", n_estimators)
    p_ <- ncol(feats)
    for (b_ in seq_len(n_estimators)) {
      r <- y - F_pred
      best <- NULL
      for (f in seq_len(p_)) {
        vals <- sort(unique(feats[, f]))
        if (length(vals) < 2) next
        mids <- (vals[-length(vals)] + vals[-1]) / 2
        for (thr in mids) {
          lf <- feats[, f] <= thr
          if (sum(lf) < 1 || sum(!lf) < 1) next
          lv <- mean(r[lf]); rv <- mean(r[!lf])
          sse <- sum((r[lf] - lv)^2) + sum((r[!lf] - rv)^2)
          gain <- sum(r^2) - sse
          if (is.null(best) || gain > best$gain)
            best <- list(gain = gain, feature = f, threshold = thr,
                          left_val = lv, right_val = rv)
        }
      }
      stumps[[b_]] <- best
      if (!is.null(best)) {
        pred <- ifelse(feats[, best$feature] <= best$threshold,
                        best$left_val, best$right_val)
        F_pred <- F_pred + learning_rate * pred
      }
      train_loss[b_] <- mean((y - F_pred)^2)
    }
    y_hat <- F_pred
  }
  resid <- y - y_hat
  list(estimate = mean(y_hat), y_hat = y_hat, train_loss = train_loss,
       se = sqrt(mean(resid^2)), n = n, method = method_used)
}

# CANONICAL TEST
# set.seed(14); M <- matrix(rnorm(160), 40, 4); y <- sign(M[,1])+0.3*rnorm(40)
# gradient_boosting_genomic(rep(0,40), y, M, n_estimators=20, seed=14)
