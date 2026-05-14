# SPDX-License-Identifier: MIT OR Apache-2.0

#' Genomic-prediction accuracy metrics
#'
#' Reports Pearson r, Spearman rho, MSE/MSPE, RMSE, R^2, calibration
#' slope and intercept.
#'
#' @param y_true Numeric observed.
#' @param y_pred Numeric predicted.
#' @return list(estimate (Pearson r), pearson_r, spearman_rho, mse, mspe,
#'   rmse, r2, slope, intercept, n, method).
#' @references Montesinos Lopez Ch 2.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
prediction_accuracy <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true); y_pred <- as.numeric(y_pred)
  n <- length(y_true)
  if (n != length(y_pred)) stop("y_true and y_pred must be same length")
  if (n < 2) return(list(estimate = NA_real_, n = n,
                          method = "Genomic prediction accuracy (n<2)"))
  mse <- mean((y_true - y_pred)^2)
  rmse <- sqrt(mse)
  var_y <- stats::var(y_true)
  r2 <- if (var_y > 0) 1 - mse / var_y else NA_real_
  r <- if (stats::sd(y_true) > 0 && stats::sd(y_pred) > 0)
    stats::cor(y_true, y_pred) else NA_real_
  rho <- if (stats::sd(y_true) > 0 && stats::sd(y_pred) > 0)
    stats::cor(y_true, y_pred, method = "spearman") else NA_real_
  slope <- if (stats::var(y_pred) > 0)
    stats::cov(y_true, y_pred) / stats::var(y_pred) else NA_real_
  intercept <- if (!is.na(slope)) mean(y_true) - slope * mean(y_pred)
               else NA_real_
  list(estimate = r, pearson_r = r, spearman_rho = rho,
       mse = mse, mspe = mse, rmse = rmse, r2 = r2,
       slope = slope, intercept = intercept,
       n = n, method = "Pearson r + Spearman rho + MSE/MSPE + calibration")
}

# CANONICAL TEST
# y <- c(1,2,3,4,5); y_hat <- c(1.1,1.9,3.2,3.8,5.1)
# prediction_accuracy(y, y_hat)$pearson_r
