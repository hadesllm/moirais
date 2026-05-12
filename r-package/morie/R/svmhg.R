# SPDX-License-Identifier: GPL-2.0-only

#' Linear SVM (primal hinge loss) — R parity
#'
#' Wraps \code{e1071::svm} with a linear kernel.
#'
#' @param x Numeric predictor matrix.
#' @param y Binary response.
#' @param C Soft-margin inverse regularisation.
#' @param seed RNG seed.
#' @return Named list: estimate, intercept, weights, train_accuracy, C,
#'   classes, n, method.
#' @importFrom stats predict
#' @export
svm_hinge_primal <- function(x, y, C = 1.0, seed = 0L) {
  if (!requireNamespace("e1071", quietly = TRUE)) {
    stop("Function 'svm_hinge_primal' requires package 'e1071'. Install with install.packages('e1071').")
  }
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); y <- as.factor(y)
  classes <- levels(y)
  if (length(classes) != 2) stop("svm_hinge_primal requires binary y")
  set.seed(seed)
  fit <- e1071::svm(x = x, y = y, kernel = "linear", cost = C, scale = FALSE)
  # Reconstruct w = sum_i alpha_i y_i x_i (libsvm sign convention: coefs are alpha_i*y_i)
  w <- as.numeric(crossprod(fit$coefs, fit$SV))
  b <- -as.numeric(fit$rho)
  preds <- predict(fit, x)
  acc <- mean(preds == y)
  list(
    estimate       = c(b, w),
    intercept      = b,
    weights        = w,
    train_accuracy = as.numeric(acc),
    C              = as.numeric(C),
    classes        = classes,
    n              = nrow(x),
    method         = "Linear SVM (primal hinge loss)"
  )
}
