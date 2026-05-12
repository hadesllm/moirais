# SPDX-License-Identifier: GPL-2.0-only

#' Kernel SVM (RBF / poly / sigmoid) — R parity
#'
#' Wraps \code{e1071::svm}.
#'
#' @param x Numeric predictor matrix.
#' @param y Binary response.
#' @param kernel One of "rbf" (radial), "poly", "sigmoid", "linear".
#' @param C Cost parameter.
#' @param gamma Kernel coefficient ("scale" -> 1/(ncol(x)*var(x)), "auto" -> 1/p, or numeric).
#' @param degree Polynomial degree.
#' @param seed RNG seed.
#' @return Named list: estimate, train_accuracy, n_support, kernel, C,
#'   gamma, degree, n, method.
#' @importFrom stats predict
#' @export
svm_kernel_trick <- function(x, y, kernel = "rbf", C = 1.0,
                              gamma = "scale", degree = 3L, seed = 0L) {
  if (!requireNamespace("e1071", quietly = TRUE)) {
    stop("Function 'svm_kernel_trick' requires package 'e1071'. Install with install.packages('e1071').")
  }
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); y <- as.factor(y)
  p <- ncol(x)
  e1071_kernel <- switch(kernel,
                         rbf = "radial",
                         poly = "polynomial",
                         sigmoid = "sigmoid",
                         linear = "linear")
  if (identical(gamma, "scale")) {
    g <- 1 / (p * stats::var(as.numeric(x)))
  } else if (identical(gamma, "auto")) {
    g <- 1 / p
  } else {
    g <- as.numeric(gamma)
  }
  set.seed(seed)
  fit <- e1071::svm(x = x, y = y, kernel = e1071_kernel, cost = C,
                    gamma = g, degree = degree, scale = FALSE)
  preds <- predict(fit, x)
  acc <- mean(preds == y)
  list(
    estimate       = as.numeric(acc),
    train_accuracy = as.numeric(acc),
    n_support      = as.integer(fit$nSV),
    kernel         = kernel,
    C              = as.numeric(C),
    gamma          = as.character(gamma),
    degree         = as.integer(degree),
    n              = nrow(x),
    method         = sprintf("Kernel SVM (%s)", kernel)
  )
}
