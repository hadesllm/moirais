# SPDX-License-Identifier: GPL-2.0-only

#' Ridge / LASSO / ElasticNet regularization path (R parity)
#'
#' Wraps \code{glmnet::glmnet}.  Returns the coefficient path across
#' the supplied \code{alphas} (lambda grid in glmnet terminology).
#'
#' @param x Numeric matrix of predictors.
#' @param y Numeric response.
#' @param penalty One of "ridge", "lasso", "elasticnet".
#' @param alphas Lambda grid.  Defaults to a logspace.
#' @param l1_ratio glmnet alpha; only used when penalty = "elasticnet".
#' @return Named list: estimate, coef_path, alphas, penalty, l1_ratio,
#'   n, method.
#' @export
regularization_path <- function(x, y, penalty = c("ridge", "lasso", "elasticnet"),
                                alphas = NULL, l1_ratio = 0.5) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Function 'regularization_path' requires package 'glmnet'. Install with install.packages('glmnet').")
  }
  penalty <- match.arg(penalty)
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); y <- as.numeric(y)
  n <- nrow(x); p <- ncol(x)
  if (is.null(alphas)) alphas <- 10^seq(-3, 2, length.out = 50)
  gn_alpha <- switch(penalty,
                     ridge = 0,
                     lasso = 1,
                     elasticnet = l1_ratio)
  fit <- glmnet::glmnet(x, y, alpha = gn_alpha, lambda = sort(alphas, decreasing = TRUE),
                        standardize = FALSE, intercept = TRUE)
  # glmnet returns columns in decreasing-lambda order; re-order to match alphas
  ord <- order(fit$lambda)
  lam <- fit$lambda[ord]
  beta <- as.matrix(fit$beta)[, ord, drop = FALSE]
  a0   <- fit$a0[ord]
  coef_path <- rbind(a0, beta)
  coef_path <- t(coef_path)
  colnames(coef_path) <- c("(intercept)", colnames(x) %||% paste0("x", seq_len(p) - 1L))
  list(
    estimate   = as.numeric(coef_path[nrow(coef_path), ]),
    coef_path  = coef_path,
    alphas     = lam,
    penalty    = penalty,
    l1_ratio   = if (penalty == "elasticnet") l1_ratio else NA_real_,
    n          = n,
    method     = sprintf("Regularization path (%s)", penalty)
  )
}

`%||%` <- function(a, b) if (is.null(a)) b else a
