# SPDX-License-Identifier: MIT OR Apache-2.0

#' ROC curve and AUC (R parity)
#'
#' Wraps \code{pROC::roc}.
#'
#' @param y_true Binary labels.
#' @param y_score Predicted scores for the positive class.
#' @return Named list: estimate, auc, fpr, tpr, thresholds, n,
#'   n_positive, n_negative, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
roc_auc_score <- function(y_true, y_score) {
  if (!requireNamespace("pROC", quietly = TRUE)) {
    stop("Function 'roc_auc_score' requires package 'pROC'. Install with install.packages('pROC').")
  }
  yt <- as.numeric(y_true); ys <- as.numeric(y_score)
  classes <- sort(unique(yt))
  if (length(classes) != 2) stop("roc_auc_score requires binary y_true")
  pos <- classes[2]
  yt_b <- as.integer(yt == pos)
  rc <- pROC::roc(response = yt_b, predictor = ys,
                  levels = c(0, 1), direction = "<",
                  quiet = TRUE)
  auc <- as.numeric(pROC::auc(rc))
  # pROC reports specificity (= 1 - FPR) and sensitivity (= TPR) per threshold
  fpr <- 1 - rc$specificities
  tpr <- rc$sensitivities
  # Sort by FPR ascending to match sklearn output order
  ord <- order(fpr, tpr)
  list(
    estimate    = as.numeric(auc),
    auc         = as.numeric(auc),
    fpr         = fpr[ord],
    tpr         = tpr[ord],
    thresholds  = rc$thresholds[ord],
    n           = length(yt),
    n_positive  = sum(yt_b == 1L),
    n_negative  = sum(yt_b == 0L),
    method      = "ROC AUC (Mann-Whitney U probability)"
  )
}
