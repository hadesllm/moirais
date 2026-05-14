# SPDX-License-Identifier: MIT OR Apache-2.0

#' Confusion matrix with precision / recall / F1 (R parity)
#'
#' Manually constructs the confusion matrix to avoid the caret dependency
#' for what is fundamentally a tabulation.
#'
#' @param y_true Observed labels.
#' @param y_pred Predicted labels.
#' @param labels Optional ordering vector.
#' @return Named list: estimate (accuracy), accuracy, confusion_matrix,
#'   labels, precision, recall, f1, macro_precision, macro_recall,
#'   macro_f1, weighted_f1, n, method.
#' @export
confusion_matrix_metrics <- function(y_true, y_pred, labels = NULL) {
  yt <- as.character(y_true); yp <- as.character(y_pred)
  if (is.null(labels)) labels <- sort(unique(c(yt, yp)))
  labels <- as.character(labels)
  K <- length(labels)
  cm <- matrix(0L, nrow = K, ncol = K, dimnames = list(labels, labels))
  for (i in seq_along(yt)) {
    a <- match(yt[i], labels); b <- match(yp[i], labels)
    cm[a, b] <- cm[a, b] + 1L
  }
  diag_ <- diag(cm)
  col_sums <- colSums(cm)
  row_sums <- rowSums(cm)
  precision <- ifelse(col_sums > 0, diag_ / col_sums, 0)
  recall    <- ifelse(row_sums > 0, diag_ / row_sums, 0)
  f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
  acc <- sum(diag_) / sum(cm)
  support <- row_sums
  total <- sum(support)
  macro_p <- mean(precision)
  macro_r <- mean(recall)
  macro_f1 <- mean(f1)
  weighted_f1 <- if (total > 0) sum(f1 * support) / total else 0
  list(
    estimate         = as.numeric(acc),
    accuracy         = as.numeric(acc),
    confusion_matrix = cm,
    labels           = labels,
    precision        = as.numeric(precision),
    recall           = as.numeric(recall),
    f1               = as.numeric(f1),
    macro_precision  = as.numeric(macro_p),
    macro_recall     = as.numeric(macro_r),
    macro_f1         = as.numeric(macro_f1),
    weighted_f1      = as.numeric(weighted_f1),
    n                = length(yt),
    method           = "Confusion matrix + precision/recall/F1"
  )
}
