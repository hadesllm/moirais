# SPDX-License-Identifier: MIT OR Apache-2.0

#' Mood's median (control-median) test (Gibbons Ch 6.5)
#'
#' Two-sample median test: contingency-table chi-square on the
#' counts above/below the pooled-sample median.
#'
#' @param x Numeric vector (control).
#' @param y Numeric vector (treatment).
#' @return Named list: statistic, p_value, df, n, grand_median, table.
#' @importFrom stats median chisq.test
#' @export
control_median_test <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  m <- length(x); n <- length(y)
  if (m < 2 || n < 2) {
    return(list(statistic = NA_real_, p_value = NA_real_, df = 1L,
                n = m + n, grand_median = NA_real_,
                method = "Control-median (Mood's median) test"))
  }
  med <- stats::median(c(x, y))
  # Ties: count == as below (matches scipy 'below')
  tbl <- matrix(c(sum(x > med), sum(x <= med),
                  sum(y > med), sum(y <= med)),
                nrow = 2, byrow = TRUE,
                dimnames = list(c("x", "y"), c("above", "below_eq")))
  ct <- suppressWarnings(stats::chisq.test(tbl, correct = TRUE))
  list(
    statistic = as.numeric(ct$statistic),
    p_value = as.numeric(ct$p.value),
    df = as.integer(ct$parameter),
    n = m + n,
    m = m,
    n_y = n,
    grand_median = med,
    table = tbl,
    method = "Control-median (Mood's median) test"
  )
}
