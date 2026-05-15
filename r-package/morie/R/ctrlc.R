# SPDX-License-Identifier: AGPL-3.0-or-later

#' Nonparametric many-to-one comparisons to a control
#' (Gibbons Ch 10.7)
#'
#' Mann-Whitney vs. control for each treatment group; Bonferroni-
#' adjusted p-values by default.
#'
#' @param groups List of numeric vectors; first (or `control_index`-th)
#'   element is the control.
#' @param control_index Integer position of the control group.  Default 1.
#' @param adjust One of `"bonferroni"`, `"none"`.
#' @return Named list: statistic, p_value, p_adjusted, n, k, control_n.
#' @importFrom stats wilcox.test p.adjust
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
control_comparison <- function(groups, control_index = 1L,
                                adjust = c("bonferroni", "none")) {
  adjust <- match.arg(adjust)
  if (!is.list(groups) || length(groups) < 2L)
    return(list(statistic = numeric(0), p_value = numeric(0),
                p_adjusted = numeric(0), n = integer(0),
                k = 0L, control_n = 0L,
                method = "Many-to-one control comparison"))
  arrs <- lapply(groups, as.numeric)
  ctrl <- arrs[[control_index]]
  trts <- arrs[-control_index]
  k <- length(trts)
  Us <- numeric(k); ps <- numeric(k)
  for (i in seq_along(trts)) {
    if (min(length(ctrl), length(trts[[i]])) < 2) {
      Us[i] <- NA_real_; ps[i] <- NA_real_; next
    }
    r <- suppressWarnings(stats::wilcox.test(ctrl, trts[[i]],
                                              exact = FALSE,
                                              correct = FALSE))
    Us[i] <- as.numeric(r$statistic)
    ps[i] <- as.numeric(r$p.value)
  }
  p_adj <- switch(adjust,
                  bonferroni = pmin(ps * k, 1),
                  none = ps)
  list(
    statistic = Us,
    p_value = ps,
    p_adjusted = p_adj,
    n = sapply(trts, length),
    k = k,
    control_n = length(ctrl),
    adjust = adjust,
    method = "Nonparametric many-to-one (Mann-Whitney) vs. control"
  )
}
