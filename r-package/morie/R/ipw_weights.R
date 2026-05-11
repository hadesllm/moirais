#' Calculate inverse probability of treatment weights (IPTW)
#'
#' Mirrors the Python `morie.calculate_ipw_weights()`. Pure-R, no extra
#' dependencies.
#'
#' Standard IPW: \eqn{w_i = T_i / e_i + (1 - T_i)/(1 - e_i)}, with the
#' propensity score \eqn{e_i} clipped at \[0.01, 0.99\] for stability.
#' Stabilised IPW replaces \eqn{T} and \eqn{1 - T} with the marginal
#' treatment probability \eqn{P(T = 1)} and \eqn{P(T = 0)} respectively.
#'
#' @param data A `data.frame` containing treatment assignment and propensity
#'   scores.
#' @param treatment Column name (string) of the binary treatment.
#' @param ps_col Column name (string) of the propensity scores.
#' @param stabilized If `TRUE`, return stabilised IPW weights. Default `FALSE`.
#' @param trim_quantiles Optional length-2 numeric vector \eqn{(q_l, q_u)}
#'   in \eqn{[0, 1]}; if supplied, weights are clipped to the
#'   \eqn{q_l}-th and \eqn{q_u}-th quantiles of the unclipped weight
#'   distribution (Crump et al. 2009 trimming). Default `NULL`.
#'
#' @return Numeric vector of IPTW weights, length `nrow(data)`.
#' @export
#' @examples
#' set.seed(1)
#' df <- data.frame(
#'   t = rbinom(100, 1, 0.4),
#'   ps = pmin(pmax(runif(100, 0.05, 0.95), 0.05), 0.95)
#' )
#' w <- calculate_ipw_weights(df, treatment = "t", ps_col = "ps")
#' summary(w)
calculate_ipw_weights <- function(data, treatment, ps_col,
                                  stabilized = FALSE,
                                  trim_quantiles = NULL) {
  ps <- pmin(pmax(data[[ps_col]], 0.01), 0.99)
  t  <- data[[treatment]]
  if (stabilized) {
    p_treated <- mean(t)
    weights <- ifelse(t == 1, p_treated / ps, (1 - p_treated) / (1 - ps))
  } else {
    weights <- (t / ps) + ((1 - t) / (1 - ps))
  }
  if (!is.null(trim_quantiles)) {
    if (length(trim_quantiles) != 2L)
      stop("trim_quantiles must be length 2 (lower, upper).")
    qs <- stats::quantile(weights, probs = trim_quantiles, na.rm = TRUE)
    weights <- pmin(pmax(weights, qs[[1L]]), qs[[2L]])
  }
  weights
}
