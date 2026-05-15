# SPDX-License-Identifier: AGPL-3.0-or-later
#' Stratified mean estimator (Cochran 1977, Sampling Techniques, Ch. 5)
#'
#' Within-stratum means averaged with population (or proportional)
#' weights W_h, with stratified-sampling variance
#'    var(y_bar_st) = sum_h W_h^2 s_h^2 / n_h.
#' Python parity: \code{morie.fn.strat.stratified_mean}.
#'
#' @param data data.frame containing outcome and stratum columns.
#' @param y character; outcome column.
#' @param strata character; stratum column.
#' @param pop_sizes optional named numeric vector mapping stratum -> N_h.
#'   If NULL, proportional weights W_h = n_h/sum(n_h) are used.
#' @return list: estimate, se, ci_lower, ci_upper, weights, strata_means,
#'   n_strata, method.
#' @keywords internal
strat <- function(data, y = "y", strata = "stratum", pop_sizes = NULL) {
  yv <- as.numeric(data[[y]])
  sv <- data[[strata]]
  strata_names <- unique(sv)
  n_h <- vapply(strata_names, function(s) sum(sv == s), integer(1))
  yb_h <- vapply(strata_names, function(s) mean(yv[sv == s]), numeric(1))
  s2_h <- vapply(strata_names, function(s) stats::var(yv[sv == s]), numeric(1))
  if (is.null(pop_sizes)) {
    W_h <- n_h / sum(n_h)
  } else {
    N <- sum(pop_sizes)
    W_h <- vapply(strata_names, function(s) pop_sizes[[as.character(s)]] / N,
                  numeric(1))
  }
  est <- sum(W_h * yb_h)
  var_st <- sum(W_h^2 * s2_h / n_h)
  se <- sqrt(var_st)
  z <- stats::qnorm(0.975)
  names(W_h) <- as.character(strata_names)
  names(yb_h) <- as.character(strata_names)
  list(estimate = as.numeric(est), se = as.numeric(se),
       ci_lower = est - z * se, ci_upper = est + z * se,
       weights = as.list(W_h), strata_means = as.list(yb_h),
       n_strata = length(strata_names),
       method = "Stratified mean (Cochran 1977)")
}

# CANONICAL TEST
# df <- data.frame(y = c(1,2,3,10,11,12), stratum = c("a","a","a","b","b","b"))
# r <- strat(df, "y", "stratum")
# # equal n per stratum, equal weights -> estimate = (mean_a + mean_b)/2 = (2+11)/2 = 6.5
# stopifnot(abs(r$estimate - 6.5) < 1e-9)

#' @rdname strat
#' @keywords internal
#' @export
stratified_sampling <- strat
