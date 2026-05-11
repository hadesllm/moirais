#' Calculate estimated Blood Alcohol Concentration (eBAC)
#'
#' Compute the continuous estimated Blood Alcohol Concentration using the
#' standard Widmark formula. Mirrors the Python `morie.calculate_ebac()`.
#'
#' The Widmark formula is:
#' \deqn{eBAC = (drinks \times 5.14) / (weight\_lbs \times r) - 0.015 \times hours}
#' where \eqn{r} is the gender constant (0.73 for men, 0.66 for women).
#' Returned values are clipped at zero.
#'
#' @param drinks Number of standard drinks consumed (1 drink = 14 g alcohol).
#' @param weight_lbs Body weight in pounds.
#' @param hours Hours elapsed since drinking began.
#' @param gender_constant Widmark gender multiplier (0.73 men, 0.66 women).
#'
#' @return Non-negative numeric scalar: estimated BAC.
#' @export
#' @examples
#' calculate_ebac(drinks = 4, weight_lbs = 180, hours = 2, gender_constant = 0.73)
calculate_ebac <- function(drinks, weight_lbs, hours, gender_constant) {
  if (weight_lbs <= 0) return(0.0)
  ebac <- (drinks * 5.14) / (weight_lbs * gender_constant) - (0.015 * hours)
  max(0.0, ebac)
}

#' Test whether an eBAC exceeds a legal driving limit
#'
#' @param ebac Numeric eBAC value (e.g. from [calculate_ebac()]).
#' @param limit Legal threshold (default 0.08, the per-se limit in most
#'   Canadian and US jurisdictions).
#'
#' @return Integer 1 if `ebac >= limit`, 0 otherwise. (Integer, not
#'   logical, to match the Python sibling and ease binary-outcome modelling.)
#' @export
#' @examples
#' is_over_legal_limit(0.09)
#' is_over_legal_limit(0.05, limit = 0.05)
is_over_legal_limit <- function(ebac, limit = 0.08) {
  if (ebac >= limit) 1L else 0L
}
