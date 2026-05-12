# SPDX-License-Identifier: GPL-2.0-only

#' Counting process for survival data
#'
#' N(infty) = the count of events (sum of indicator delta_i = 1).
#'
#' @param t Numeric vector of observed times.
#' @param event Integer/logical vector (1 = event, 0 = censored).
#' @return Named list with estimate (total events), n, method.
#' @references Kosorok (2008), Ch 8.
#' @export
ksr17_kosorok_counting_process <- function(t, event) {
  t <- as.numeric(t); event <- as.integer(event)
  list(
    estimate = as.integer(sum(event)),
    n        = length(t),
    method   = "Counting process N(infty) = sum 1{T_i finite, delta_i=1}"
  )
}

# CANONICAL TEST
# ksr17_kosorok_counting_process(1:10, c(1,1,0,1,1,0,1,1,1,0))

#' @rdname ksr17_kosorok_counting_process
#' @keywords internal
#' @export
kosorok_counting_process <- ksr17_kosorok_counting_process
