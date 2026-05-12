# SPDX-License-Identifier: GPL-2.0-only

#' Optimal Classification cutting point (Poole 2000; Armstrong Ch 3)
#'
#' Brute-force 1-D cutting-point search that maximises classification
#' of a binary vote vector. Reports PRE (proportional reduction in
#' error) against the modal-class baseline.
#'
#' @param x Numeric vector of ideal points.
#' @param votes Integer 0/1 vector of observed votes (optional).
#' @return Named list with `cut`, `correct_class`, `polarity`, `pre`,
#'   `n`, `method`.
#' @export
optcl <- function(x, votes = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n == 0L) {
    return(list(cut = NA_real_, correct_class = 0L, polarity = 1L,
                pre = NA_real_, n = 0L, method = "optimal_classification"))
  }
  if (is.null(votes)) {
    return(list(cut = stats::median(x),
                correct_class = as.integer(n %/% 2L + n %% 2L),
                polarity = 1L, pre = NA_real_, n = n,
                method = "optimal_classification"))
  }
  y <- as.integer(votes)
  xs <- sort(x)
  cand <- c(xs[1] - 1, (xs[-length(xs)] + xs[-1]) / 2, xs[length(xs)] + 1)
  best_cc <- -1L; best_cut <- stats::median(x); best_pol <- 1L
  for (c in cand) for (pol in c(1L, -1L)) {
    pred <- if (pol == 1L) as.integer(x > c) else as.integer(x <= c)
    cc   <- sum(pred == y)
    if (cc > best_cc) { best_cc <- cc; best_cut <- c; best_pol <- pol }
  }
  p <- mean(y); base_correct <- max(p, 1 - p) * n
  pre <- if (n > base_correct) (best_cc - base_correct) / (n - base_correct) else 0
  list(cut = best_cut, correct_class = best_cc, polarity = best_pol,
       pre = pre, n = n, method = "optimal_classification")
}

#' @keywords internal
#' @rdname optcl
#' @export
optimal_classification <- optcl
