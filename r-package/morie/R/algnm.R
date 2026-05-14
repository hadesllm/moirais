# SPDX-License-Identifier: MIT OR Apache-2.0

#' Rice party cohesion index (Rice 1928; Armstrong Ch 8)
#'
#' Rice_p = |%yea_p - %nay_p| over a series of roll calls; with a
#' `party` indicator, averages per-roll-call Rice over each party.
#'
#' @param x Either a 0/1 vote vector for one party, or a numeric
#'   matrix (n_legislators by n_roll_calls).
#' @param party Optional party label vector (length = n rows of x).
#' @return Named list with `estimate`, `per_party` (if 2-D), `n`,
#'   `method`.
#' @export
algnm <- function(x, party = NULL) {
  X <- if (is.matrix(x)) x else as.numeric(x)
  if (!is.matrix(X)) {
    valid <- X[!is.na(X)]
    if (length(valid) == 0L)
      return(list(estimate = NA_real_, n = 0L, method = "rice_cohesion"))
    p_yea <- mean(valid == 1); p_nay <- mean(valid == 0)
    return(list(estimate = abs(p_yea - p_nay), pct_yea = p_yea,
                pct_nay = p_nay, n = length(valid),
                method = "rice_cohesion"))
  }
  n <- nrow(X); m <- ncol(X)
  per <- list()
  rice_for <- function(sub) {
    vapply(seq_len(m), function(j) {
      col <- sub[, j]; col <- col[!is.na(col)]
      if (length(col) == 0L) NA_real_
      else abs(mean(col == 1) - mean(col == 0))
    }, numeric(1))
  }
  if (is.null(party)) {
    rice <- rice_for(X)
    per[["all"]] <- mean(rice, na.rm = TRUE)
    overall <- per[["all"]]
  } else {
    if (length(party) != n) stop("party length must match n rows")
    for (lbl in unique(party)) {
      per[[as.character(lbl)]] <- mean(rice_for(X[party == lbl, , drop = FALSE]),
                                      na.rm = TRUE)
    }
    overall <- mean(unlist(per), na.rm = TRUE)
  }
  list(estimate = overall, per_party = per, n = n, m = m,
       method = "rice_cohesion")
}

#' @keywords internal
#' @rdname algnm
#' @export
party_alignment <- algnm
