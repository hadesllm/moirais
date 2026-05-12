# SPDX-License-Identifier: GPL-2.0-only

#' Banzhaf and Shapley-Shubik voting-power indices (Armstrong Ch 10)
#'
#' Exact enumeration for n <= 10; Monte Carlo for larger games.
#' Quota defaults to strict simple majority.
#'
#' @param x Numeric weight vector w.
#' @param quota Winning threshold q (default sum(w)/2).
#' @return Named list with `banzhaf`, `shapley_shubik`, `quota`,
#'   `weights`, `method`.
#' @export
vtpwr <- function(x, quota = NULL) {
  w <- as.numeric(x); n <- length(w)
  if (n == 0L)
    return(list(banzhaf = numeric(0), shapley_shubik = numeric(0),
                quota = NA_real_, weights = w,
                method = "voting_power_index"))
  if (is.null(quota)) quota <- sum(w) / 2 + 1e-9
  if (n > 10L) {
    set.seed(0L); N_mc <- 20000L
    swings <- rep(0, n); ss <- rep(0, n)
    for (k in seq_len(N_mc)) {
      mask <- as.logical(sample.int(2L, n, replace = TRUE) - 1L)
      tot_in <- sum(w[mask])
      for (i in seq_len(n)) {
        if (mask[i]) {
          swings[i] <- swings[i] + (tot_in >= quota &&
                                    (tot_in - w[i]) < quota)
        } else {
          swings[i] <- swings[i] + ((tot_in + w[i]) >= quota &&
                                    tot_in < quota)
        }
      }
      ord <- sample.int(n); cum <- 0
      for (idx in ord) { prev <- cum; cum <- cum + w[idx]
        if (prev < quota && quota <= cum) { ss[idx] <- ss[idx] + 1L; break }
      }
    }
    banzhaf <- swings / max(sum(swings), 1)
    shapley <- ss / N_mc
    return(list(banzhaf = banzhaf, shapley_shubik = shapley,
                quota = quota, weights = w,
                method = "voting_power_index_mc"))
  }
  # Exact Banzhaf via subset enumeration
  swings <- rep(0, n)
  for (mask_int in 0:(2^n - 1L)) {
    mask <- as.logical(intToBits(mask_int)[1:n])
    tot_in <- sum(w[mask])
    for (i in seq_len(n)) {
      if (mask[i]) {
        swings[i] <- swings[i] + (tot_in >= quota &&
                                  (tot_in - w[i]) < quota)
      } else {
        swings[i] <- swings[i] + ((tot_in + w[i]) >= quota &&
                                  tot_in < quota)
      }
    }
  }
  banzhaf <- swings / max(sum(swings), 1)
  # Exact Shapley-Shubik by enumerating all n! orderings
  shapley <- rep(0, n)
  perms <- function(v) {
    if (length(v) == 1L) return(matrix(v, 1L, 1L))
    do.call(rbind, lapply(seq_along(v), function(i)
      cbind(v[i], perms(v[-i]))))
  }
  P <- perms(seq_len(n))
  for (r in seq_len(nrow(P))) {
    ord <- P[r, ]; cum <- 0
    for (idx in ord) { prev <- cum; cum <- cum + w[idx]
      if (prev < quota && quota <= cum) { shapley[idx] <- shapley[idx] + 1L; break }
    }
  }
  shapley <- shapley / factorial(n)
  list(banzhaf = banzhaf, shapley_shubik = shapley,
       quota = quota, weights = w,
       method = "voting_power_index_exact")
}

#' @keywords internal
#' @rdname vtpwr
#' @export
voting_power_index <- vtpwr
