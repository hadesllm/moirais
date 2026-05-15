# SPDX-License-Identifier: AGPL-3.0-or-later
#' General two-sample permutation test (Good 2005)
#'
#' Tests H0: F_x = F_y by Monte-Carlo permutation of pooled samples.
#' p-value uses (1 + count of permutation statistics at least T_obs)
#' divided by (B+1), per Phipson and Smyth 2010.
#'
#' @param x,y numeric vectors.
#' @param statistic function(x, y) returning a scalar; default mean diff.
#' @param B integer; number of permutations (default 5000).
#' @param alternative "two-sided", "less", or "greater".
#' @param seed integer.
#' @return Named list: statistic, p_value, n_x, n_y, B, alternative, method.
#' @keywords internal
permt <- function(x, y, statistic = NULL, B = 5000L,
                  alternative = c("two-sided", "less", "greater"),
                  seed = 42L) {
  alternative <- match.arg(alternative)
  x <- as.numeric(x); y <- as.numeric(y)
  n_x <- length(x); n_y <- length(y)
  if (n_x < 1L || n_y < 1L) {
    return(list(statistic = NA_real_, p_value = NA_real_,
                n_x = n_x, n_y = n_y, method = "permt (empty)"))
  }
  if (is.null(statistic))
    statistic <- function(a, b) mean(a) - mean(b)
  T_obs <- statistic(x, y)
  pool <- c(x, y); m <- length(pool)
  set.seed(seed)
  T_perm <- numeric(B)
  for (b in seq_len(B)) {
    ord <- sample.int(m)
    T_perm[b] <- statistic(pool[ord[seq_len(n_x)]],
                           pool[ord[(n_x + 1):m]])
  }
  p <- switch(alternative,
              greater   = (1 + sum(T_perm >= T_obs)) / (B + 1),
              less      = (1 + sum(T_perm <= T_obs)) / (B + 1),
              `two-sided` = (1 + sum(abs(T_perm) >= abs(T_obs))) / (B + 1))
  list(statistic = as.numeric(T_obs), p_value = as.numeric(p),
       n_x = as.integer(n_x), n_y = as.integer(n_y),
       B = as.integer(B), alternative = alternative,
       method = "Permutation test (Good 2005)")
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(30); y <- rnorm(30)
# r <- permt(x, y, B = 1000, seed = 0)
# stopifnot(r$p_value > 0, r$p_value < 1)

#' @rdname permt
#' @keywords internal
#' @export
permutation_test_general <- permt
