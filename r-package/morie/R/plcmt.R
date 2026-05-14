# SPDX-License-Identifier: MIT OR Apache-2.0

#' Rank placements of Y among X order statistics (Gibbons Ch 2.11.3)
#'
#' For each Y_j: placement P_j = number of X_i less than Y_j.  Their sum is the
#' Mann-Whitney U statistic for Y vs X.
#'
#' @param x,y Numeric vectors.
#' @return Named list: placements, ranks_y, U_y, E_U, Var_U, m, n.
#' @export
rank_placements <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  m <- length(x); n <- length(y)
  if (m < 1 || n < 1) {
    return(list(placements = integer(0), ranks_y = numeric(0),
                U_y = NA_real_, E_U = NA_real_, Var_U = NA_real_,
                m = m, n = n, method = "Rank placements"))
  }
  xs <- sort(x)
  placements <- as.integer(findInterval(y, xs, left.open = FALSE))
  # equivalent of np.searchsorted side='left': number of xs < yj
  # findInterval(left.open=FALSE) returns number of xs <= yj; subtract ties at exactly yj
  # but for continuous data placements = findInterval(left.open=FALSE)
  ranks_all <- rank(c(x, y))
  ranks_y <- ranks_all[(m + 1):(m + n)]
  U_y <- sum(placements)
  E_U <- m * n / 2
  Var_U <- m * n * (m + n + 1) / 12
  list(
    placements = placements,
    ranks_y = ranks_y,
    U_y = U_y,
    E_U = E_U,
    Var_U = Var_U,
    m = m,
    n = n,
    method = "Rank placements"
  )
}
