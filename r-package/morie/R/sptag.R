# SPDX-License-Identifier: MIT OR Apache-2.0

#' Pairwise vote-agreement matrix (Armstrong Ch 8)
#'
#' A_ij = proportion of roll calls on which legislators i, j voted the
#' same way (excluding mutually absent items).
#'
#' @param x Vote matrix (n by m); NA = absent.
#' @return Named list with `agreement` (n by n), `mean_agreement`,
#'   `n`, `m`, `method`.
#' @export
sptag <- function(x) {
  M <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1L)
  n <- nrow(M); m <- ncol(M)
  if (n < 2L)
    return(list(agreement = diag(n), mean_agreement = NA_real_,
                n = n, m = m, method = "spatial_agreement"))
  A <- diag(n); valid <- !is.na(M)
  for (i in seq_len(n - 1L)) for (j in (i + 1L):n) {
    both <- valid[i, ] & valid[j, ]
    denom <- sum(both)
    if (denom == 0L) A[i, j] <- A[j, i] <- NA_real_
    else {
      same <- sum(M[i, both] == M[j, both])
      A[i, j] <- A[j, i] <- same / denom
    }
  }
  iu <- upper.tri(A)
  list(agreement = A, mean_agreement = mean(A[iu], na.rm = TRUE),
       n = n, m = m, method = "spatial_agreement")
}

#' @keywords internal
#' @rdname sptag
#' @export
spatial_agreement <- sptag
