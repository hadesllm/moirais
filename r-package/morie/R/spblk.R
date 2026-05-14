# SPDX-License-Identifier: MIT OR Apache-2.0
#' Ordinary block kriging.
#'
#' Solve the OK system with point-to-block averaged covariances:
#'   \deqn{C_{\text{bar}}(s_i, B) = (1/|B|) \int_B C(s_i, u) du.}
#'
#' @param x Numeric vector.
#' @param coords Coord matrix (n by d).
#' @param blocks list; each element either (q by d) quadrature points or a
#'   (2 by d) matrix giving (lo, hi) for a box. Boxes are quadratured with
#'   `n_quad` regular nodes (d = 1) or sqrt(n_quad)^2 nodes (d = 2).
#' @param n_quad Default 25.
#' @param nugget,sill,range_ Exponential-covariance parameters.
#' @return Named list: estimate, se, n, method.
#' @references Schabenberger & Gotway (2005), Ch 4.
#' @export
spblk <- function(x, coords, blocks, n_quad = 25,
                  nugget = 0, sill = 1, range_ = 1) {
  x <- as.numeric(x); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  d <- ncol(coords)
  c0 <- nugget; c1 <- sill - nugget; a <- range_
  cov_fn <- function(h) c1 * exp(-h / a) + ifelse(h == 0, c0, 0)
  D <- as.matrix(stats::dist(coords))
  C <- cov_fn(D)
  A <- matrix(0, n + 1, n + 1)
  A[1:n, 1:n] <- C; A[1:n, n + 1] <- 1; A[n + 1, 1:n] <- 1
  block_to_pts <- function(b) {
    b <- if (is.matrix(b)) b else matrix(as.numeric(unlist(b)), ncol = d)
    if (nrow(b) == 2 && ncol(b) == d) {
      lo <- b[1, ]; hi <- b[2, ]
      if (d == 1) return(matrix(seq(lo[1], hi[1], length.out = n_quad), ncol = 1))
      if (d == 2) {
        k <- round(sqrt(n_quad))
        g1 <- seq(lo[1], hi[1], length.out = k)
        g2 <- seq(lo[2], hi[2], length.out = k)
        return(as.matrix(expand.grid(g1, g2)))
      }
      stop("box-form blocks supported only for d <= 2")
    }
    b
  }
  m <- length(blocks)
  ests <- numeric(m); ses <- numeric(m)
  pairwise_cov <- function(P, Q) {
    DPQ <- sqrt(outer(rowSums(P ^ 2), rowSums(Q ^ 2), "+") - 2 * P %*% t(Q))
    DPQ[DPQ < 0] <- 0
    cov_fn(DPQ)
  }
  for (b_idx in seq_len(m)) {
    pts <- block_to_pts(blocks[[b_idx]])
    Cbar_iB <- rowMeans(pairwise_cov(coords, pts))
    Cbar_BB <- mean(pairwise_cov(pts, pts))
    rhs <- c(Cbar_iB, 1)
    sol <- tryCatch(solve(A, rhs),
                    error = function(e) qr.solve(A, rhs))
    lam <- sol[1:n]; mu <- sol[n + 1]
    ests[b_idx] <- sum(lam * x)
    ses[b_idx] <- sqrt(max(Cbar_BB - sum(lam * Cbar_iB) - mu, 0))
  }
  list(estimate = ests, se = ses, n = n,
       method = "Ordinary block kriging (exp. cov, MC quadrature)")
}

#' @rdname spblk
#' @keywords internal
#' @export
spatial_block_kriging <- spblk
