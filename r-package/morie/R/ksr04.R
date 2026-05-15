# SPDX-License-Identifier: AGPL-3.0-or-later

#' VC dimension for affine half-spaces in R^d
#'
#' VC dimension of the linear half-space classifier in R^d equals d+1.
#'
#' @param x Numeric matrix or vector; d is its number of columns.
#' @return Named list with estimate, n, method.
#' @references Kosorok (2008), Ch 2; Vapnik & Chervonenkis (1971).
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
ksr04_kosorok_vc_dimension <- function(x) {
  if (is.null(dim(x))) {
    d <- 1L; n <- length(x)
  } else {
    n <- nrow(x); d <- ncol(x)
  }
  list(
    estimate = as.integer(d + 1L),
    n        = as.integer(n),
    method   = "VC(affine half-spaces in R^d) = d+1"
  )
}

# CANONICAL TEST
# ksr04_kosorok_vc_dimension(matrix(0, 100, 3))

#' @rdname ksr04_kosorok_vc_dimension
#' @keywords internal
#' @export
kosorok_vc_dimension <- ksr04_kosorok_vc_dimension
