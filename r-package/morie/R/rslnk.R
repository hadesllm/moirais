# SPDX-License-Identifier: MIT OR Apache-2.0

#' Residual / skip connection
#'
#' R parity for \code{morie.fn.rslnk.residual_connection}.
#'
#' \deqn{y = \mathcal{F}(x) + x}
#'
#' @param x Numeric array.
#' @param f Function applied as the residual branch; defaults to identity.
#' @return Named list \code{(y, estimate, Fx, method)}.
#' @references He, Zhang, Ren & Sun (2016), CVPR.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
rslnk_residual_connection <- function(x, f = NULL) {
  x <- as.array(x)
  Fx <- if (is.null(f)) x else as.array(f(x))
  if (!identical(dim(Fx), dim(x)) && length(Fx) != length(x))
    stop("Residual branch shape does not match identity shape.")
  y <- Fx + x
  list(y = y, estimate = y, Fx = Fx,
       method = "Residual identity shortcut")
}

#' @rdname rslnk_residual_connection
#' @keywords internal
#' @export
residual_connection <- rslnk_residual_connection
