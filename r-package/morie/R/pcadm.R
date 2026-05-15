# SPDX-License-Identifier: AGPL-3.0-or-later

#' PCA via SVD for dimension reduction (R parity)
#'
#' Wraps \code{stats::prcomp}.
#'
#' @param x Numeric matrix.
#' @param n_components Number of components (default min(n, p)).
#' @param seed Unused for the SVD path; kept for API parity.
#' @return Named list: estimate, components, explained_variance,
#'   explained_variance_ratio, singular_values, scores, n_components,
#'   n, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
pca_dimension_reduction <- function(x, n_components = NULL, seed = 0L) {
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x)
  n <- nrow(x); p <- ncol(x)
  k <- if (is.null(n_components)) min(n, p) else n_components
  pc <- stats::prcomp(x, center = TRUE, scale. = FALSE, rank. = k)
  sv <- pc$sdev[seq_len(k)]
  ev <- sv^2
  ratio <- ev / sum(pc$sdev^2)
  # In sklearn convention components_ is (k, p) -- rows = directions; prcomp rotation is (p, k)
  components <- t(pc$rotation[, seq_len(k), drop = FALSE])
  scores <- pc$x[, seq_len(k), drop = FALSE]
  list(
    estimate                 = as.numeric(ratio[1]),
    components               = components,
    explained_variance       = as.numeric(ev),
    explained_variance_ratio = as.numeric(ratio),
    singular_values          = as.numeric(sv * sqrt(n - 1)),  # match sklearn's S
    scores                   = scores,
    n_components             = as.integer(k),
    n                        = n,
    method                   = "PCA via SVD"
  )
}
