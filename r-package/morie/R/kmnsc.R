# SPDX-License-Identifier: MIT OR Apache-2.0

#' K-means clustering (R parity)
#'
#' Wraps \code{stats::kmeans} with Hartigan-Wong (the default).
#'
#' @param x Numeric matrix.
#' @param n_clusters Number of clusters K.
#' @param n_init Number of random restarts.
#' @param max_iter Max Lloyd iterations.
#' @param seed RNG seed.
#' @return Named list: estimate (inertia), labels, centers, inertia,
#'   n_iter, n_clusters, n, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
kmeans_clustering <- function(x, n_clusters = 3L, n_init = 10L,
                               max_iter = 300L, seed = 0L) {
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x)
  set.seed(seed)
  fit <- stats::kmeans(x, centers = n_clusters, iter.max = max_iter,
                       nstart = n_init, algorithm = "Hartigan-Wong")
  list(
    estimate    = as.numeric(fit$tot.withinss),
    labels      = as.integer(fit$cluster - 1L),  # 0-indexed for Py parity
    centers     = fit$centers,
    inertia     = as.numeric(fit$tot.withinss),
    n_iter      = as.integer(fit$iter),
    n_clusters  = as.integer(n_clusters),
    n           = nrow(x),
    method      = "K-means (Hartigan-Wong)"
  )
}
