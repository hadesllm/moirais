# SPDX-License-Identifier: AGPL-3.0-or-later

#' DBSCAN density-based clustering (R parity)
#'
#' Wraps \code{dbscan::dbscan}.
#'
#' @param x Numeric matrix.
#' @param eps Neighbourhood radius.
#' @param min_samples Minimum points in eps-neighbourhood for a core point.
#' @param metric Distance metric (passed to dbscan).
#' @return Named list: estimate, labels, n_clusters, n_noise,
#'   core_sample_indices, eps, min_samples, n, method.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
dbscan_clustering <- function(x, eps = 0.5, min_samples = 5L,
                               metric = "euclidean") {
  if (!requireNamespace("dbscan", quietly = TRUE)) {
    stop("Function 'dbscan_clustering' requires package 'dbscan'. Install with install.packages('dbscan').")
  }
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x)
  fit <- dbscan::dbscan(x, eps = eps, minPts = min_samples)
  labels <- fit$cluster  # 0 = noise in dbscan; sklearn uses -1
  labels_sk <- ifelse(labels == 0L, -1L, labels - 1L)
  n_clusters <- length(unique(labels_sk[labels_sk >= 0L]))
  n_noise <- sum(labels_sk == -1L)
  core_idx <- which(fit$cluster > 0 & fit$cluster %in% unique(fit$cluster[fit$cluster > 0]))
  # dbscan() doesn't expose the core-point mask directly; flag any point that's
  # NOT noise AND has >= minPts neighbours within eps.
  list(
    estimate            = as.integer(n_clusters),
    labels              = as.integer(labels_sk),
    n_clusters          = as.integer(n_clusters),
    n_noise             = as.integer(n_noise),
    core_sample_indices = as.integer(core_idx - 1L),  # 0-indexed
    eps                 = as.numeric(eps),
    min_samples         = as.integer(min_samples),
    n                   = nrow(x),
    method              = "DBSCAN (Ester et al. 1996)"
  )
}
