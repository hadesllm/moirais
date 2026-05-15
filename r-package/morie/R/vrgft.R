# SPDX-License-Identifier: AGPL-3.0-or-later
#' Variogram model fit by weighted least squares.
#'
#' Models: exponential, gaussian, spherical.
#' Exponential form: \deqn{\gamma(h) = c_0 + c_1(1 - e^{-h/a})}.
#'
#' @param x Numeric vector.
#' @param coords Numeric coordinate matrix.
#' @param model Character; one of "exponential", "gaussian", "spherical".
#' @param n_bins Number of distance bins for the empirical variogram.
#' @param max_dist Upper distance cutoff.
#' @return Named list with estimate (nugget, sill, range, params,
#'   converged, model), n, method.
#' @references Cressie (1985); Schabenberger & Gotway (2005), Ch 3.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
vrgft <- function(x, coords, model = "exponential",
                  n_bins = 10, max_dist = NULL) {
  ev <- vrgm(x, coords, n_bins = n_bins, max_dist = max_dist)
  mids <- ev$estimate$bins; gammas <- ev$estimate$gamma
  npairs <- ev$estimate$n_pairs
  keep <- !is.na(gammas) & npairs > 0
  mids <- mids[keep]; gammas <- gammas[keep]; npairs <- npairs[keep]
  while (length(mids) < 3 && n_bins > 3) {
    n_bins <- n_bins - 1
    ev <- vrgm(x, coords, n_bins = n_bins, max_dist = max_dist)
    mids <- ev$estimate$bins; gammas <- ev$estimate$gamma
    npairs <- ev$estimate$n_pairs
    keep <- !is.na(gammas) & npairs > 0
    mids <- mids[keep]; gammas <- gammas[keep]; npairs <- npairs[keep]
  }
  if (length(mids) < 3) stop("need at least 3 non-empty bins")
  g_max <- max(gammas); h_max <- max(mids)
  p0 <- c(0, g_max, max(h_max / 3, 1e-6))
  model_fn <- function(h, c0, c1, a) {
    switch(model,
           exponential = c0 + c1 * (1 - exp(-h / a)),
           gaussian    = c0 + c1 * (1 - exp(-(h ^ 2) / (a ^ 2))),
           spherical   = ifelse(h <= a,
                                c0 + c1 * (1.5 * h / a - 0.5 * (h / a) ^ 3),
                                c0 + c1),
           stop("unknown model"))
  }
  weights <- pmax(npairs, 1) / pmax(gammas, 1e-12) ^ 2
  obj <- function(p) {
    pred <- model_fn(mids, p[1], p[2], p[3])
    sum(weights * (gammas - pred) ^ 2)
  }
  res <- tryCatch(
    stats::optim(p0, obj, method = "L-BFGS-B",
                 lower = c(0, 1e-12, 1e-12),
                 upper = c(g_max * 5 + 1e-6, g_max * 10 + 1, h_max * 10)),
    error = function(e) list(par = p0, convergence = -1))
  c0 <- res$par[1]; c1 <- res$par[2]; a <- res$par[3]
  list(
    estimate = list(model = model, nugget = c0, sill = c0 + c1, range = a,
                    params = c(c0, c1, a),
                    converged = isTRUE(res$convergence == 0)),
    n = length(x), method = sprintf("Variogram model fit (%s, WLS)", model)
  )
}

# CANONICAL TEST
# vrgft(c(1,2,3,4,5), matrix(0:4, ncol=1), "exponential", 4, 4)

#' @rdname vrgft
#' @keywords internal
#' @export
variogram_fitting <- vrgft
