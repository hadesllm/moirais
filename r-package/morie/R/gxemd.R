# SPDX-License-Identifier: GPL-2.0-only

#' Two-way GxE ANOVA with EMS variance components
#'
#' @param x Genotype IDs (length n).
#' @param y Numeric response.
#' @param env Environment IDs (length n).
#' @return list(estimate, g, e, ge, var_g, var_e, var_ge, var_eps, se, n, method).
#' @references Montesinos Lopez Ch 11.
#' @export
gxe_interaction_model <- function(x, y, env) {
  g_id <- x; e_id <- env; yv <- as.numeric(y); n <- length(yv)
  g_levels <- unique(g_id); e_levels <- unique(e_id)
  G <- length(g_levels); E <- length(e_levels)
  mu <- mean(yv)
  g_eff <- sapply(g_levels, function(lv) mean(yv[g_id == lv]) - mu)
  e_eff <- sapply(e_levels, function(lv) mean(yv[e_id == lv]) - mu)
  cell_mean <- matrix(NA_real_, G, E)
  cell_count <- matrix(0L, G, E)
  for (i in seq_along(g_levels)) for (j in seq_along(e_levels)) {
    msk <- g_id == g_levels[i] & e_id == e_levels[j]
    cell_count[i, j] <- sum(msk)
    if (any(msk)) cell_mean[i, j] <- mean(yv[msk])
  }
  ge_eff <- cell_mean - mu - outer(g_eff, e_eff, "+") + 0
  # Equivalent: cell_mean - mu - g_eff[i] - e_eff[j]
  ge_eff <- sweep(sweep(cell_mean - mu, 1, g_eff, "-"), 2, e_eff, "-")
  n_per_cell <- if (sum(cell_count) > 0) mean(cell_count) else 1
  ss_g <- sum(rowSums(cell_count) * g_eff^2)
  ss_e <- sum(colSums(cell_count) * e_eff^2)
  valid <- !is.na(ge_eff)
  ss_ge <- sum(cell_count[valid] * ge_eff[valid]^2)
  resid <- rep(0, n)
  for (i in seq_along(g_levels)) for (j in seq_along(e_levels)) {
    msk <- g_id == g_levels[i] & e_id == e_levels[j]
    if (any(msk)) resid[msk] <- yv[msk] - cell_mean[i, j]
  }
  ss_eps <- sum(resid^2)
  df_g <- max(G - 1, 1); df_e <- max(E - 1, 1)
  df_ge <- max((G - 1) * (E - 1), 1); df_eps <- max(n - G * E, 1)
  ms_eps <- ss_eps / df_eps; ms_ge <- ss_ge / df_ge
  var_eps <- ms_eps
  var_ge <- max(0, (ms_ge - ms_eps) / max(n_per_cell, 1))
  var_g <- max(0, (ss_g / df_g - ms_ge) / max(E * n_per_cell, 1))
  var_e <- max(0, (ss_e / df_e - ms_ge) / max(G * n_per_cell, 1))
  list(estimate = mu, g = g_eff, e = e_eff, ge = ge_eff,
       var_g = var_g, var_e = var_e, var_ge = var_ge, var_eps = var_eps,
       se = sqrt(var_eps), n = n,
       method = "Two-way GxE ANOVA + EMS variance components")
}

# CANONICAL TEST
# x <- c(1,1,2,2,3,3,1,1,2,2,3,3); env <- c(1,1,1,1,1,1,2,2,2,2,2,2)
# y <- c(1,2,3,4,5,6,2,3,4,5,6,7); gxe_interaction_model(x, y, env)
