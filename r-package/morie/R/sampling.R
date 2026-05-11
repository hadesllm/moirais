#' Survey sampling methods for MORIE
#'
#' Implements simple random sampling, stratified, cluster, PPS, bootstrap,
#' jackknife, calibration weights, effective sample size, and design effect.
#'
#' @name sampling
#' @keywords internal
NULL


# ---------------------------------------------------------------------------
# Simple random sampling
# ---------------------------------------------------------------------------

#' Simple random sample from a data frame
#'
#' @param df A data frame.
#' @param n Number of units to select.
#' @param replace Sample with replacement? Default `FALSE`.
#' @param seed Random seed for reproducibility.
#' @return A data frame of `n` sampled rows with a `.weight` column added.
#' @export
#' @examples
#' df <- data.frame(x = 1:100)
#' srs_sample <- simple_random_sample(df, 20)
simple_random_sample <- function(df, n, replace = FALSE, seed = 42L) {
  set.seed(seed)
  N <- nrow(df)
  if (n > N && !replace) stop("n exceeds population size for SRS WOR.")
  idx <- sample.int(N, size = n, replace = replace)
  out <- df[idx, , drop = FALSE]
  out$.weight <- if (replace) rep(1, n) else rep(N / n, n)
  out
}


# ---------------------------------------------------------------------------
# Stratified random sampling
# ---------------------------------------------------------------------------

#' Proportional or fixed stratified random sample
#'
#' @param df A data frame.
#' @param strata_col Name of the stratification column.
#' @param n_per_stratum Either an integer (equal allocation) or a named integer
#'   vector mapping stratum levels to sample sizes.  If `proportional = TRUE`,
#'   `n_per_stratum` is treated as the total desired sample size and allocation
#'   is proportional to stratum size.
#' @param proportional Logical; if `TRUE`, allocate proportionally to strata sizes.
#' @param seed Random seed.
#' @return Data frame of sampled rows with a `.weight` column.
#' @export
#' @examples
#' df <- data.frame(g = c(rep("A", 60), rep("B", 40)), x = rnorm(100))
#' stratified_sample(df, "g", n_per_stratum = 10)
stratified_sample <- function(df, strata_col, n_per_stratum,
                               proportional = FALSE, seed = 42L) {
  set.seed(seed)
  strata <- split(seq_len(nrow(df)), df[[strata_col]])
  strata_sizes <- lengths(strata)

  if (proportional) {
    total_n <- if (is.numeric(n_per_stratum) && length(n_per_stratum) == 1L) {
      n_per_stratum
    } else stop("For proportional = TRUE, supply a single integer for n_per_stratum.")
    alloc <- round(strata_sizes / sum(strata_sizes) * total_n)
    alloc <- pmax(alloc, 1L)
  } else {
    if (length(n_per_stratum) == 1L) {
      alloc <- setNames(rep(n_per_stratum, length(strata)), names(strata))
    } else {
      alloc <- n_per_stratum[names(strata)]
    }
  }

  rows <- unlist(mapply(function(idx, m) {
    sample(idx, size = min(m, length(idx)), replace = FALSE)
  }, strata, alloc, SIMPLIFY = FALSE))

  out <- df[rows, , drop = FALSE]
  weights <- strata_sizes[df[[strata_col]][rows]] / alloc[df[[strata_col]][rows]]
  out$.weight <- as.numeric(weights)
  out
}


# ---------------------------------------------------------------------------
# Cluster sampling
# ---------------------------------------------------------------------------

#' Two-stage cluster sampling
#'
#' Randomly selects `n_clusters` clusters, then takes all units within
#' selected clusters.
#'
#' @param df A data frame.
#' @param cluster_col Name of the cluster identifier column.
#' @param n_clusters Number of clusters to select.
#' @param seed Random seed.
#' @return Data frame of selected units with `.weight` column.
#' @export
cluster_sample <- function(df, cluster_col, n_clusters, seed = 42L) {
  set.seed(seed)
  all_clusters <- unique(df[[cluster_col]])
  N_clusters <- length(all_clusters)
  if (n_clusters > N_clusters) stop("n_clusters exceeds total number of clusters.")
  selected <- sample(all_clusters, n_clusters, replace = FALSE)
  out <- df[df[[cluster_col]] %in% selected, , drop = FALSE]
  out$.weight <- rep(N_clusters / n_clusters, nrow(out))
  out
}


# ---------------------------------------------------------------------------
# PPS sampling
# ---------------------------------------------------------------------------

#' Probability proportional to size (PPS) sampling
#'
#' @param df A data frame.
#' @param size_col Name of the size measure column.
#' @param n Number of units to select.
#' @param seed Random seed.
#' @return Data frame of selected units with `.weight` (Hansen-Hurwitz weights).
#' @export
pps_sample <- function(df, size_col, n, seed = 42L) {
  set.seed(seed)
  sizes <- as.numeric(df[[size_col]])
  if (any(sizes <= 0, na.rm = TRUE)) stop("size_col must be positive.")
  probs <- sizes / sum(sizes, na.rm = TRUE)
  idx <- sample.int(nrow(df), size = n, replace = TRUE, prob = probs)
  out <- df[idx, , drop = FALSE]
  # Hansen-Hurwitz weight: 1 / (n * pi_i)
  out$.weight <- 1 / (n * probs[idx])
  out
}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

#' Bootstrap resampling for any statistic
#'
#' @param df A data frame.
#' @param statistic A function taking a data frame and returning a scalar.
#' @param n_bootstrap Number of bootstrap replicates.
#' @param seed Random seed.
#' @return Named list: `estimate`, `se`, `ci_lower`, `ci_upper`,
#'   `distribution` (numeric vector of bootstrap statistics).
#' @export
#' @examples
#' df <- data.frame(x = rnorm(100))
#' bootstrap_sample(df, statistic = function(d) mean(d$x))
bootstrap_sample <- function(df, statistic, n_bootstrap = 1000L, seed = 42L) {
  set.seed(seed)
  n <- nrow(df)
  boot_stats <- vapply(seq_len(n_bootstrap), function(i) {
    idx <- sample.int(n, n, replace = TRUE)
    statistic(df[idx, , drop = FALSE])
  }, numeric(1))
  est  <- mean(boot_stats)
  se   <- stats::sd(boot_stats)
  ci   <- stats::quantile(boot_stats, c(0.025, 0.975))
  list(
    estimate    = est,
    se          = se,
    ci_lower    = ci[1],
    ci_upper    = ci[2],
    distribution = boot_stats
  )
}


# ---------------------------------------------------------------------------
# Jackknife
# ---------------------------------------------------------------------------

#' Delete-1 jackknife variance estimate
#'
#' @param df A data frame.
#' @param statistic A function taking a data frame and returning a scalar.
#' @return Named list: `estimate`, `se`, `bias`.
#' @export
jackknife_estimate <- function(df, statistic) {
  n <- nrow(df)
  theta_full <- statistic(df)
  theta_minus_i <- vapply(seq_len(n), function(i) {
    statistic(df[-i, , drop = FALSE])
  }, numeric(1))
  theta_bar <- mean(theta_minus_i)
  se   <- sqrt((n - 1) / n * sum((theta_minus_i - theta_bar)^2))
  bias <- (n - 1) * (theta_bar - theta_full)
  list(estimate = theta_full, se = se, bias = bias)
}


# ---------------------------------------------------------------------------
# Effective sample size and design effect
# ---------------------------------------------------------------------------

#' Kish effective sample size
#'
#' @param weights Numeric vector of sampling weights.
#' @return Numeric ESS.
#' @export
effective_sample_size <- function(weights) {
  w <- as.numeric(weights)
  w <- w[!is.na(w) & w > 0]
  (sum(w)^2) / sum(w^2)
}

#' Design effect (DEFF)
#'
#' @param weights Numeric vector of sampling weights.
#' @return Numeric design effect (= n / ESS).
#' @export
design_effect <- function(weights) {
  w <- as.numeric(weights)
  w <- w[!is.na(w) & w > 0]
  length(w) / effective_sample_size(w)
}


# ---------------------------------------------------------------------------
# Design weights
# ---------------------------------------------------------------------------

#' Compute inverse-probability design weights
#'
#' @param df A data frame.
#' @param strata_col Name of the stratification column.
#' @param population_sizes Named integer vector: stratum level -> population size.
#' @return Numeric vector of design weights (same length as `nrow(df)`).
#' @export
compute_design_weights <- function(df, strata_col, population_sizes) {
  strata <- df[[strata_col]]
  sample_sizes <- table(strata)
  pop_sizes <- population_sizes[names(sample_sizes)]
  weights <- pop_sizes[strata] / sample_sizes[strata]
  as.numeric(weights)
}


# ---------------------------------------------------------------------------
# Calibration (raking / IPF)
# ---------------------------------------------------------------------------

#' Calibration weights via iterative proportional fitting (raking)
#'
#' Adjusts initial design weights so that weighted marginal totals match
#' known population totals for each auxiliary variable.
#'
#' @param df A data frame.
#' @param aux_vars Character vector of categorical auxiliary variable names.
#' @param population_totals Named list: `"var_level"` -> population count.
#'   Keys should be `"varname_level"` (e.g. `"gender_female"`).
#' @param initial_weights Optional numeric vector of starting weights.
#' @param max_iter Maximum IPF iterations.
#' @param tol Convergence tolerance.
#' @return Numeric vector of calibrated weights.
#' @export
calibration_weights <- function(df, aux_vars, population_totals,
                                 initial_weights = NULL,
                                 max_iter = 50L, tol = 1e-6) {
  n <- nrow(df)
  w <- if (!is.null(initial_weights)) initial_weights else rep(1, n)

  for (iter in seq_len(max_iter)) {
    w_old <- w
    for (v in aux_vars) {
      levels_v <- unique(df[[v]])
      for (lv in levels_v) {
        key <- paste0(v, "_", lv)
        if (!key %in% names(population_totals)) next
        pop_tot <- population_totals[[key]]
        mask    <- df[[v]] == lv
        sample_tot <- sum(w[mask])
        if (sample_tot > 0) w[mask] <- w[mask] * pop_tot / sample_tot
      }
    }
    if (max(abs(w - w_old)) < tol) break
  }
  w
}
