# SPDX-License-Identifier: MIT OR Apache-2.0
#' MRM-framework analyses on Toronto Police Service (TPS) open data
#'
#' Four callables for TPS public-release crime-incident CSVs, used in
#' the MRM empirical companion paper.
#'
#' Functions:
#' * `mrm_tps_levy_scaling()`: Hill-MLE Pareto exponent of inter-incident
#'   step-length distribution on the lat/long-coded event stream.
#' * `mrm_tps_moran_clustering()`: global Moran's I + DBSCAN cluster
#'   summary on the lat/long-coded event stream.
#' * `mrm_tps_neighbourhood_recurrence_km()`: Kaplan-Meier inter-event
#'   gap distribution per HOOD_158 neighbourhood.
#' * `mrm_tps_load_hawkes_refit()`: convenience loader that pulls the
#'   precomputed per-category Hawkes (Markovian + Weibull/sin)
#'   fits from the `paper_hawkes_refit.json` manifest if available.
#'
#' @name mrm_tps
NULL


# ---------------------------------------------------------------------------
# Internal: pairwise lat/long distance in kilometres (haversine)
# ---------------------------------------------------------------------------

.haversine_km <- function(lat1, lon1, lat2, lon2) {
  R <- 6371
  rad <- pi / 180
  dlat <- (lat2 - lat1) * rad
  dlon <- (lon2 - lon1) * rad
  a <- sin(dlat / 2)^2 + cos(lat1 * rad) * cos(lat2 * rad) * sin(dlon / 2)^2
  2 * R * asin(pmin(1, sqrt(a)))
}


# ---------------------------------------------------------------------------
# 1. Levy scaling
# ---------------------------------------------------------------------------

#' Levy-flight Hill-MLE exponent on TPS inter-incident step lengths
#'
#' Treats consecutive events in chronological order as a single stream
#' and computes the inter-event step length (km) via haversine on
#' WGS84 latitude/longitude. Returns the Hill-MLE exponent restricted
#' to steps above `min_step_km`.
#'
#' @param data A data.frame with at least the columns named in
#'   `date_col`, `lat_col`, `lon_col`.
#' @param date_col Column name of the date / timestamp
#'   (default `"OCC_DATE"`).
#' @param lat_col Column name of WGS84 latitude
#'   (default `"LAT_WGS84"`).
#' @param lon_col Column name of WGS84 longitude
#'   (default `"LONG_WGS84"`).
#' @param min_step_km Lower-tail cutoff in km (default `0.5`).
#' @param x_min Hill-MLE cutoff (default = `min_step_km`).
#' @return A list with `n_events`, `n_steps_tail`, `min_step_km`,
#'   `hill_alpha`.
#' @export
#' @examples
#' if (FALSE) {
#'   tps <- read.csv("Assault_Open_Data.csv")
#'   mrm_tps_levy_scaling(tps)
#' }
mrm_tps_levy_scaling <- function(
  data,
  date_col = "OCC_DATE",
  lat_col = "LAT_WGS84",
  lon_col = "LONG_WGS84",
  min_step_km = 0.5,
  x_min = NULL
) {
  stopifnot(is.data.frame(data))
  stopifnot(all(c(date_col, lat_col, lon_col) %in% names(data)))
  if (is.null(x_min)) x_min <- min_step_km

  ord <- order(as.character(data[[date_col]]))
  lat <- data[[lat_col]][ord]
  lon <- data[[lon_col]][ord]
  keep <- is.finite(lat) & is.finite(lon)
  lat <- lat[keep]; lon <- lon[keep]

  n <- length(lat)
  if (n < 2L) return(list(n_events = n, n_steps_tail = 0L,
                          min_step_km = min_step_km, hill_alpha = NA_real_))
  step <- .haversine_km(lat[-n], lon[-n], lat[-1], lon[-1])
  tail <- step[step >= min_step_km]
  alpha <- if (length(tail) >= 2L) 1 + length(tail) / sum(log(tail / x_min)) else NA_real_
  list(
    n_events = n,
    n_steps_tail = length(tail),
    min_step_km = min_step_km,
    hill_alpha = round(alpha, 4)
  )
}


# ---------------------------------------------------------------------------
# 2. Moran's I + DBSCAN clustering
# ---------------------------------------------------------------------------

#' Global Moran's I + DBSCAN summary on TPS lat/long event data
#'
#' Grids the lat/long extent of `data` into a coarse raster of
#' `grid_resolution` cells, counts events per cell, and computes the
#' global Moran's I via a rook contiguity matrix. Also runs DBSCAN on
#' the raw lat/long points (rescaled to km) and reports cluster
#' counts.
#'
#' This function is a thin computational wrapper. For high-precision
#' computations on full-sized TPS files use the `morie` Python
#' `tps_spatial_advanced` pipeline; the R version is for quick
#' interactive auditing.
#'
#' @param data A data.frame with `lat_col` and `lon_col`.
#' @param lat_col Column name of WGS84 latitude.
#' @param lon_col Column name of WGS84 longitude.
#' @param grid_resolution Number of cells per axis (default `40L`).
#' @param dbscan_eps DBSCAN radius in km (default `0.3`).
#' @param dbscan_minpts DBSCAN minimum points per core (default `5L`).
#' @return A list with `morans_I`, `morans_z`, `dbscan_n_clusters`,
#'   `dbscan_n_noise`, `dbscan_largest`.
#' @export
#' @examples
#' if (FALSE) {
#'   tps <- read.csv("Assault_Open_Data.csv")
#'   mrm_tps_moran_clustering(tps)
#' }
mrm_tps_moran_clustering <- function(
  data,
  lat_col = "LAT_WGS84",
  lon_col = "LONG_WGS84",
  grid_resolution = 40L,
  dbscan_eps = 0.3,
  dbscan_minpts = 5L
) {
  stopifnot(is.data.frame(data))
  stopifnot(all(c(lat_col, lon_col) %in% names(data)))
  lat <- data[[lat_col]]; lon <- data[[lon_col]]
  keep <- is.finite(lat) & is.finite(lon)
  lat <- lat[keep]; lon <- lon[keep]
  n <- length(lat)
  if (n < 10L) {
    return(list(morans_I = NA_real_, morans_z = NA_real_,
                dbscan_n_clusters = 0L, dbscan_n_noise = 0L,
                dbscan_largest = 0L))
  }

  # --- Moran via raster counts ---
  lat_breaks <- seq(min(lat), max(lat), length.out = grid_resolution + 1L)
  lon_breaks <- seq(min(lon), max(lon), length.out = grid_resolution + 1L)
  i_idx <- as.integer(cut(lat, lat_breaks, include.lowest = TRUE))
  j_idx <- as.integer(cut(lon, lon_breaks, include.lowest = TRUE))
  counts <- matrix(0L, grid_resolution, grid_resolution)
  for (k in seq_len(n)) counts[i_idx[k], j_idx[k]] <- counts[i_idx[k], j_idx[k]] + 1L
  z <- as.vector(counts) - mean(counts)
  N <- length(z)
  # rook contiguity neighbour pairs
  W_sum <- 0; num <- 0
  for (i in seq_len(grid_resolution)) for (j in seq_len(grid_resolution)) {
    if (i < grid_resolution) {
      num <- num + z[(j - 1) * grid_resolution + i] * z[(j - 1) * grid_resolution + i + 1L]
      W_sum <- W_sum + 1
    }
    if (j < grid_resolution) {
      num <- num + z[(j - 1) * grid_resolution + i] * z[j * grid_resolution + i]
      W_sum <- W_sum + 1
    }
  }
  num <- 2 * num   # rook is symmetric
  W_sum <- 2 * W_sum
  morans_I <- (N / W_sum) * num / sum(z^2)
  # Approximate z-score under randomisation H0 (E[I] = -1/(N-1))
  EI <- -1 / (N - 1)
  varI <- 2 / (N - 1)^2
  morans_z <- (morans_I - EI) / sqrt(varI)

  # --- DBSCAN ---
  if (requireNamespace("dbscan", quietly = TRUE)) {
    pts <- cbind(lat * 111, lon * 111 * cos(mean(lat) * pi / 180))
    db <- dbscan::dbscan(pts, eps = dbscan_eps, minPts = dbscan_minpts)
    cl <- db$cluster
    n_clusters <- length(unique(cl[cl != 0L]))
    n_noise <- sum(cl == 0L)
    largest <- if (n_clusters > 0L) max(table(cl[cl != 0L])) else 0L
  } else {
    n_clusters <- NA_integer_; n_noise <- NA_integer_; largest <- NA_integer_
  }

  list(
    morans_I = round(morans_I, 6),
    morans_z = round(morans_z, 2),
    dbscan_n_clusters = n_clusters,
    dbscan_n_noise = as.integer(n_noise),
    dbscan_largest = as.integer(largest)
  )
}


# ---------------------------------------------------------------------------
# 3. Neighbourhood inter-event recurrence KM
# ---------------------------------------------------------------------------

#' Kaplan-Meier inter-event recurrence on TPS by neighbourhood
#'
#' For each HOOD_158 neighbourhood, sorts events chronologically and
#' computes the gap (in days) between consecutive events. Returns the
#' per-neighbourhood mean, median, and total gaps. No censoring is
#' applied (every gap is observed).
#'
#' @param data A data.frame with `date_col` and `hood_col`.
#' @param date_col Column name of the date column
#'   (default `"OCC_DATE"`).
#' @param hood_col Column name of the neighbourhood ID
#'   (default `"HOOD_158"`).
#' @param min_gap_days Smallest gap to include (default `0`).
#' @return A data.frame with one row per neighbourhood, columns
#'   `hood`, `n_events`, `n_gaps`, `mean_gap_days`, `median_gap_days`,
#'   `p25_gap_days`, `p75_gap_days`.
#' @export
#' @examples
#' if (FALSE) {
#'   tps <- read.csv("Assault_Open_Data.csv")
#'   mrm_tps_neighbourhood_recurrence_km(tps)
#' }
mrm_tps_neighbourhood_recurrence_km <- function(
  data,
  date_col = "OCC_DATE",
  hood_col = "HOOD_158",
  min_gap_days = 0
) {
  stopifnot(is.data.frame(data))
  stopifnot(all(c(date_col, hood_col) %in% names(data)))
  # Tolerant date parser: TPS Open Data emits "M/D/YYYY HH:MM:SS AM/PM";
  # ArcGIS GeoJSON emits ISO; try a few formats before giving up.
  raw <- data[[date_col]]
  d <- suppressWarnings(as.POSIXct(raw, format = "%m/%d/%Y %I:%M:%S %p", tz = "UTC"))
  if (all(is.na(d))) d <- suppressWarnings(as.POSIXct(raw, format = "%Y-%m-%d %H:%M:%S", tz = "UTC"))
  if (all(is.na(d))) d <- suppressWarnings(as.POSIXct(raw, tz = "UTC"))
  d <- as.Date(d)
  h <- data[[hood_col]]
  ok <- !is.na(d) & !is.na(h)
  d <- d[ok]; h <- h[ok]
  ord <- order(h, d)
  h <- h[ord]; d <- d[ord]
  rows <- by(d, h, function(dd) {
    if (length(dd) < 2L) return(NULL)
    gaps <- as.numeric(diff(dd))
    gaps <- gaps[gaps >= min_gap_days]
    if (length(gaps) == 0L) return(NULL)
    data.frame(
      n_events = length(dd),
      n_gaps = length(gaps),
      mean_gap_days = round(mean(gaps), 2),
      median_gap_days = stats::median(gaps),
      p25_gap_days = stats::quantile(gaps, 0.25, names = FALSE),
      p75_gap_days = stats::quantile(gaps, 0.75, names = FALSE)
    )
  }, simplify = FALSE)
  out <- do.call(rbind, Map(cbind, hood = names(rows[!sapply(rows, is.null)]),
                            rows[!sapply(rows, is.null)]))
  rownames(out) <- NULL
  out
}


# ---------------------------------------------------------------------------
# 4. Hawkes precomputed loader
# ---------------------------------------------------------------------------

#' Load the precomputed per-category TPS Hawkes refit manifest
#'
#' Reads `paper_hawkes_refit.json` and returns the Markovian (exponential
#' kernel, constant baseline) and non-Markovian (Weibull kernel,
#' sinusoidal baseline) AIC, branching ratio, and KS p-value per
#' category as a tidy data.frame.
#'
#' @param manifest_path Path to `paper_hawkes_refit.json`.
#' @return A data.frame with one row per category, columns
#'   `category`, `n_fitted`, `T_days`, `aic_mark`, `kappa_mark`,
#'   `ks_p_mark`, `aic_nm`, `eta_nm`, `ks_p_nm`, `delta_aic`.
#' @export
#' @examples
#' if (FALSE) {
#'   mrm_tps_load_hawkes_refit("paper_hawkes_refit.json")
#' }
mrm_tps_load_hawkes_refit <- function(manifest_path) {
  stopifnot(file.exists(manifest_path))
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required for mrm_tps_load_hawkes_refit().")
  }
  d <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  cats <- names(d)
  rows <- lapply(cats, function(c) {
    e <- d[[c]]
    data.frame(
      category = c,
      n_fitted = e$n_fitted,
      T_days = round(e$T_days, 1),
      aic_mark = round(e$markovian$aic, 1),
      kappa_mark = round(e$markovian$branching_ratio, 3),
      ks_p_mark = round(e$markovian$ks_pvalue, 3),
      aic_nm = round(e$weibull_sin$aic, 1),
      eta_nm = round(e$weibull_sin$branching_ratio, 3),
      ks_p_nm = round(e$weibull_sin$ks_pvalue, 3),
      delta_aic = round(e$delta_aic, 1)
    )
  })
  do.call(rbind, rows)
}
