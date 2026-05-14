# SPDX-License-Identifier: MIT OR Apache-2.0

#' Kulldorff space-time scan statistic on TPS event data
#'
#' R parity of \code{morie.mrm_kulldorff.mrm_tps_kulldorff_scan()}.
#' Implements Kulldorff's 1997 Poisson log-likelihood-ratio space-time
#' scan with a Monte-Carlo permutation test for significance.
#'
#' The scan iterates over (centre, radius, time-window) tuples,
#' computing the Poisson LRT against \eqn{H_0} (events uniformly
#' distributed in space and time). The maximum LRT is the test
#' statistic; permutations of event timestamps generate the null.
#'
#' @references
#' Kulldorff, M. (1997). A spatial scan statistic. Communications in
#' Statistics: Theory and Methods, 26(6), 1481--1496.
#'
#' @name mrm_kulldorff
NULL


.haversine_km_mat <- function(lat1, lon1, lat2, lon2) {
  R <- 6371; rad <- pi / 180
  dlat <- (lat2 - lat1) * rad
  dlon <- (lon2 - lon1) * rad
  a <- sin(dlat / 2)^2 + cos(lat1 * rad) * cos(lat2 * rad) * sin(dlon / 2)^2
  2 * R * asin(pmin(1, sqrt(a)))
}


.poisson_lrt <- function(n_obs, n_in, n_exp, n_tot) {
  if (n_in == 0 || n_obs == 0 || n_obs <= n_exp) return(0.0)
  obs_out <- n_tot - n_obs
  exp_out <- n_tot - n_exp
  if (obs_out == 0 || exp_out <= 0) return(0.0)
  n_obs * log(n_obs / n_exp) + obs_out * log(obs_out / exp_out)
}


#' Run a 3-d (lat, lon, time) Kulldorff scan with MC inference
#'
#' @param data data.frame with date_col, lat_col, lon_col.
#' @param date_col Column name of the event date (default
#'   \code{"OCC_DATE"}).
#' @param lat_col,lon_col WGS84 lat/long column names.
#' @param radii_km Candidate cylinder radii in km.
#' @param window_years Time-cylinder length in years.
#' @param n_centers Number of random candidate centres sub-sampled.
#' @param n_permutations Monte-Carlo permutations.
#' @param seed Random seed.
#' @return A one-row data.frame describing the top cluster, with
#'   columns \code{center_lat}, \code{center_lon}, \code{radius_km},
#'   \code{t_start}, \code{t_end}, \code{n_observed}, \code{n_expected},
#'   \code{relative_risk}, \code{log_lrt}, \code{p_value}.
#' @export
#' @examples
#' if (FALSE) {
#'   tps <- morie_sample("tps_assault")
#'   mrm_tps_kulldorff_scan(tps, n_permutations = 49)
#' }
mrm_tps_kulldorff_scan <- function(
  data,
  date_col = "OCC_DATE",
  lat_col  = "LAT_WGS84",
  lon_col  = "LONG_WGS84",
  radii_km = c(1, 2, 3, 5, 8),
  window_years = 4,
  n_centers = 60L,
  n_permutations = 199L,
  seed = 42L
) {
  stopifnot(is.data.frame(data))
  stopifnot(all(c(date_col, lat_col, lon_col) %in% names(data)))
  set.seed(as.integer(seed))

  df <- data[, c(date_col, lat_col, lon_col)]
  d  <- suppressWarnings(as.POSIXct(df[[date_col]],
                                    format = "%m/%d/%Y %I:%M:%S %p",
                                    tz = "UTC"))
  if (all(is.na(d))) d <- suppressWarnings(as.POSIXct(df[[date_col]], tz = "UTC"))
  df[[date_col]] <- as.Date(d)
  df <- df[stats::complete.cases(df), ]
  df <- df[order(df[[date_col]]), ]
  n <- nrow(df)
  if (n < 100L) return(data.frame())

  lat <- df[[lat_col]]; lon <- df[[lon_col]]
  t   <- as.integer(df[[date_col]])
  center_idx <- sample.int(n, min(n_centers, n))
  window_days <- round(window_years * 365.25)
  starts <- seq(min(t), max(t) - window_days, length.out = max(2L, (max(t) - min(t)) %/% window_days))

  scan_one <- function(t_obs) {
    best <- list(lrt = 0, ci = -1L, ri = -1L, ti = -1L, n_in = 0L, n_space = 0L)
    for (ci in center_idx) {
      d_km <- .haversine_km_mat(lat[ci], lon[ci], lat, lon)
      for (ri in seq_along(radii_km)) {
        r <- radii_km[ri]
        in_space <- d_km <= r
        n_space <- sum(in_space)
        if (n_space < 5L) next
        for (ti in starts) {
          in_time <- (t_obs >= ti) & (t_obs < ti + window_days)
          if (!any(in_time)) next
          in_cyl <- in_space & in_time
          n_in_cyl <- sum(in_cyl)
          if (n_in_cyl < 5L) next
          n_exp <- n_space * sum(in_time) / n
          lrt <- .poisson_lrt(n_in_cyl, n_space, n_exp, n)
          if (lrt > best$lrt) {
            best <- list(lrt = lrt, ci = ci, ri = ri, ti = ti,
                         n_in = n_in_cyl, n_space = n_space)
          }
        }
      }
    }
    best
  }

  obs  <- scan_one(t)
  null <- numeric(n_permutations)
  for (k in seq_len(n_permutations)) null[k] <- scan_one(sample(t))$lrt
  p_value <- (sum(null >= obs$lrt) + 1) / (n_permutations + 1)

  if (obs$ci < 0) return(data.frame())

  r       <- radii_km[obs$ri]
  t_start <- as.Date(obs$ti, origin = "1970-01-01")
  t_end   <- as.Date(obs$ti + window_days, origin = "1970-01-01")
  n_exp   <- obs$n_space * window_days / max(1L, max(t) - min(t))
  rr      <- obs$n_in / n_exp

  data.frame(
    center_lat   = lat[obs$ci],
    center_lon   = lon[obs$ci],
    radius_km    = r,
    t_start      = t_start,
    t_end        = t_end,
    n_observed   = obs$n_in,
    n_expected   = round(n_exp, 2),
    relative_risk = round(rr, 3),
    log_lrt      = round(obs$lrt, 2),
    p_value      = round(p_value, 4)
  )
}
