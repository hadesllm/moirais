# SPDX-License-Identifier: GPL-2.0-only

#' Load a bundled reference sample CSV by name
#'
#' Parity with the Python \code{morie.load_sample()} helper.  Reads a
#' small reference dataset shipped under \code{inst/extdata/samples/}.
#'
#' @param name One of \code{"otis_b01"}, \code{"otis_b09"},
#'   \code{"otis_c11"}, \code{"tps_assault"}.
#' @return A \code{data.frame} loaded from the bundled CSV.
#' @examples
#' df <- morie_sample("otis_b01")
#' head(df)
#' @export
morie_sample <- function(name) {
  files <- list(
    otis_b01    = "otis_b01_sample.csv",
    otis_b09    = "otis_b09_sample.csv",
    otis_c11    = "otis_c11_sample.csv",
    tps_assault = "tps_assault_sample.csv"
  )
  if (!name %in% names(files)) {
    stop(sprintf("Unknown sample '%s'; choices: %s", name,
                 paste(names(files), collapse = ", ")))
  }
  path <- system.file("extdata", "samples", files[[name]], package = "morie")
  if (path == "") {
    stop(sprintf("Sample CSV for '%s' not found in installed package.", name))
  }
  utils::read.csv(path)
}
