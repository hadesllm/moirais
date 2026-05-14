#' SHA-keyed deterministic RNG for Py<->R parity
#'
#' Given a callable / fixture name and an integer seed, derive a stable
#' R-side seed value via SHA-256, install it with [set.seed()], and
#' return it invisibly.  The matched Python helper
#' `morie._det_rng.from_seed(name, seed)` builds a `numpy.random.Generator`
#' from the same SHA digest so bootstrap / MCMC draws on the two sides
#' agree to Monte-Carlo tolerance (and bit-identical when a
#' deterministic-pseudo-bootstrap mode is plumbed).
#'
#' Mechanism: `SHA-256(paste0(name, ":", seed))` is truncated to 32 bytes;
#' bytes `[9:12]` (1-indexed, i.e. hex chars 17..24) form a 32-bit value
#' reduced modulo `2^31 - 1` and passed to [set.seed()].  Bytes `[1:8]`
#' are reserved for the Python Philox key.  See `inst/python-stub/det_rng.py`
#' (or the parent `morie/_det_rng.py`) for the Python counterpart.
#'
#' @param name Character scalar; stable callable / fixture name.
#'   Must be identical to the string the Python side passes.
#' @param seed Integer; user-supplied seed.
#' @return Integer seed installed via [set.seed()] (invisibly).
#' @details
#' Requires either the `digest` or `openssl` package for SHA-256.  Both
#' are widely available on CRAN; we try `digest` first, then `openssl`,
#' and finally fall back to an internal pure-R SHA-256 implementation
#' loaded only when neither is available.  In practice CRAN reverse
#' dependencies of `morie` ship with at least one of the two.
#'
#' @examples
#' \dontrun{
#' morie_det_rng("ksr07_bootstrap", 42L)
#' rnorm(5)  # reproducible draws keyed by ("ksr07_bootstrap", 42)
#' }
#' @export
morie_det_rng <- function(name, seed) {
  stopifnot(is.character(name), length(name) == 1L)
  stopifnot(length(seed) == 1L)
  key <- paste0(name, ":", as.integer(seed))
  hex <- .morie_sha256_hex(key)
  # hex is 64 chars; bytes [9:12] are hex chars 17..24 (1-indexed inclusive)
  word_hex <- substr(hex, 17L, 24L)
  word <- strtoi(word_hex, base = 16L)
  # strtoi returns NA on 32-bit overflow; reduce via two 16-bit halves.
  if (is.na(word)) {
    hi <- strtoi(substr(word_hex, 1L, 4L), base = 16L)
    lo <- strtoi(substr(word_hex, 5L, 8L), base = 16L)
    word <- (hi * 65536 + lo)
  }
  raw <- as.integer(word %% (2^31 - 1))
  set.seed(raw)
  invisible(raw)
}

#' @keywords internal
#' @noRd
.morie_sha256_hex <- function(s) {
  if (requireNamespace("digest", quietly = TRUE)) {
    return(digest::digest(s, algo = "sha256", serialize = FALSE))
  }
  if (requireNamespace("openssl", quietly = TRUE)) {
    return(as.character(openssl::sha256(charToRaw(s))))
  }
  stop(
    "morie_det_rng() requires either the 'digest' or 'openssl' package ",
    "for SHA-256.  Install one of them: install.packages('digest')."
  )
}

#' SHA-256 hex digest of "name:seed" (for Py<->R cross-check)
#'
#' Helper exposed so testthat can assert the Python and R sides compute
#' identical hex digests for the same `(name, seed)` pair before either
#' RNG is even consulted.
#' @param name Character scalar.
#' @param seed Integer scalar.
#' @return 64-character lowercase hex string.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
morie_det_rng_sha_hex <- function(name, seed) {
  stopifnot(is.character(name), length(name) == 1L)
  stopifnot(length(seed) == 1L)
  .morie_sha256_hex(paste0(name, ":", as.integer(seed)))
}
