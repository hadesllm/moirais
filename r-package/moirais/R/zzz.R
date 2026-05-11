#' @keywords internal
.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    "moirais was renamed to morie in v0.1.3. ",
    "library(morie) is the canonical entry point; ",
    "library(moirais) remains as a deprecated alias."
  )
}
