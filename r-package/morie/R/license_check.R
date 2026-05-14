# SPDX-License-Identifier: MIT OR Apache-2.0

#' Runtime license-compatibility guard for morie
#'
#' R parity of \code{morie._license_check}.  Exposes the
#' FSF GPL-compatible licence list and a
#' \code{check_plugin_license()} helper that downstream R packages /
#' plugins can call to confirm GPL compatibility before linking
#' against morie internals.  The guard is advisory --- it warns or
#' raises but does not enforce at the R-namespace level.  For
#' stronger guarantees see the companion userspace LSM-style daemon
#' (\code{daemon/morie_lsm.py}) and the kernel companion module
#' (\code{kernel-module/morie.c}).
#'
#' @name license_check
NULL


#' Vector of SPDX identifiers recognised as GPL-compatible
#'
#' Mirrors the FSF list at
#' \url{https://www.gnu.org/licenses/license-list.html}.  Apache-2.0
#' is GPL-3 compatible but not GPL-2 compatible; morie is
#' GPL-2.0-only so the choice rests with downstream consumers.
#'
#' @return Character vector of SPDX identifiers.
#' @export
morie_gpl_compatible_licenses <- function() {
  c("GPL-2.0-only", "GPL-2.0-or-later",
    "GPL-3.0-only", "GPL-3.0-or-later",
    "LGPL-2.1-only", "LGPL-2.1-or-later",
    "LGPL-3.0-only", "LGPL-3.0-or-later",
    "Apache-2.0",
    "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC",
    "MPL-2.0",
    "CC0-1.0",
    "Unlicense",
    "Zlib")
}


#' morie's SPDX-style licence metadata
#'
#' @return A named list summarising morie's licence posture, useful for
#'   pipeline build manifests, auditd logs, and downstream
#'   compliance pipelines.
#' @export
morie_license_metadata <- function() {
  list(
    package = "morie",
    spdx = "GPL-2.0-only",
    fsf_libre = "yes",
    osi_approved = "yes",
    kernel_compatible = "yes (MODULE_LICENSE(\"GPL v2\") accepts this)"
  )
}


#' Check whether a downstream package's SPDX is GPL-compatible
#'
#' @param plugin_spdx SPDX identifier (e.g. \code{"MIT"},
#'   \code{"Apache-2.0"}).
#' @param raise_on_incompatible If \code{TRUE}, throw an error rather
#'   than warning when the licence is not GPL-compatible.
#' @return \code{TRUE} if compatible.  Issues a warning (or error)
#'   otherwise.
#' @export
#' @examples
#' morie_check_plugin_license("MIT")
#' \dontrun{
#'   morie_check_plugin_license("LicenseRef-Proprietary",
#'                              raise_on_incompatible = TRUE)
#' }
morie_check_plugin_license <- function(plugin_spdx,
                                        raise_on_incompatible = FALSE) {
  if (is.null(plugin_spdx) || !nzchar(plugin_spdx)) {
    msg <- "Plugin reports empty SPDX identifier."
    if (raise_on_incompatible) stop(msg) else warning(msg, call. = FALSE)
    return(FALSE)
  }
  ok <- plugin_spdx %in% morie_gpl_compatible_licenses()
  if (!ok) {
    msg <- sprintf(
      "Plugin SPDX '%s' is not on the FSF GPL-compatible list; linking against morie may violate GPL-2.0-only.",
      plugin_spdx
    )
    if (raise_on_incompatible) stop(msg) else warning(msg, call. = FALSE)
  }
  ok
}
