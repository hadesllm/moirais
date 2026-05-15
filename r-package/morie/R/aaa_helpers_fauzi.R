# SPDX-License-Identifier: AGPL-3.0-or-later

# Shared Fauzi-suite helpers.
#
# Sourced before the rest of the fz*.R files so callers don't depend on R's
# alphabetical load order. Loaded first via the leading underscore in the
# filename and via the explicit Collate: field in DESCRIPTION.

#' @importFrom stats sd quantile
#' @noRd
.morie_silverman_h <- function(x) {
  n <- length(x); s <- stats::sd(x)
  iq <- diff(stats::quantile(x, c(.25, .75), names = FALSE)) / 1.34
  sigma <- if (iq > 0) min(s, iq) else s
  if (sigma <= 0) sigma <- 1
  1.06 * sigma * n^(-1/5)
}
