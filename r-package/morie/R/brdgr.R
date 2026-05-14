# SPDX-License-Identifier: MIT OR Apache-2.0

#' Bridge observations across sessions / chambers (Armstrong Ch 6)
#'
#' Counts legislators appearing in both sessions. With ID vectors:
#' returns set intersection. With matrices: rows non-empty in both.
#'
#' @param x Vector of session-1 IDs or session-1 vote matrix.
#' @param y Vector of session-2 IDs or session-2 vote matrix.
#' @return Named list with `n_bridges`, `bridge_ids`, `share`, `n1`,
#'   `n2`, `method`.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
brdgr <- function(x, y = NULL) {
  if (is.null(y)) {
    xb <- as.logical(x)
    return(list(n_bridges = sum(xb), bridge_ids = which(xb),
                share = sum(xb) / max(length(xb), 1L),
                n1 = length(xb), n2 = length(xb),
                method = "bridge_observations"))
  }
  if (!is.matrix(x) && !is.matrix(y)) {
    s1 <- unique(x); s2 <- unique(y); common <- sort(intersect(s1, s2))
    return(list(n_bridges = length(common), bridge_ids = common,
                share = length(common) / max(length(s1), 1L),
                n1 = length(s1), n2 = length(s2),
                method = "bridge_observations"))
  }
  if (!is.matrix(x) || !is.matrix(y) || nrow(x) != nrow(y))
    stop("x and y must be matrices with matching n rows")
  has1 <- rowSums(!is.na(x)) > 0
  has2 <- rowSums(!is.na(y)) > 0
  bridges <- has1 & has2
  list(n_bridges = sum(bridges), bridge_ids = which(bridges),
       share = sum(bridges) / max(nrow(x), 1L),
       n1 = sum(has1), n2 = sum(has2),
       method = "bridge_observations")
}

#' @keywords internal
#' @rdname brdgr
#' @export
bridge_observations <- brdgr
