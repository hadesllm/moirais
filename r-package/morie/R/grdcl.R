# SPDX-License-Identifier: AGPL-3.0-or-later

#' Global-norm gradient clip (Pascanu 2013)
#'
#' @param x Numeric vector or list of gradient tensors.
#' @param max_norm Numeric clipping threshold (default 1).
#' @return Named list with tensor (clipped), clip_coef, total_norm, max_norm, method.
#' @keywords internal
gradient_clipping <- function(x, max_norm = 1) {
  is_list <- is.list(x)
  cat_vec <- if (is_list) unlist(lapply(x, as.numeric))
             else as.numeric(x)
  total <- sqrt(sum(cat_vec * cat_vec))
  coef <- min(1, max_norm / (total + 1e-12))
  clipped <- if (is_list) lapply(x, function(g) as.numeric(g) * coef)
             else as.numeric(x) * coef
  list(tensor = clipped, clip_coef = coef,
       total_norm = total, max_norm = max_norm,
       method = "global-norm-clip")
}
