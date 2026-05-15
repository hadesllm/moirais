# SPDX-License-Identifier: AGPL-3.0-or-later

#' Repetition penalty (Keskar 2019)
#'
#' @param x Numeric vector of logits.
#' @param generated Integer vector of already-generated token ids.
#' @param alpha Numeric penalty factor (default 1.2).
#' @return Named list with tensor (penalised logits), penalised_idx, alpha, method.
#' @keywords internal
repetition_penalty <- function(x, generated, alpha = 1.2) {
  z <- as.numeric(x)
  if (alpha == 1) return(list(tensor = z, penalised_idx = integer(0),
                               alpha = alpha, method = "rep-penalty"))
  idx <- unique(as.integer(generated))
  idx <- idx[idx >= 0L & idx < length(z)]
  for (i in idx)
    z[i + 1L] <- if (z[i + 1L] > 0) z[i + 1L] / alpha else z[i + 1L] * alpha
  list(tensor = z, penalised_idx = idx, alpha = alpha,
       method = "rep-penalty")
}
