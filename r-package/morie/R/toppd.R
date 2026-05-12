# SPDX-License-Identifier: GPL-2.0-only

#' Top-p nucleus sampling (Holtzman 2020)
#'
#' @param x Numeric vector of logits.
#' @param p Numeric nucleus mass cutoff in (0, 1] (default 0.9).
#' @param T Numeric softmax temperature (default 1).
#' @return Named list with tensor, keep_mask, n_kept, p, method.
#' @keywords internal
top_p_nucleus <- function(x, p = 0.9, T = 1) {
  if (p <= 0 || p > 1) stop("p must be in (0, 1]")
  z <- as.numeric(x) / T; z <- z - max(z)
  probs <- exp(z); probs <- probs / sum(probs)
  ord <- order(-probs)
  cs <- cumsum(probs[ord])
  cutoff <- max(1L, min(which(cs >= p)[1L], length(probs)))
  keep <- logical(length(probs))
  keep[ord[seq_len(cutoff)]] <- TRUE
  filtered <- ifelse(keep, probs, 0)
  filtered <- filtered / sum(filtered)
  list(tensor = filtered, keep_mask = keep, n_kept = sum(keep),
       p = p, method = "top-p")
}
