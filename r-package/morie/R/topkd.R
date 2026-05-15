# SPDX-License-Identifier: AGPL-3.0-or-later

#' Top-k filtered softmax (Fan 2018)
#'
#' @param x Numeric vector of logits.
#' @param k Integer truncation rank (default 5).
#' @param T Numeric softmax temperature (default 1).
#' @return Named list with tensor, topk_indices, topk_logits, k, method.
#' @keywords internal
top_k_decoding <- function(x, k = 5L, T = 1) {
  z <- as.numeric(x) / T
  Vlen <- length(z)
  k <- max(1L, min(as.integer(k), Vlen))
  thresh <- sort(z, decreasing = TRUE)[k]
  z_f <- ifelse(z >= thresh, z, -Inf)
  z_f <- z_f - max(z_f)
  e <- exp(z_f); p <- e / sum(e)
  topk_idx <- order(z, decreasing = TRUE)[seq_len(k)]
  list(tensor = p, topk_indices = topk_idx - 1L,
       topk_logits = z[topk_idx], k = k, method = "top-k")
}
