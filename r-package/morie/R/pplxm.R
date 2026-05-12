# SPDX-License-Identifier: GPL-2.0-only

#' Perplexity (Jelinek 1977)
#'
#' @param x Numeric vector of per-token log-probabilities.
#' @param base Character; log base ("e" or "2").
#' @return Named list with value, nll, n, method.
#' @keywords internal
perplexity_metric <- function(x, base = "e") {
  logp <- as.numeric(x)
  if (!length(logp)) stop("Need at least one token log-prob")
  if (identical(base, "2")) logp <- logp * log(2)
  else if (!identical(base, "e")) stop("base must be 'e' or '2'")
  nll <- -mean(logp); ppl <- exp(nll)
  list(value = ppl, nll = nll, n = length(logp),
       method = "perplexity")
}
